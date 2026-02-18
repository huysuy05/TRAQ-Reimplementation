# run.py
# TRAQ retriever-set + Adaptive Rejection CP (Sec 4.2) integrated end-to-end.
#
# Key points:
# - Build C_ret(q) via conformal retriever threshold tau_ret using Contriever similarities.
# - For each question:
#   - Sample M_rej times from LLM WITHOUT passages (question-only) => clusters => NE => p_cant=NE, p_can=1-NE.
#   - Sample M times from LLM WITH each passage in C_ret(q) => answer clusters and confidences.
#   - Cluster scores for answers = (frequency/confidence) - NE (paper: "frequency minus NE").
# - Calibrate rejection thresholds (q_cant, q_can) label-conditionally with alpha0/alpha1 coupled by Eq. 12.
# - Recompute text threshold on retained answerables (Eq. 15) and grid search (Eq. 16).
# - Inference:
#   - If (p_can < q_can) and (p_cant > q_cant): REJECT => prediction set = ["Can't answer"]
#   - Elif (p_can < q_can) and (p_cant <= q_cant): ADD "Can't answer" to set (coverage guard)
#   - Else: standard CP answer set
#


import os
import re
import json
import math
import time
import argparse
import csv
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# -------------------------
# IO utils
# -------------------------
def load_env_from_dotenv(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_env_from_dotenv(_ENV_PATH)


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON on line {i}: {e}")
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: str, header: List[str], rows: List[List[Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_question(ex: Dict[str, Any]) -> str:
    return ex.get("Question") or ex.get("question") or ""


def get_gold_aliases_norm(ex: Dict[str, Any]) -> List[str]:
    ans = ex.get("Answer") or ex.get("answer") or {}
    golds: List[str] = []
    if isinstance(ans, dict):
        val = ans.get("Value") if "Value" in ans else ans.get("value")
        if val:
            golds.append(str(val))
        als = ans.get("Aliases") if "Aliases" in ans else (ans.get("aliases") or [])
        for a in als or []:
            if a:
                golds.append(str(a))
    out: List[str] = []
    seen = set()
    for g in golds:
        ng = norm_text(g)
        if ng and ng not in seen:
            out.append(ng)
            seen.add(ng)
    return out


def any_gold_match(pred: str, golds_norm: List[str]) -> bool:
    p = norm_text(pred)
    if not p:
        return False
    return any((g in p) or (p in g) for g in golds_norm)


def get_searchresults(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    srs = ex.get("SearchResults") or ex.get("search_results") or ex.get("searchResults") or []
    return srs if isinstance(srs, list) else []


def searchresult_to_text(sr: Dict[str, Any]) -> str:
    for key in ["text", "passage", "snippet", "contents", "content", "Text", "Passage", "Snippet"]:
        if sr.get(key):
            return str(sr.get(key))
    title = str(sr.get("Title") or sr.get("title") or "")
    desc = str(sr.get("Description") or sr.get("description") or "")
    body = str(sr.get("body") or sr.get("summary") or "")
    return " ".join([p for p in [title, desc, body] if p]).strip()


def get_searchresults_texts(ex: Dict[str, Any], topk_sr: int) -> List[str]:
    srs = get_searchresults(ex)
    out: List[str] = []
    for sr in srs[:topk_sr]:
        if isinstance(sr, dict):
            t = searchresult_to_text(sr).strip()
            if t:
                out.append(t)
        elif isinstance(sr, str):
            t = sr.strip()
            if t:
                out.append(t)
    return out


def pick_p_star_from_searchresults(ex: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    srs = get_searchresults(ex)
    if not srs:
        return None, None
    best_idx = 0
    best_rank = None
    for i, sr in enumerate(srs):
        if not isinstance(sr, dict):
            continue
        if "Rank" in sr and sr["Rank"] is not None:
            try:
                r = int(sr["Rank"])
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best_idx = i
            except Exception:
                pass
    sr0 = srs[best_idx]
    txt = searchresult_to_text(sr0).strip() if isinstance(sr0, dict) else str(sr0).strip()
    return (txt if txt else None), best_idx


def rouge1_f1(a: str, b: str) -> float:
    ta = norm_text(a).split()
    tb = norm_text(b).split()
    if not ta or not tb:
        return 0.0
    ca = {}
    cb = {}
    for w in ta:
        ca[w] = ca.get(w, 0) + 1
    for w in tb:
        cb[w] = cb.get(w, 0) + 1
    overlap = 0
    for w in ca:
        overlap += min(ca.get(w, 0), cb.get(w, 0))
    if overlap == 0:
        return 0.0
    prec = overlap / max(len(ta), 1)
    rec = overlap / max(len(tb), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def cluster_by_rouge(samples: List[str], thr: float = 0.7) -> List[List[str]]:
    clusters: List[List[str]] = []
    for s in samples:
        s = (s or "").strip()
        if not s:
            continue
        s_norm = norm_text(s)
        if not s_norm or s_norm == "unknown":
            continue
        placed = False
        for c in clusters:
            if rouge1_f1(s_norm, c[0]) >= thr:
                c.append(s_norm)
                placed = True
                break
        if not placed:
            clusters.append([s_norm])
    return clusters


def normalized_entropy_from_cluster_sizes(cluster_sizes: List[int]) -> float:
    # Paper NE in [0,1]. Use standard normalized entropy over cluster probabilities.
    K = len(cluster_sizes)
    if K <= 1:
        return 0.0
    m = float(sum(cluster_sizes))
    if m <= 0:
        return 0.0
    ps = np.array([c / m for c in cluster_sizes], dtype=float)
    ent = -float(np.sum(ps * np.log(ps + 1e-12)))
    return float(ent / max(math.log(K), 1e-12))


def conformal_quantile(values: List[float], alpha: float) -> float:
    """
    Quantile({v_i}; ceil((n+1)*(1-alpha))/n)
    """
    if not values:
        return float("inf")
    v = np.sort(np.asarray(values, dtype=np.float32))
    n = len(v)
    k = int(math.ceil((n + 1) * (1 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    return float(v[k])


def refusal_metrics_prob(labels_ans: List[int], rejections: List[bool]) -> Tuple[float, float, float]:
    """
    labels_ans[i]=1 => answerable (ground truth / heuristic)
    labels_ans[i]=0 => unanswerable
    rejections[i]=True => predicted reject ("Can't answer" only-set or contains can't-answer)
    Returns: (R_refuse, P_refuse, f1_paper)
    """
    n = len(labels_ans)
    if n == 0:
        return 0.0, 0.0, 0.0

    p_unans = sum(1 for y in labels_ans if y == 0) / n
    p_reject = sum(1 for r in rejections if r) / n
    p_joint = sum(1 for y, r in zip(labels_ans, rejections) if (y == 0 and r)) / n  # P(reject ∧ unanswerable)

    r_refuse = (p_joint / p_unans) if p_unans > 0 else 0.0
    p_refuse = (p_joint / p_reject) if p_reject > 0 else 0.0

    denom = max(2.0 - p_refuse * r_refuse, 1e-12)
    f1_paper = (p_refuse * r_refuse) / denom
    return float(r_refuse), float(p_refuse), float(f1_paper)


def refusal_metrics_components(labels_ans: List[int], rejections: List[bool]) -> Dict[str, float]:
    n = len(labels_ans)
    if n == 0:
        return {
            "p_joint": 0.0,
            "p_unans": 0.0,
            "p_reject": 0.0,
            "r_refuse": 0.0,
            "p_refuse": 0.0,
            "f1_paper": 0.0,
        }
    p_unans = sum(1 for y in labels_ans if y == 0) / n
    p_reject = sum(1 for r in rejections if r) / n
    p_joint = sum(1 for y, r in zip(labels_ans, rejections) if (y == 0 and r)) / n
    r_refuse = (p_joint / p_unans) if p_unans > 0 else 0.0
    p_refuse = (p_joint / p_reject) if p_reject > 0 else 0.0
    denom = max(2.0 - p_refuse * r_refuse, 1e-12)
    f1_paper = (p_refuse * r_refuse) / denom
    return {
        "p_joint": float(p_joint),
        "p_unans": float(p_unans),
        "p_reject": float(p_reject),
        "r_refuse": float(r_refuse),
        "p_refuse": float(p_refuse),
        "f1_paper": float(f1_paper),
    }


def print_ne_quantiles(ne_vals: List[float], tag: str) -> None:
    if not ne_vals:
        print(f"[NE Quantiles] {tag}: (empty)")
        return
    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    arr = np.asarray(ne_vals, dtype=float)
    vals = np.quantile(arr, qs)
    parts = [f"p{int(q * 100):02d}={v:.3f}" for q, v in zip(qs, vals)]
    print(f"[NE Quantiles] {tag}: " + "  ".join(parts))


def empirical_coverage_curve(
    cal_scores: List[float],
    eval_scores: List[float],
    alpha_grid: np.ndarray,
) -> Tuple[List[float], List[float]]:
    expected: List[float] = []
    empirical: List[float] = []
    eval_arr = np.asarray(eval_scores, dtype=float)
    for alpha in alpha_grid:
        tau = conformal_quantile(list(cal_scores), alpha=float(alpha))
        expected.append(1.0 - float(alpha))
        emp = float(np.mean(eval_arr <= tau)) if eval_arr.size > 0 else 0.0
        empirical.append(emp)
    return expected, empirical


def save_coverage_plot(
    expected: List[float],
    empirical: List[float],
    title: str,
    path: str,
) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot(expected, empirical, marker="o", linewidth=1.5, label="Empirical")
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, label="Ideal y=x")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Expected Coverage (1 - alpha)")
    plt.ylabel("Empirical Coverage")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


class ContrieverEncoder:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 32, max_length: int = 256) -> np.ndarray:
        embs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tok = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tok = {k: v.to(self.device) for k, v in tok.items()}
            out = self.model(**tok)
            last = out.last_hidden_state
            mask = tok["attention_mask"].unsqueeze(-1)
            mean = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embs.append(mean.detach().cpu().numpy())
        return np.concatenate(embs, axis=0) if embs else np.zeros((0, 1), dtype=np.float32)


def dot_sim(q_emb: np.ndarray, p_embs: np.ndarray) -> np.ndarray:
    return (p_embs @ q_emb.reshape(-1, 1)).reshape(-1)


ZERO_SHOT_TEMPLATE = (
    "Answer the following question based on the given context; Answer the question shortly.\n"
    "Question: {question}\n"
    "Context: {context}\n"
    "Answer:"
)

QUESTION_ONLY_TEMPLATE = (
    "Answer the following question shortly. If the question cannot be answered confidently, say Can't answer.\n"
    "Question: {question}\n"
    "Answer:"
)


def cache_key_llm(question: str, passage: str, M: int, model: str) -> str:
    h = hashlib.sha256()
    h.update(norm_text(question).encode("utf-8"))
    h.update(b"||")
    h.update(norm_text(passage)[:4000].encode("utf-8"))
    h.update(b"||")
    h.update(str(M).encode("utf-8"))
    h.update(b"||")
    h.update(model.encode("utf-8"))
    return h.hexdigest()


def cache_key_llm_question_only(question: str, M: int, model: str) -> str:
    h = hashlib.sha256()
    h.update(b"q_only||")
    h.update(norm_text(question).encode("utf-8"))
    h.update(b"||")
    h.update(str(M).encode("utf-8"))
    h.update(b"||")
    h.update(model.encode("utf-8"))
    return h.hexdigest()


def load_cache(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def save_cache(path: Optional[str], cache: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def llm_call_text(
    client: "OpenAI",
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    retries: int,
    retry_sleep_s: float,
) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            txt = (r.output_text or "").strip()
            if not txt:
                return "UNKNOWN"
            return txt.splitlines()[0].strip()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_sleep_s * (2**attempt))
                continue
            raise RuntimeError(f"LLM call failed: {last_err}") from last_err
    return "UNKNOWN"


def llm_sample_answers(
    client: "OpenAI",
    model: str,
    question: str,
    passage: str,
    M: int,
    temperature: float,
    max_output_tokens: int,
    retries: int,
    retry_sleep_s: float,
) -> List[str]:
    prompt = ZERO_SHOT_TEMPLATE.format(question=question, context=passage)
    return [
        llm_call_text(client, model, prompt, temperature, max_output_tokens, retries, retry_sleep_s)
        for _ in range(M)
    ]


def llm_sample_answers_question_only(
    client: "OpenAI",
    model: str,
    question: str,
    M: int,
    temperature: float,
    max_output_tokens: int,
    retries: int,
    retry_sleep_s: float,
) -> List[str]:
    prompt = QUESTION_ONLY_TEMPLATE.format(question=question)
    return [
        llm_call_text(client, model, prompt, temperature, max_output_tokens, retries, retry_sleep_s)
        for _ in range(M)
    ]


def alpha1_from_alpha0(alpha: float, alpha0: float, rcorrect: float) -> float:
    """
    Eq. 12:
      alpha1 = ((1 - rcorrect) * (alpha - alpha0)) / (rcorrect * (1 - alpha))
    """
    if rcorrect <= 1e-12 or alpha >= 1.0:
        return 0.0
    num = (1.0 - rcorrect) * (alpha - alpha0)
    den = rcorrect * (1.0 - alpha)
    a1 = num / max(den, 1e-12)
    return float(min(max(a1, 0.0), 1.0))


def decide_rejection_mode(p_can: float, p_cant: float, q_can: float, q_cant: float) -> str:
    """
    Based on paper wording (semantic):
    - Reject when "low probability of answering" AND "high probability of being unanswerable"
      => p_can < q_can and p_cant > q_cant
    - Add "Can't answer" label for coverage guard when p_can < q_can and p_cant <= q_cant
    - Otherwise standard CP
    """
    if (p_can < q_can) and (p_cant > q_cant):
        return "reject"
    if (p_can < q_can) and (p_cant <= q_cant):
        return "add_cant"
    return "standard"


# -------------------------
# Main
# -------------------------
def run(args: argparse.Namespace) -> None:
    rows = load_json_or_jsonl(args.input_path)
    if not rows:
        raise RuntimeError(f"No examples found in {args.input_path}")
    if args.max_rows is not None:
        rows = rows[: max(int(args.max_rows), 0)]
    if not rows:
        raise RuntimeError("No examples left after applying --max_rows.")

    n = len(rows)
    n_cal = n if args.n_cal is None else min(int(args.n_cal), n)
    calib = rows[:n_cal]
    eval_start = n_cal if n_cal < n else 0
    retriever_set_cap = 20

    # ---- Retriever calibration (TRAQ retriever set) ----
    enc = ContrieverEncoder(args.contriever_model, device=args.device)
    calib_scores: List[float] = []
    retr_eval_scores: List[float] = []
    for ex in tqdm(calib, desc="Calibrate retriever NCMs (p* from SearchResults)"):
        q = get_question(ex)
        p_star_text, _ = pick_p_star_from_searchresults(ex)
        if not q or not p_star_text:
            continue
        q_emb = enc.encode([q], batch_size=1, max_length=args.max_length)[0]
        p_emb = enc.encode([p_star_text], batch_size=1, max_length=args.max_length)[0]
        sim = float(dot_sim(q_emb, p_emb.reshape(1, -1))[0])
        calib_scores.append(-sim)  # s_i = -R(q,p*)
    if not calib_scores:
        raise RuntimeError("No retrieval calibration scores computed.")
    tau_ret = conformal_quantile(calib_scores, alpha=args.alpha_ret)
    print(f"\nRetriever tau_ret: {tau_ret:.6f} (target 1-alpha_ret={1-args.alpha_ret:.3f})")

    # ---- Build C_ret(q) over combined ctxs + SearchResults ----
    annotated: List[Dict[str, Any]] = []
    c_ret_sizes: List[int] = []
    coverage_hits = 0
    coverage_total = 0

    for ex_idx, ex in enumerate(tqdm(rows, desc="Build C_ret(q)")):
        q = get_question(ex)
        if not q:
            annotated.append(ex)
            continue

        ctxs = ex.get("ctxs") or []
        if not isinstance(ctxs, list):
            ctxs = []

        sr_texts = get_searchresults_texts(ex, args.topk_sr)
        sr_entries = [{"text": t, "source": "searchresult"} for t in sr_texts if (t or "").strip()]

        combined: List[Dict[str, Any]] = []
        for c in ctxs[: args.topk_ctx]:
            if isinstance(c, dict):
                combined.append(c)
        base_len = len(combined)
        p_star_text, _ = pick_p_star_from_searchresults(ex)
        p_star_norm = norm_text(p_star_text) if p_star_text else ""
        p_star_idx_in_combined: Optional[int] = None

        for j, sr in enumerate(sr_entries):
            if p_star_norm and norm_text(sr.get("text", "")) == p_star_norm and p_star_idx_in_combined is None:
                p_star_idx_in_combined = base_len + j
            combined.append(sr)

        if not combined:
            ex["ctxs"] = []
            ex["C_ret_size"] = 0
            ex["p_star_in_C_ret"] = False
            annotated.append(ex)
            continue

        texts: List[str] = []
        valid_idx: List[int] = []
        for i, it in enumerate(combined):
            t = (it.get("text") or "").strip()
            if t:
                texts.append(t)
                valid_idx.append(i)
        if not texts:
            ex["ctxs"] = combined
            ex["C_ret_size"] = 0
            ex["p_star_in_C_ret"] = False
            annotated.append(ex)
            continue

        q_emb = enc.encode([q], batch_size=1, max_length=args.max_length)[0]
        p_embs = enc.encode(texts, batch_size=args.batch_size, max_length=args.max_length)
        sims = dot_sim(q_emb, p_embs)

        in_set_indices: List[int] = []
        for local_k, idx in enumerate(valid_idx):
            sim = float(sims[local_k])
            combined[idx]["contriever_sim"] = sim
            in_set = bool((-sim) <= tau_ret)
            combined[idx]["in_C_ret"] = in_set
            if in_set:
                in_set_indices.append(idx)

        # hard cap retriever set size to 20
        if len(in_set_indices) > retriever_set_cap:
            in_set_indices.sort(key=lambda i: float(combined[i].get("contriever_sim", float("-inf"))), reverse=True)
            keep = set(in_set_indices[:retriever_set_cap])
            for i in in_set_indices[retriever_set_cap:]:
                combined[i]["in_C_ret"] = False
            in_set_indices = sorted(list(keep))

        ex["ctxs"] = combined
        ex["C_ret_size"] = len(in_set_indices)
        c_ret_sizes.append(len(in_set_indices))

        # coverage check
        p_star_in = False
        if p_star_text and p_star_idx_in_combined is not None:
            coverage_total += 1
            p_star_in = bool(combined[p_star_idx_in_combined].get("in_C_ret") is True)
            p_star_sim = float(combined[p_star_idx_in_combined].get("contriever_sim", 0.0))
            if ex_idx >= eval_start:
                retr_eval_scores.append(-p_star_sim)
            if p_star_in:
                coverage_hits += 1
        ex["p_star_in_C_ret"] = p_star_in

        annotated.append(ex)

    print(f"\nAvg |C_ret|: {float(np.mean(c_ret_sizes)) if c_ret_sizes else 0.0:.3f}")
    print(f"Retriever coverage p* in C_ret: {coverage_hits}/{coverage_total} = {coverage_hits/max(coverage_total,1):.3f}")

    if args.plot_coverage:
        alpha_grid = np.linspace(0.01, 0.5, num=max(int(args.coverage_plot_points), 2))
        retr_expected, retr_empirical = empirical_coverage_curve(calib_scores, retr_eval_scores, alpha_grid)
        save_coverage_plot(
            retr_expected,
            retr_empirical,
            title="Retriever: Expected vs Empirical Coverage",
            path=args.retriever_coverage_plot_path,
        )
        print(f"Saved retriever coverage plot: {args.retriever_coverage_plot_path}")

    # ---- LLM stage required for Adaptive Rejection CP ----
    if not args.enable_llm:
        write_jsonl(args.output_path, annotated)
        print(f"\nSaved annotated JSONL: {args.output_path}")
        print("Done (retriever-only).")
        return

    if OpenAI is None:
        raise RuntimeError("openai package not installed but --enable_llm was set.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    cache = load_cache(args.llm_cache_path)

    # ---- Precompute question-only samples => NE => p_can/p_cant, and heuristic answerable label ----
    # label_i = 1 if ANY question-only sample matches gold (paper heuristic)
    rej_feat: List[Dict[str, Any]] = []
    for row_idx, ex in enumerate(tqdm(annotated, desc="Question-only sampling for rejection features")):
        q = get_question(ex)
        golds = get_gold_aliases_norm(ex)
        if not q or not golds:
            ex["rej_p_can"] = None
            ex["rej_p_cant"] = None
            ex["rej_NE"] = None
            ex["rej_label_ans"] = None
            continue

        key = cache_key_llm_question_only(q, args.M_rej, args.llm_model)
        if key in cache:
            q_only_samples = cache[key]["samples"]
        else:
            q_only_samples = llm_sample_answers_question_only(
                client=client,
                model=args.llm_model,
                question=q,
                M=args.M_rej,
                temperature=args.llm_temperature,
                max_output_tokens=args.max_output_tokens,
                retries=args.retries,
                retry_sleep_s=args.retry_sleep_s,
            )
            cache[key] = {"samples": q_only_samples}

        clusters = cluster_by_rouge(q_only_samples, thr=args.rouge_thr)
        sizes = [len(c) for c in clusters] if clusters else [1]
        NE = normalized_entropy_from_cluster_sizes(sizes)
        p_cant = float(NE)
        p_can = float(1.0 - NE)

        label_ans = 1 if any(any_gold_match(s, golds) for s in q_only_samples) else 0

        ex["rej_NE"] = NE
        ex["rej_p_cant"] = p_cant
        ex["rej_p_can"] = p_can
        ex["rej_label_ans"] = label_ans
        rej_feat.append({"row_idx": row_idx, "p_can": p_can, "p_cant": p_cant, "label": label_ans})

    # ---- Passage-conditioned sampling for answer candidates (TRAQ answer-set part) ----
    # We aggregate answer clusters across passages in C_ret(q) by taking max confidence per normalized answer string.
    answer_cands: Dict[int, Dict[str, float]] = {}  # row_idx -> {answer_norm: conf_max}
    for row_idx, ex in enumerate(tqdm(annotated, desc="Passage-conditioned sampling for answer candidates")):
        q = get_question(ex)
        if not q:
            continue
        ctxs = ex.get("ctxs") or []
        if not isinstance(ctxs, list):
            continue

        cand_map: Dict[str, float] = {}
        for ctx in ctxs:
            if ctx.get("in_C_ret") is not True:
                continue
            passage = (ctx.get("text") or "").strip()
            if not passage:
                continue

            key = cache_key_llm(q, passage, args.M, args.llm_model)
            if key in cache:
                samples = cache[key]["samples"]
            else:
                samples = llm_sample_answers(
                    client=client,
                    model=args.llm_model,
                    question=q,
                    passage=passage,
                    M=args.M,
                    temperature=args.llm_temperature,
                    max_output_tokens=args.max_output_tokens,
                    retries=args.retries,
                    retry_sleep_s=args.retry_sleep_s,
                )
                cache[key] = {"samples": samples}

            clusters = cluster_by_rouge(samples, thr=args.rouge_thr)
            m = max(len(samples), 1)
            for c in clusters:
                rep = c[0]
                conf = float(len(c) / m)
                repn = norm_text(rep)
                if not repn:
                    continue
                cand_map[repn] = max(cand_map.get(repn, 0.0), conf)

        answer_cands[row_idx] = cand_map

    # ---- Adaptive Rejection CP calibration + grid search ----
    # Use calibration subset only (row_idx < n_cal).
    cal_items = [rf for rf in rej_feat if rf["row_idx"] < n_cal]
    if not cal_items:
        raise RuntimeError("No calibration items available for rejection module (need gold answers).")

    cal_ne_all = [float(annotated[it["row_idx"]]["rej_NE"]) for it in cal_items]
    cal_ne_ans = [float(annotated[it["row_idx"]]["rej_NE"]) for it in cal_items if int(it["label"]) == 1]
    cal_ne_unans = [float(annotated[it["row_idx"]]["rej_NE"]) for it in cal_items if int(it["label"]) == 0]

    print_ne_quantiles(cal_ne_all, "calibration (all)")
    print_ne_quantiles(cal_ne_ans, "calibration (answerable=1)")
    print_ne_quantiles(cal_ne_unans, "calibration (unanswerable=0)")

    rcorrect = float(np.mean([it["label"] for it in cal_items]))
    alpha = float(args.alpha_cp)

    # Prepare label-conditional lists for quantile computations
    p_cant_unans = [it["p_cant"] for it in cal_items if it["label"] == 0]
    p_can_ans = [it["p_can"] for it in cal_items if it["label"] == 1]

    if len(p_cant_unans) == 0 or len(p_can_ans) == 0:
        raise RuntimeError(
            "Rejection CP needs both answerable and unanswerable in calibration by the paper heuristic.\n"
            "If you're only using TriviaQA, you won't get unanswerables unless you constructed them."
        )

    # Grid search over alpha0, derive alpha1 via Eq. 12, then compute thresholds and text quantile (Eq. 15),
    # and evaluate expected set size (Eq. 16).
    alpha0_grid = np.linspace(args.alpha0_min, min(args.alpha0_max, alpha), num=args.alpha0_grid).tolist()

    best = {"avg_size": float("inf"), "alpha0": None, "alpha1": None, "q_cant": None, "q_can": None, "tau_text": None}
    best_s_hat: List[float] = []

    # Precompute per-calibration answer candidate lists + NE
    # cluster_score = conf - NE
    cal_answer_struct: Dict[int, Dict[str, Any]] = {}
    for it in cal_items:
        ridx = int(it["row_idx"])
        ex = annotated[ridx]
        NE = float(ex.get("rej_NE") or 0.0)
        cand_map = answer_cands.get(ridx, {})
        # Convert to list of (ans, score)
        cand_scores = [(a, float(conf) - NE) for a, conf in cand_map.items()]
        cal_answer_struct[ridx] = {"NE": NE, "cand_scores": cand_scores, "golds": get_gold_aliases_norm(ex)}

    for alpha0 in alpha0_grid:
        alpha1 = alpha1_from_alpha0(alpha=alpha, alpha0=float(alpha0), rcorrect=rcorrect)
        # Quantiles (Eq. 13/14 semantics):
        q_cant = conformal_quantile(p_cant_unans, alpha=float(alpha0))
        q_can = conformal_quantile(p_can_ans, alpha=float(alpha1))

        # Determine rejection modes on calibration
        modes: Dict[int, str] = {}
        for it in cal_items:
            ridx = int(it["row_idx"])
            ex = annotated[ridx]
            p_can = float(ex["rej_p_can"])
            p_cant = float(ex["rej_p_cant"])
            modes[ridx] = decide_rejection_mode(p_can, p_cant, q_can=q_can, q_cant=q_cant)

        # Eq. 15: recompute text nonconformity on retained answerables (true label=1, not rejected)
        # Nonconformity for correct answer cluster: s_hat = -(max(score_correct)) where score_correct = conf - NE
        s_hat: List[float] = []
        for it in cal_items:
            ridx = int(it["row_idx"])
            if int(it["label"]) != 1:
                continue
            if modes.get(ridx) == "reject":
                continue
            golds = cal_answer_struct[ridx]["golds"]
            cand_scores = cal_answer_struct[ridx]["cand_scores"]
            best_score = None
            for ans, sc in cand_scores:
                if any_gold_match(ans, golds):
                    best_score = sc if best_score is None else max(best_score, sc)
            if best_score is None:
                # worst nonconformity if we didn't produce a correct cluster
                s_hat.append(1.0)
            else:
                s_hat.append(float(-best_score))

        tau_text = conformal_quantile(s_hat, alpha=alpha) if s_hat else float("inf")

        # Evaluate expected set size on calibration (Eq. 16)
        sizes: List[int] = []
        for it in cal_items:
            ridx = int(it["row_idx"])
            ex = annotated[ridx]
            NE = float(ex.get("rej_NE") or 0.0)
            p_can = float(ex["rej_p_can"])
            p_cant = float(ex["rej_p_cant"])
            mode = decide_rejection_mode(p_can, p_cant, q_can=q_can, q_cant=q_cant)

            cand_map = answer_cands.get(ridx, {})
            # include answers whose nonconformity <= tau_text, i.e., -(conf-NE) <= tau_text
            ans_set = []
            for a, conf in cand_map.items():
                sc = float(conf) - NE
                if (-sc) <= tau_text:
                    ans_set.append(a)

            if mode == "reject":
                sizes.append(1)
            elif mode == "add_cant":
                sizes.append(len(set(ans_set)) + 1)
            else:
                sizes.append(len(set(ans_set)))

        avg_size = float(np.mean(sizes)) if sizes else float("inf")
        if avg_size < best["avg_size"]:
            best = {
                "avg_size": avg_size,
                "alpha0": float(alpha0),
                "alpha1": float(alpha1),
                "q_cant": float(q_cant),
                "q_can": float(q_can),
                "tau_text": float(tau_text),
            }
            best_s_hat = list(s_hat)

    print("\n[Adaptive Rejection CP - Selected Params]")
    print(f"rcorrect (cal): {rcorrect:.3f}")
    print(f"alpha (global CP): {alpha:.3f}")
    print(f"alpha0*: {best['alpha0']:.4f}  alpha1*: {best['alpha1']:.4f}")
    print(f"q_cant*: {best['q_cant']:.4f}  q_can*: {best['q_can']:.4f}")
    print(f"tau_text*: {best['tau_text']:.4f}")
    print(f"Avg set size on cal (efficiency): {best['avg_size']:.3f}")

    q_cant_star = float(best["q_cant"])
    q_can_star = float(best["q_can"])
    tau_text_star = float(best["tau_text"])

    # ---- Inference: build final C_agg with rejection CP + TRAQ answers ----
    refusal_flags: List[bool] = []
    labels_for_refusal: List[int] = []

    avg_set_sizes_all: List[int] = []
    hit_gold_all = 0
    total_gold_all = 0

    for row_idx, ex in enumerate(tqdm(annotated, desc="Inference: build C_agg with rejection CP")):
        q = get_question(ex)
        golds = get_gold_aliases_norm(ex)
        if not q or not golds or ex.get("rej_p_can") is None:
            ex["C_agg"] = ex.get("C_agg") or []
            continue

        NE = float(ex.get("rej_NE") or 0.0)
        p_can = float(ex["rej_p_can"])
        p_cant = float(ex["rej_p_cant"])
        mode = decide_rejection_mode(p_can, p_cant, q_can=q_can_star, q_cant=q_cant_star)
        ex["rej_mode"] = mode
        ex["rej_q_cant"] = q_cant_star
        ex["rej_q_can"] = q_can_star
        ex["rej_tau_text"] = tau_text_star

        cand_map = answer_cands.get(row_idx, {})
        ans_set = []
        for a, conf in cand_map.items():
            sc = float(conf) - NE
            if (-sc) <= tau_text_star:
                ans_set.append(a)

        if mode == "reject":
            C_agg = ["Can't answer"]
        elif mode == "add_cant":
            C_agg = ["Can't answer"] + sorted(list(set(ans_set)))
        else:
            C_agg = sorted(list(set(ans_set)))

        ex["C_agg"] = C_agg

        # refusal eval bookkeeping (only for examples with heuristic label present)
        if ex.get("rej_label_ans") is not None:
            labels_for_refusal.append(int(ex["rej_label_ans"]))
            said_cant = any("can't answer" in str(a).lower() for a in C_agg)
            refusal_flags.append(bool(said_cant))

        # coverage bookkeeping (treat "Can't answer" as correct when label=0, else need gold hit)
        if golds:
            total_gold_all += 1
            label = int(ex.get("rej_label_ans") or 0)
            if label == 0:
                hit = any("can't answer" in str(a).lower() for a in C_agg)
            else:
                hit = any(any_gold_match(a, golds) for a in C_agg)
            ex["hit_gold"] = bool(hit)
            if hit:
                hit_gold_all += 1

        avg_set_sizes_all.append(len(C_agg))

    # refusal metrics
    r_refuse, p_refuse, f1_refuse = refusal_metrics_prob(labels_for_refusal, refusal_flags)
    refs = refusal_metrics_components(labels_for_refusal, refusal_flags)
    print("\n[LLM Rejection Metrics]")
    print(f"P(reject ∧ unanswerable): {refs['p_joint']:.3f}")
    print(f"P(unanswerable): {refs['p_unans']:.3f}")
    print(f"P(reject): {refs['p_reject']:.3f}")
    print(f"Recall (R_refuse): {r_refuse:.3f}")
    print(f"Precision (P_refuse): {p_refuse:.3f}")
    print(f"F1 (paper): {f1_refuse:.3f}")

    if avg_set_sizes_all:
        print(f"\nAvg |C_agg| (all): {float(np.mean(avg_set_sizes_all)):.3f}")
    if total_gold_all > 0:
        print(f"Coverage proxy (hit_gold / total): {hit_gold_all}/{total_gold_all} = {hit_gold_all/total_gold_all:.3f}")

    if args.plot_coverage:
        llm_eval_scores: List[float] = []
        for row_idx, ex in enumerate(annotated):
            if row_idx < eval_start:
                continue
            if ex.get("rej_p_can") is None:
                continue
            if int(ex.get("rej_label_ans") or 0) != 1:
                continue
            mode = decide_rejection_mode(
                float(ex["rej_p_can"]),
                float(ex["rej_p_cant"]),
                q_can=q_can_star,
                q_cant=q_cant_star,
            )
            if mode == "reject":
                continue
            golds = get_gold_aliases_norm(ex)
            if not golds:
                continue
            NE = float(ex.get("rej_NE") or 0.0)
            cand_map = answer_cands.get(row_idx, {})
            best_score = None
            for ans, conf in cand_map.items():
                if any_gold_match(ans, golds):
                    sc = float(conf) - NE
                    best_score = sc if best_score is None else max(best_score, sc)
            llm_eval_scores.append(1.0 if best_score is None else float(-best_score))

        if best_s_hat and llm_eval_scores:
            alpha_grid = np.linspace(0.01, 0.5, num=max(int(args.coverage_plot_points), 2))
            llm_expected, llm_empirical = empirical_coverage_curve(best_s_hat, llm_eval_scores, alpha_grid)
            save_coverage_plot(
                llm_expected,
                llm_empirical,
                title="LLM (text score): Expected vs Empirical Coverage",
                path=args.llm_coverage_plot_path,
            )
            print(f"Saved LLM coverage plot: {args.llm_coverage_plot_path}")
        else:
            print("Skipped LLM coverage plot (insufficient calibration/eval scores).")

    # Save cache + outputs
    save_cache(args.llm_cache_path, cache)
    write_jsonl(args.output_path, annotated)
    print(f"\nSaved annotated JSONL: {args.output_path}")

    # Export small CSV summary
    if args.summary_csv_path:
        rows_out = []
        for ex in annotated:
            rows_out.append([
                get_question(ex),
                ex.get("rej_NE"),
                ex.get("rej_p_can"),
                ex.get("rej_p_cant"),
                ex.get("rej_mode"),
                json.dumps(ex.get("C_agg") or [], ensure_ascii=False),
                ex.get("hit_gold"),
            ])
        write_csv(
            args.summary_csv_path,
            ["question", "NE", "p_can", "p_cant", "rej_mode", "C_agg", "hit_gold"],
            rows_out,
        )
        print(f"Saved summary CSV: {args.summary_csv_path}")

    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("TRAQ + Adaptive Rejection CP (Sec 4.2)")
    p.add_argument("--input_path", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--max_rows", type=int, default=400)

    # calibration split
    p.add_argument("--n_cal", type=int, default=200)

    # retriever params
    p.add_argument("--alpha_ret", type=float, default=0.1)
    p.add_argument("--topk_sr", type=int, default=10)
    p.add_argument("--topk_ctx", type=int, default=10)
    p.add_argument("--max_c_ret_size", type=int, default=20)

    p.add_argument("--contriever_model", type=str, default="facebook/contriever-msmarco")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=256)

    # LLM params
    p.add_argument("--enable_llm", action="store_true")
    p.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    p.add_argument("--llm_temperature", type=float, default=1.0)
    p.add_argument("--max_output_tokens", type=int, default=16)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry_sleep_s", type=float, default=0.5)
    p.add_argument("--llm_cache_path", type=str, default="cache_llm_samples.json")
    p.add_argument("--rouge_thr", type=float, default=0.7)

    # sampling
    p.add_argument("--M", type=int, default=15, help="samples per (q,passage) for answers")
    p.add_argument("--M_rej", type=int, default=5, help="samples per question-only for rejection")

    # Adaptive Rejection CP
    p.add_argument("--alpha_cp", type=float, default=0.1, help="global CP level alpha (Sec 4.2 uses same alpha)")
    p.add_argument("--alpha0_min", type=float, default=0.01)
    p.add_argument("--alpha0_max", type=float, default=0.2)
    p.add_argument("--alpha0_grid", type=int, default=20)

    # outputs
    p.add_argument("--summary_csv_path", type=str, default="summary_rejection_cp.csv")
    p.add_argument("--plot_coverage", action="store_true", help="Plot expected vs empirical coverage curves")
    p.add_argument("--coverage_plot_points", type=int, default=30, help="Number of alpha grid points for coverage plots")
    p.add_argument("--retriever_coverage_plot_path", type=str, default="out/retriever_coverage.png")
    p.add_argument("--llm_coverage_plot_path", type=str, default="out/llm_coverage_plot.png")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
