# End-to-end TRAQ-style pipeline (matches paper’s retriever-set construction):
# 1) Retriever NCM: s_i = -R(q_i, p*_i) where p*_i is the annotated most relevant passage.
#    In your data, p* is taken from SearchResults (Rank==0 / first).
# 2) tau_ret = Quantile({s_i}; ceil((N+1)(1-alpha_ret))/N)  (finite-sample conformal quantile)
# 3) Build C_ret(q) over COMBINED candidates = ctxs + top-k SearchResults:
#       C_ret(q) = {p | -R(q,p) <= tau_ret}
# 4) Report:
#    - retrieval coverage = fraction of examples where p* is in C_ret(q)
#    - avg |C_ret(q)| over the dataset
# 5) (Optional) LLM conformal sets (unchanged).

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

    # Try JSON first
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    # Fallback JSONL
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


def write_retrieval_csv(
    path: str,
    rows: List[Dict[str, Any]],
    quantile: float,
    threshold: float,
    guarantee: float,
    avg_c_ret_size: float,
    coverage_hits: int,
    coverage_total: int,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "question",
                "nonconformity",
                "topk_sr",
                "sim_p_star",
                "num_searchresults",
                "quantile",
                "threshold",
                "retriever_guarantee",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.get("question", ""),
                    r.get("nonconformity", ""),
                    r.get("topk_sr", ""),
                    r.get("sim_p_star", ""),
                    r.get("num_searchresults", ""),
                    f"{quantile:.6f}",
                    f"{threshold:.6f}",
                    f"{guarantee:.6f}",
                ]
            )
        writer.writerow(["__AVG_C_RET_SIZE__", f"{avg_c_ret_size:.6f}"])
        coverage_rate = coverage_hits / max(coverage_total, 1)
        writer.writerow(["__RETRIEVAL_COVERAGE__", f"{coverage_rate:.6f}"])
        writer.writerow(["__RETRIEVAL_COVERAGE_HITS__", f"{coverage_hits}/{coverage_total}"])


def write_retriever_set_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "c_ret_passages"])
        for r in rows:
            writer.writerow([r.get("question", ""), r.get("c_ret_passages_json", "")])


def write_prediction_set_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "passage", "prediction_set"])
        for r in rows:
            writer.writerow([r.get("question", ""), r.get("passage", ""), r.get("prediction_set_json", "")])


def load_few_shot_examples(path: Optional[str]) -> List[Dict[str, str]]:
    if not path:
        return []
    rows = load_json_or_jsonl(path)
    out: List[Dict[str, str]] = []
    for ex in rows:
        norm = _normalize_few_shot_example(ex)
        if norm:
            out.append(norm)
    return out



def norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_question(ex: Dict[str, Any]) -> str:
    return ex.get("Question") or ex.get("question") or ""

def get_gold_aliases_norm(ex: Dict[str, Any]) -> List[str]:
    """
    Supports:
      ex["Answer"] = {"Value": ..., "Aliases": [...]}
      ex["answer"] = {"value": ..., "aliases": [...]}
    """
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
    # lenient: substring either way
    return any((g in p) or (p in g) for g in golds_norm)


def llm_example_nonconformity(samples: List[str], golds_norm: List[str], rouge_thr: float) -> float:
    M = max(len(samples), 1)
    clusters = cluster_by_rouge(samples, thr=rouge_thr)

    best: Optional[float] = None
    for c in clusters:
        conf = len(c) / M
        score = -conf
        if any(any_gold_match(x, golds_norm) for x in c):
            best = score if best is None else min(best, score)

    return best if best is not None else 0.0



def conformal_quantile(scores: List[float], alpha: float) -> float:
    """
    tau = Quantile({s_n}; ceil((N+1)(1-alpha))/N)
    Implemented as:
      k = ceil((N+1)*(1-alpha)) - 1   (0-index)
      tau = sorted_scores[k]
    """
    if not scores:
        return float("inf")
    s = np.sort(np.asarray(scores, dtype=np.float32))
    n = len(s)
    k = int(math.ceil((n + 1) * (1 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    return float(s[k])


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
            batch = texts[i:i + batch_size]
            tok = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tok = {k: v.to(self.device) for k, v in tok.items()}
            out = self.model(**tok)
            last = out.last_hidden_state  # [B, T, H]
            mask = tok["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            mean = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            # If you want cosine-sim instead of raw inner product, uncomment:
            # mean = torch.nn.functional.normalize(mean, p=2, dim=1)
            embs.append(mean.detach().cpu().numpy())
        return np.concatenate(embs, axis=0) if embs else np.zeros((0, 1), dtype=np.float32)


def dot_sim(q_emb: np.ndarray, p_embs: np.ndarray) -> np.ndarray:
    return (p_embs @ q_emb.reshape(-1, 1)).reshape(-1)


def searchresult_to_text(sr: Dict[str, Any]) -> str:
    for key in ["text", "passage", "snippet", "contents", "content", "Text", "Passage", "Snippet"]:
        if sr.get(key):
            return str(sr.get(key))
    title = str(sr.get("Title") or sr.get("title") or "")
    desc = str(sr.get("Description") or sr.get("description") or "")
    body = str(sr.get("body") or sr.get("summary") or "")
    return " ".join([p for p in [title, desc, body] if p]).strip()


def get_searchresults(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    srs = ex.get("SearchResults") or ex.get("search_results") or ex.get("searchResults") or []
    return srs if isinstance(srs, list) else []


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
    """
    p* = annotated most relevant SR. For your data, we assume:
      - use SR with smallest Rank if present
      - else SR[0]
    Returns: (p_star_text, p_star_rank_in_list)
    """
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
    if isinstance(sr0, dict):
        txt = searchresult_to_text(sr0).strip()
    else:
        txt = str(sr0).strip()

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
        s = norm_text(s)
        if not s or s == "unknown":
            continue
        placed = False
        for c in clusters:
            if rouge1_f1(s, c[0]) >= thr:
                c.append(s)
                placed = True
                break
        if not placed:
            clusters.append([s])
    return clusters


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


def load_cache(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
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


ZERO_SHOT_TEMPLATE = (
    "Answer the following question based on the given context; Answer the question shortly.\n"
    "Question: {question}\n"
    "Context: {context}\n"
    "Answer:"
)


def _normalize_few_shot_example(ex: Dict[str, Any]) -> Optional[Dict[str, str]]:
    q = ex.get("question") or ex.get("Question")
    c = ex.get("context") or ex.get("Context")
    a = ex.get("answer") or ex.get("Answer")
    if not (q and c and a):
        return None
    return {"question": str(q), "context": str(c), "answer": str(a)}


def build_prompt(question: str, context: str, few_shot: Sequence[Dict[str, str]]) -> str:
    if not few_shot:
        return ZERO_SHOT_TEMPLATE.format(question=question, context=context)

    parts = ["Answer the following question based on the given context; Answer the question shortly."]
    for ex in few_shot:
        parts.append("Question: {question}\nContext: {context}\nAnswer: {answer}".format(**ex))
    parts.append(f"Question: {question}\nContext: {context}\nAnswer:")
    return "\n".join(parts)


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
    few_shot: Sequence[Dict[str, str]],
) -> List[str]:
    msg = build_prompt(question=question, context=passage, few_shot=few_shot)
    outs: List[str] = []
    last_err: Optional[Exception] = None
    for _ in range(M):
        for attempt in range(retries + 1):
            try:
                r = client.responses.create(
                    model=model,
                    input=msg,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                a = (r.output_text or "").strip().splitlines()[0].strip()
                outs.append(a if a else "UNKNOWN")
                break
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(retry_sleep_s * (2 ** attempt))
                    continue
                raise RuntimeError(f"LLM call failed: {last_err}") from last_err
    return outs


def llm_prediction_set(samples: List[str], tau_llm: float, rouge_thr: float = 0.7) -> List[str]:
    M = max(len(samples), 1)
    clusters = cluster_by_rouge(samples, thr=rouge_thr)
    reps: List[str] = []
    for c in clusters:
        conf = len(c) / M
        score = -conf
        if score <= tau_llm:
            reps.append(c[0])
    return reps


def llm_prediction_set_with_scores(
    samples: List[str],
    tau_llm: float,
    rouge_thr: float = 0.7,
) -> List[Tuple[str, float]]:
    M = max(len(samples), 1)
    clusters = cluster_by_rouge(samples, thr=rouge_thr)
    reps: List[Tuple[str, float]] = []
    for c in clusters:
        conf = len(c) / M
        score = -conf
        if score <= tau_llm:
            reps.append((c[0], float(conf)))
    return reps


def weighted_quantile(values: torch.Tensor, q: float, sample_weight: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    q = float(min(max(q, 0.0), 1.0))
    values = values.detach().float().view(-1)
    weights = sample_weight.detach().float().view(-1)
    if weights.numel() != values.numel():
        raise ValueError("sample_weight must have same length as values.")
    sorter = torch.argsort(values)
    v = values[sorter]
    w = weights[sorter].clamp(min=0)
    if float(w.sum()) <= 0:
        w = torch.ones_like(w)
    cdf = torch.cumsum(w, dim=0)
    cdf = (cdf - 0.5 * w) / float(w.sum())
    cdf_np = cdf.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()
    out = float(np.interp(q, cdf_np, v_np))
    return torch.tensor(out, dtype=torch.float32)


def normalized_entropy(prob: Sequence[int]) -> float:
    """
    prob: cluster counts of m sampled responses
    returns NE in [0, 1]
    """
    m = int(sum(prob)) if int(sum(prob)) > 0 else 30
    cnt = prob
    probs = np.array([c / m for c in cnt], dtype=float)
    ent = -np.sum(probs * np.log(probs + 1e-12))
    return float(ent / max(math.log(m), 1e-12))

def compute_refusal_metrics(labels: List[int], rejections: List[bool]) -> Tuple[float, float, float]:
    """
    labels: 1 if answerable, 0 if unanswerable
    rejections: True if LLM refused to answer (i.e., said "Can't answer")

    Returns: (R_refuse, P_refuse, f1)
    """
    if not labels or not rejections or len(labels) != len(rejections):
        return 0.0, 0.0, 0.0

    n = len(labels)
    rcorrect = sum(labels) / n  # fraction of answerable

    q_reject = [i for i, r in enumerate(rejections) if r]
    n_reject = len(q_reject)

    # Recall: among unanswerables, how many got rejected
    num_unans = sum(1 for i in range(n) if labels[i] == 0)
    if num_unans == 0:
        r_refuse = 0.0
    else:
        r_refuse = sum(1 for i in q_reject if labels[i] == 0) / (1 - rcorrect)

    # Precision: among all rejections, how many were correct
    if n_reject == 0:
        p_refuse = 0.0
    else:
        p_refuse = sum(1 for i in q_reject if labels[i] == 0) / n_reject

    denom = max(2 - p_refuse * r_refuse, 1e-6)
    f1 = (p_refuse * r_refuse) / denom if denom else 0.0

    return r_refuse, p_refuse, f1


def MonCP_cal(
    all_logits: torch.Tensor,
    labels: torch.Tensor,
    ratio: float,
    grouped_text: List[List[Tuple[str, float, bool]]],
    score: Sequence[float],
    ne: Sequence[float],
    weight: Sequence[float],
    qs: int = 10,
    use_mlp: bool = False,
    use_weight: bool = False,
) -> Tuple[List[List[Tuple[str, bool]]], List[float], int]:
    yloc = labels.view(-1)
    ratio = float(np.mean(labels.detach().cpu().numpy()))
    ncs = all_logits[:, 1]
    if not use_mlp:
        ncs = 1 - torch.tensor(ne, dtype=torch.float32) + 0.01 * all_logits[:, 1]
    ncs0 = ncs[yloc == 0]
    ncs1 = ncs[yloc == 1]
    qs0s = [i / 10 for i in range(10, qs * 10 + 1)]
    max_len = 1000000000
    m_ans: List[List[Tuple[str, bool]]] = []
    a: List[float] = []
    for qs0 in qs0s:
        lenth = 0
        if ratio <= 1e-12:
            qs1 = 0.0
        else:
            qs1 = (1 - ratio) / ratio * (qs - qs0)
        qs1 = float(min(max(qs1, 0.0), 100.0))
        if use_weight:
            w = torch.tensor(weight, dtype=torch.float32)
            quantile0 = weighted_quantile(ncs0, 1 - qs0 / 100, w[yloc == 0]) if ncs0.numel() > 0 else torch.tensor(0.0)
            quantile1 = (
                1 - weighted_quantile(1 - ncs1, 1 - qs1 / 100, w[yloc == 1]) if ncs1.numel() > 0 else torch.tensor(0.0)
            )
        else:
            quantile0 = torch.quantile(ncs0, 1 - qs0 / 100) if ncs0.numel() > 0 else torch.tensor(0.0)
            quantile1 = torch.quantile(ncs1, qs1 / 100) if ncs1.numel() > 0 else torch.tensor(0.0)
        if qs0 == qs:
            quantile1 = torch.tensor(-100.0)

        Non_scores: List[float] = []
        weight_non: List[float] = []
        for i in range(len(grouped_text)):
            if ncs[i] > quantile0 and ncs[i] > quantile1:
                if labels[i] != 0:
                    Non_scores.append(float(score[i]))
                    weight_non.append(float(weight[i]))
            elif not (ncs[i] < quantile0 and ncs[i] < quantile1):
                if labels[i] == 1:
                    Non_scores.append(float(score[i]))
                    weight_non.append(float(weight[i]))

        if len(Non_scores) == 0:
            text_quantile = float("inf")
        elif not use_weight:
            text_quantile = float(np.quantile(Non_scores, qs / 100))
        else:
            text_quantile = float(
                -weighted_quantile(
                    -torch.tensor(Non_scores, dtype=torch.float32),
                    1 - qs / 100,
                    torch.tensor(weight_non, dtype=torch.float32),
                ).item()
            )

        ans_list: List[List[Tuple[str, bool]]] = []
        for i in range(len(grouped_text)):
            if ncs[i] < quantile0 and ncs[i] < quantile1:
                tmp_ans: List[Tuple[str, bool]] = [("Can't answer", bool(labels[i] == 0))]
                ans_list.append(tmp_ans)
            elif ncs[i] > quantile0 and ncs[i] > quantile1:
                tmp_ans = []
                for ans in grouped_text[i]:
                    if float(ans[1]) >= text_quantile:
                        lenth += 1
                        tmp_ans.append((ans[0], bool(ans[2])))
                ans_list.append(tmp_ans)
            else:
                lenth += 1
                tmp_ans = [("Can't answer", bool(labels[i] == 0))]
                for ans in grouped_text[i]:
                    if float(ans[1]) >= text_quantile:
                        lenth += 1
                        tmp_ans.append((ans[0], bool(ans[2])))
                ans_list.append(tmp_ans)

        if lenth < max_len:
            max_len = lenth
            m_ans = ans_list
            a = [float(quantile0), float(quantile1), float(text_quantile), float(qs0), float(qs1)]

    return m_ans, a, max_len


def MonCP_infer(
    all_logits: torch.Tensor,
    labels: torch.Tensor,
    grouped_text: List[List[Tuple[str, float, bool]]],
    ne: Sequence[float],
    a: Sequence[float],
    use_mlp: bool = False,
) -> List[List[Tuple[str, bool]]]:
    quantile0, quantile1, text_quantile = float(a[0]), float(a[1]), float(a[2])
    ncs = all_logits[:, 1]
    if not use_mlp:
        ncs = 1 - torch.tensor(ne, dtype=torch.float32) + 0.01 * all_logits[:, 1]
    ans_list: List[List[Tuple[str, bool]]] = []
    for i in range(len(grouped_text)):
        if ncs[i] < quantile0 and ncs[i] < quantile1:
            ans_list.append([("Can't answer", bool(labels[i] == 0))])
        elif ncs[i] > quantile0 and ncs[i] > quantile1:
            tmp_ans: List[Tuple[str, bool]] = []
            for ans in grouped_text[i]:
                if float(ans[1]) >= text_quantile:
                    tmp_ans.append((ans[0], bool(ans[2])))
            ans_list.append(tmp_ans)
        else:
            tmp_ans = [("Can't answer", bool(labels[i] == 0))]
            for ans in grouped_text[i]:
                if float(ans[1]) >= text_quantile:
                    tmp_ans.append((ans[0], bool(ans[2])))
            ans_list.append(tmp_ans)
    return ans_list


def empirical_coverage_curve(
    cal_scores: Sequence[float],
    eval_scores: Sequence[float],
    alpha_grid: Sequence[float],
) -> Tuple[List[float], List[float]]:
    expected: List[float] = []
    empirical: List[float] = []
    eval_arr = np.asarray(eval_scores, dtype=float)
    for alpha in alpha_grid:
        tau = conformal_quantile(list(cal_scores), alpha=alpha)
        exp = 1.0 - float(alpha)
        emp = float(np.mean(eval_arr <= tau)) if eval_arr.size > 0 else 0.0
        expected.append(exp)
        empirical.append(emp)
    return expected, empirical


def save_coverage_plot(
    expected: Sequence[float],
    empirical: Sequence[float],
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



def run(args: argparse.Namespace) -> None:
    rows = load_json_or_jsonl(args.input_path)
    if not rows:
        raise RuntimeError(f"No examples found in {args.input_path}")
    if args.max_rows is not None:
        rows = rows[: max(int(args.max_rows), 0)]
    if not rows:
        raise RuntimeError("No examples left after applying --max_rows.")

    n = len(rows)
    n_cal = n if args.n_cal is None else min(args.n_cal, n)
    calib = rows[:n_cal]  # shares dict objects with rows

    enc = ContrieverEncoder(args.contriever_model, device=args.device)

    # ---- 1) Calibration: s_i = -R(q_i, p*_i) using SearchResults p* ----
    calib_scores: List[float] = []
    calib_rows: List[Dict[str, Any]] = []
    retr_eval_scores: List[float] = []
    eval_start = n_cal if n_cal < n else 0

    printed_nonconformity = 0
    for ex in tqdm(calib, desc="Calibrate retrieval NCMs (p* from SearchResults)"):
        q = get_question(ex)
        if not q:
            continue

        p_star_text, _ = pick_p_star_from_searchresults(ex)
        if not p_star_text:
            continue

        q_emb = enc.encode([q], batch_size=1, max_length=args.max_length)[0]
        p_emb = enc.encode([p_star_text], batch_size=1, max_length=args.max_length)[0]
        sim_p_star = float(dot_sim(q_emb, p_emb.reshape(1, -1))[0])

        # NCM per paper: s_i = -R(q, p*)
        s_i = -sim_p_star
        calib_scores.append(s_i)
        calib_rows.append(
            {
                "question": q,
                "nonconformity": s_i,
                "topk_sr": args.topk_sr,
                "sim_p_star": sim_p_star,
                "num_searchresults": len(get_searchresults(ex)),
            }
        )

        if printed_nonconformity < 5:
            print(f"[Retriever NCM sample {printed_nonconformity + 1}] s_i={s_i:.6f}  (sim={sim_p_star:.6f})")
            printed_nonconformity += 1

    if not calib_scores:
        raise RuntimeError("No calibration scores computed. Check that SearchResults exist and have text.")

    tau_ret = conformal_quantile(calib_scores, alpha=args.alpha_ret)
    print(f"\nRetriever threshold tau_ret (quantile of NCMs): {tau_ret:.6f}")
    print(f"Target guarantee: 1 - alpha_ret = {1 - args.alpha_ret:.3f}")

    # ---- 2) Build C_ret on COMBINED candidates: ctxs + top-k SearchResults ----
    annotated: List[Dict[str, Any]] = []
    retriever_rows: List[Dict[str, Any]] = []

    coverage_hits = 0
    coverage_total = 0
    c_ret_sizes: List[int] = []

    for ex_idx, ex in enumerate(tqdm(rows, desc="Build C_ret on (ctxs + SearchResults)")):
        q = get_question(ex)
        if not q:
            annotated.append(ex)
            continue

        # original ctxs
        ctxs = ex.get("ctxs") or []
        if not isinstance(ctxs, list):
            ctxs = []

        # top-k SearchResults as additional candidate passages
        sr_texts_topk = get_searchresults_texts(ex, args.topk_sr)
        sr_entries = [{"text": t, "source": "searchresult"} for t in sr_texts_topk if (t or "").strip()]

        # mark p* within SR list (for coverage check)
        p_star_text, _ = pick_p_star_from_searchresults(ex)
        p_star_text_norm = norm_text(p_star_text) if p_star_text else ""
        p_star_idx_in_combined: Optional[int] = None

        combined: List[Dict[str, Any]] = []
        # keep only topk_ctx ctxs (paper uses retriever pool; your choice here)
        for c in ctxs[:args.topk_ctx]:
            if isinstance(c, dict):
                combined.append(c)
        base_len = len(combined)

        for j, sr in enumerate(sr_entries):
            if p_star_text_norm and norm_text(sr.get("text", "")) == p_star_text_norm and p_star_idx_in_combined is None:
                p_star_idx_in_combined = base_len + j
            combined.append(sr)

        if not combined:
            ex["h_retrieval"] = tau_ret
            ex["C_ret_size"] = 0
            ex["p_star_in_C_ret"] = False
            annotated.append(ex)
            continue

        # encode all candidates
        texts: List[str] = []
        valid_indices: List[int] = []
        for i, item in enumerate(combined):
            t = (item.get("text") or "").strip()
            if t:
                texts.append(t)
                valid_indices.append(i)

        if not texts:
            ex["h_retrieval"] = tau_ret
            ex["C_ret_size"] = 0
            ex["p_star_in_C_ret"] = False
            ex["ctxs"] = combined
            annotated.append(ex)
            continue

        q_emb = enc.encode([q], batch_size=1, max_length=args.max_length)[0]
        p_embs = enc.encode(texts, batch_size=args.batch_size, max_length=args.max_length)
        sims = dot_sim(q_emb, p_embs)

        c_ret_size = 0
        for local_k, idx in enumerate(valid_indices):
            sim = float(sims[local_k])
            combined[idx]["contriever_sim"] = sim
            # paper rule: include if -R(q,p) <= tau_ret
            in_set = bool((-sim) <= tau_ret)
            combined[idx]["in_C_ret"] = in_set
            if in_set:
                c_ret_size += 1

        ex["ctxs"] = combined
        ex["h_retrieval"] = tau_ret
        ex["C_ret_size"] = c_ret_size

        # Coverage: is p* in C_ret?
        p_star_in_set = False
        if p_star_text and p_star_idx_in_combined is not None:
            coverage_total += 1
            p_star_in_set = bool(combined[p_star_idx_in_combined].get("in_C_ret") is True)
            p_star_sim = float(combined[p_star_idx_in_combined].get("contriever_sim", 0.0))
            if ex_idx >= eval_start:
                retr_eval_scores.append(-p_star_sim)
            if p_star_in_set:
                coverage_hits += 1
        ex["p_star_in_C_ret"] = p_star_in_set

        # size stats
        c_ret_sizes.append(c_ret_size)

        # write retriever set passages for debugging/export
        c_ret_passages = [
            (it.get("text") or "").strip()
            for it in combined
            if it.get("in_C_ret") is True and (it.get("text") or "").strip()
        ]
        retriever_rows.append({"question": q, "c_ret_passages_json": json.dumps(c_ret_passages, ensure_ascii=False)})

        annotated.append(ex)

    # ---- 3) Report & write CSVs ----
    avg_c_ret_size = float(np.mean(c_ret_sizes)) if c_ret_sizes else 0.0
    coverage_rate = coverage_hits / max(coverage_total, 1)

    print(f"\nAvg |C_ret(q)| over dataset: {avg_c_ret_size:.3f}")
    print(f"Retriever coverage (p* in C_ret): {coverage_hits}/{coverage_total} = {coverage_rate:.3f}")

    write_retriever_set_csv(args.retriever_set_csv_path, retriever_rows)
    print(f"Saved retriever set CSV: {args.retriever_set_csv_path}")

    write_retrieval_csv(
        args.retrieval_csv_path,
        calib_rows,
        quantile=tau_ret,
        threshold=tau_ret,
        guarantee=1 - args.alpha_ret,
        avg_c_ret_size=avg_c_ret_size,
        coverage_hits=coverage_hits,
        coverage_total=coverage_total,
    )
    print(f"Saved retriever NCM CSV: {args.retrieval_csv_path}")

    if args.plot_coverage:
        alpha_grid = np.linspace(0.01, 0.5, num=max(args.coverage_plot_points, 2))
        retr_expected, retr_empirical = empirical_coverage_curve(calib_scores, retr_eval_scores, alpha_grid)
        save_coverage_plot(
            retr_expected,
            retr_empirical,
            title="Retriever: Expected vs Empirical Coverage",
            path=args.retriever_coverage_plot_path,
        )
        print(f"Saved retriever coverage plot: {args.retriever_coverage_plot_path}")

    # ---- Optional LLM stage (same as before; uses ex['ctxs'] which is now combined) ----
    if args.enable_llm:
        if OpenAI is None:
            raise RuntimeError("openai package not installed but --enable_llm was set.")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")

        few_shot_examples = load_few_shot_examples(args.few_shot_path)
        client = OpenAI()
        llm_cache = load_cache(args.llm_cache_path)

        # (A) Calibrate tau_llm using calibration examples with gold answers and p* (from SearchResults)
        llm_cal_scores: List[float] = []
        printed_llm_cal = 0

        for ex in tqdm(calib, desc="Calibrate LLM tau_llm on (q, p*)"):
            q = get_question(ex)
            if not q:
                continue

            golds = get_gold_aliases_norm(ex)
            if not golds:
                continue

            p_star_text, _ = pick_p_star_from_searchresults(ex)
            if not p_star_text:
                continue

            key = cache_key_llm(q, p_star_text, args.M, args.llm_model)
            if key in llm_cache:
                samples = llm_cache[key]["samples"]
            else:
                samples = llm_sample_answers(
                    client=client,
                    model=args.llm_model,
                    question=q,
                    passage=p_star_text,
                    M=args.M,
                    temperature=args.llm_temperature,
                    max_output_tokens=args.max_output_tokens,
                    retries=args.retries,
                    retry_sleep_s=args.retry_sleep_s,
                    few_shot=few_shot_examples,
                )
                llm_cache[key] = {"samples": samples}

            s_i_llm = llm_example_nonconformity(samples, golds_norm=golds, rouge_thr=args.rouge_thr)
            llm_cal_scores.append(s_i_llm)

            if printed_llm_cal < 3:
                print("\n[LLM calibration sample]")
                print("Q:", q)
                print("p*:", p_star_text[:250] + ("..." if len(p_star_text) > 250 else ""))
                print("NCM s_i_llm:", s_i_llm)
                printed_llm_cal += 1

        if not llm_cal_scores:
            raise RuntimeError(
                "No LLM calibration scores computed. Make sure calibration examples contain Answer/answer with value+aliases."
            )

        tau_llm = conformal_quantile(llm_cal_scores, alpha=args.alpha_llm)
        print(f"\nLLM threshold tau_llm (quantile of LLM NCMs): {tau_llm:.6f}")
        print(f"Target LLM guarantee: 1 - alpha_llm = {1 - args.alpha_llm:.3f}")

        llm_eval_scores: List[float] = []
        if args.plot_coverage:
            eval_rows = rows[eval_start:] if eval_start > 0 else rows
            for ex in tqdm(eval_rows, desc="Collect LLM eval NCMs for coverage plot"):
                q = get_question(ex)
                golds = get_gold_aliases_norm(ex)
                p_star_text, _ = pick_p_star_from_searchresults(ex)
                if not (q and golds and p_star_text):
                    continue
                key = cache_key_llm(q, p_star_text, args.M, args.llm_model)
                if key in llm_cache:
                    samples = llm_cache[key]["samples"]
                else:
                    samples = llm_sample_answers(
                        client=client,
                        model=args.llm_model,
                        question=q,
                        passage=p_star_text,
                        M=args.M,
                        temperature=args.llm_temperature,
                        max_output_tokens=args.max_output_tokens,
                        retries=args.retries,
                        retry_sleep_s=args.retry_sleep_s,
                        few_shot=few_shot_examples,
                    )
                    llm_cache[key] = {"samples": samples}
                llm_eval_scores.append(llm_example_nonconformity(samples, golds_norm=golds, rouge_thr=args.rouge_thr))

        # (B) Build C_llm(q,p) for each p in C_ret(q), then C_agg(q)=union_p C_llm(q,p) and recluster
        pred_rows: List[Dict[str, Any]] = []
        printed_llm_pred = 0

        agg_sizes: List[int] = []
        agg_hits = 0
        agg_total = 0

        moncp_records: List[Dict[str, Any]] = []
        for row_idx, ex in enumerate(tqdm(annotated, desc="Build C_llm and C_agg")):
            q = get_question(ex)
            if not q:
                continue

            ctxs = ex.get("ctxs") or []
            if not isinstance(ctxs, list) or not ctxs:
                ex["C_agg"] = []
                ex["hit_gold"] = False
                continue

            golds = get_gold_aliases_norm(ex)  # optional for evaluation
            agg_total += 1 if golds else 0

            agg_pool: List[str] = []
            grouped_entry: List[Tuple[str, float, bool]] = []
            best_conf = 0.0
            best_ne = 1.0
            score_i = 0.0
            any_sample_hit = False

            for ctx in ctxs:
                if ctx.get("in_C_ret") is not True:
                    continue
                passage = (ctx.get("text") or "").strip()
                if not passage:
                    continue

                key = cache_key_llm(q, passage, args.M, args.llm_model)
                if key in llm_cache:
                    samples = llm_cache[key]["samples"]
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
                        few_shot=few_shot_examples,
                    )
                    llm_cache[key] = {"samples": samples}

                # Build MonCP features from raw sample clusters.
                clusters = cluster_by_rouge(samples, thr=args.rouge_thr)
                if clusters:
                    m = max(len(samples), 1)
                    c_sizes = [len(c) for c in clusters]
                    conf = float(max(c_sizes) / m)
                    ne_ctx = normalized_entropy(c_sizes)
                    if conf >= best_conf:
                        best_conf = conf
                        best_ne = ne_ctx
                if golds:
                    if any(any_gold_match(s, golds) for s in samples):
                        any_sample_hit = True

                # Paper-style conformal LLM prediction set for this (q,p)
                pred_set = llm_prediction_set(samples, tau_llm=tau_llm, rouge_thr=args.rouge_thr)
                ctx["llm_pred_set"] = pred_set

                if printed_llm_pred < 3:
                    print("\n[LLM pred-set sample]")
                    print("Q:", q)
                    print("P:", passage[:250] + ("..." if len(passage) > 250 else ""))
                    print("Pred set:", pred_set)
                    printed_llm_pred += 1

                pred_rows.append(
                    {
                        "question": q,
                        "passage": passage[:5000],
                        "prediction_set_json": json.dumps(pred_set, ensure_ascii=False),
                    }
                )
                pred_with_scores = llm_prediction_set_with_scores(samples, tau_llm=tau_llm, rouge_thr=args.rouge_thr)
                for ans_text, conf in pred_with_scores:
                    gt = bool(golds and any_gold_match(ans_text, golds))
                    grouped_entry.append((ans_text, float(conf), gt))
                    score_i = max(score_i, float(conf))

                agg_pool.extend(pred_set)

            # C_agg(q) = union over p in C_ret(q) of C_llm(q,p)
            # then recluster/dedup (paper does semantic dedup; we approximate with ROUGE clustering)
            uniq = []
            seen = set()
            for a in agg_pool:
                na = norm_text(a)
                if na and na not in seen:
                    uniq.append(na)
                    seen.add(na)

            agg_clusters = cluster_by_rouge(uniq, thr=args.rouge_thr)
            C_agg = [c[0] for c in agg_clusters]

            ex["C_agg"] = C_agg
            agg_sizes.append(len(C_agg))

            hit = bool(golds and any(any_gold_match(a, golds) for a in C_agg))
            ex["hit_gold"] = hit
            if golds and hit:
                agg_hits += 1
            if args.use_rejection_module:
                moncp_records.append(
                    {
                        "row_idx": row_idx,
                        "logit": float(best_conf),
                        "label": int(1 if any_sample_hit else 0),
                        "grouped_text": grouped_entry,
                        "score": float(score_i),
                        "ne": float(best_ne),
                        "weight": 1.0,
                    }
                )

        if args.use_rejection_module:
            cal_recs = [r for r in moncp_records if r["row_idx"] < n_cal]
            if not cal_recs:
                raise RuntimeError("Rejection module requested but no calibration records are available for MonCP.")

            cal_logits = torch.tensor([[1 - r["logit"], r["logit"]] for r in cal_recs], dtype=torch.float32)
            cal_labels = torch.tensor([r["label"] for r in cal_recs], dtype=torch.long)
            cal_grouped = [r["grouped_text"] for r in cal_recs]
            cal_score = [r["score"] for r in cal_recs]
            cal_ne = [r["ne"] for r in cal_recs]
            cal_weight = [r["weight"] for r in cal_recs]
            qs = int(round(float(args.alpha_llm) * 100))
            _, moncp_a, _ = MonCP_cal(
                all_logits=cal_logits,
                labels=cal_labels,
                ratio=float(np.mean(cal_labels.detach().cpu().numpy())),
                grouped_text=cal_grouped,
                score=cal_score,
                ne=cal_ne,
                weight=cal_weight,
                qs=max(qs, 1),
                use_mlp=args.moncp_mlp,
                use_weight=args.moncp_weight,
            )

            all_logits_m = torch.tensor([[1 - r["logit"], r["logit"]] for r in moncp_records], dtype=torch.float32)
            all_labels_m = torch.tensor([r["label"] for r in moncp_records], dtype=torch.long)
            all_grouped_m = [r["grouped_text"] for r in moncp_records]
            all_ne_m = [r["ne"] for r in moncp_records]
            moncp_ans = MonCP_infer(
                all_logits=all_logits_m,
                labels=all_labels_m,
                grouped_text=all_grouped_m,
                ne=all_ne_m,
                a=moncp_a,
                use_mlp=args.moncp_mlp,
            )

            for rec, ans_list in zip(moncp_records, moncp_ans):
                ex = annotated[rec["row_idx"]]
                dedup: List[str] = []
                seen = set()
                for ans, _ in ans_list:
                    na = norm_text(ans)
                    if not na:
                        continue
                    if na in seen:
                        continue
                    dedup.append(ans if ans == "Can't answer" else na)
                    seen.add(na)
                ex["C_agg"] = dedup
                ex["llm_rejection_mode"] = "moncp"
                ex["llm_rejection_params"] = moncp_a

            # GET THE PREC, REC, F1 metrics for refusals
            rejections = []
            labels_true = []
            for rec in moncp_records:
                ex = annotated[rec["row_idx"]]
                c_agg = ex.get("C_agg") or []
                labels_true.append(rec["label"])  # 1 = answerable, 0 = unanswerable
                said_cant = any("can't answer" in (a or "").lower() for a in c_agg)
                rejections.append(said_cant)

            r_refuse, p_refuse, f1_refuse = compute_refusal_metrics(labels_true, rejections)
            print(f"\n[LLM Rejection Metrics]")
            print(f"Recall (R_refuse): {r_refuse:.3f}")
            print(f"Precision (P_refuse): {p_refuse:.3f}")
            print(f"F1: {f1_refuse:.3f}")


            # Recompute metrics after MonCP overwrite.
            agg_sizes = []
            agg_hits = 0
            agg_total = 0
            for rec in moncp_records:
                ex = annotated[rec["row_idx"]]
                golds = get_gold_aliases_norm(ex)
                c_agg = ex.get("C_agg") or []
                agg_sizes.append(len(c_agg))
                if golds:
                    agg_total += 1
                    hit = bool(any(any_gold_match(a, golds) for a in c_agg))
                    ex["hit_gold"] = hit
                    if hit:
                        agg_hits += 1
                else:
                    ex["hit_gold"] = False

        # persist + export
        save_cache(args.llm_cache_path, llm_cache)
        write_prediction_set_csv(args.prediction_set_csv_path, pred_rows)
        print(f"Saved prediction set CSV: {args.prediction_set_csv_path}")

        if agg_sizes:
            print(f"\nAvg |C_agg(q)| over dataset: {float(np.mean(agg_sizes)):.3f}")
        if agg_total > 0:
            print(f"Answer-set coverage (hit_gold via aliases): {agg_hits}/{agg_total} = {agg_hits / agg_total:.3f}")

        if args.plot_coverage:
            alpha_grid = np.linspace(0.01, 0.5, num=max(args.coverage_plot_points, 2))
            llm_expected, llm_empirical = empirical_coverage_curve(llm_cal_scores, llm_eval_scores, alpha_grid)
            save_coverage_plot(
                llm_expected,
                llm_empirical,
                title="LLM: Expected vs Empirical Coverage",
                path=args.llm_coverage_plot_path,
            )
            print(f"Saved LLM coverage plot: {args.llm_coverage_plot_path}")

    # ---- Write annotated output ----
    write_jsonl(args.output_path, annotated)
    print(f"\nSaved annotated JSONL: {args.output_path}")
    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("TRAQ retriever-set (Contriever + conformal tau_ret) + optional LLM sets")
    p.add_argument("--input_path", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--max_rows", type=int, default=300, help="Use only the first N rows from input")

    p.add_argument("--n_cal", type=int, default=None, help="Use this many examples for calibration (default: all)")
    p.add_argument("--alpha_ret", type=float, default=0.1)
    p.add_argument("--topk_sr", type=int, default=None, help="How many SearchResults to include as candidates")
    p.add_argument("--topk_ctx", type=int, default=None, help="How many ctxs to include as candidates")

    p.add_argument("--contriever_model", type=str, default="facebook/contriever-msmarco")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=256)

    p.add_argument("--enable_llm", action="store_true")
    p.add_argument("--M", type=int, default=15)
    p.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    p.add_argument("--llm_temperature", type=float, default=1.0)
    p.add_argument("--max_output_tokens", type=int, default=16)
    p.add_argument("--rouge_thr", type=float, default=0.7)
    p.add_argument("--few_shot_path", type=str, default=None)

    p.add_argument("--retrieval_csv_path", type=str, default="retrieval_nonconformity.csv")
    p.add_argument("--retriever_set_csv_path", type=str, default="retriever_set.csv")
    p.add_argument("--prediction_set_csv_path", type=str, default="prediction_set.csv")

    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry_sleep_s", type=float, default=0.5)
    p.add_argument("--llm_cache_path", type=str, default="cache_llm_samples.json")
    p.add_argument("--alpha_llm", type=float, default=0.1, help="LLM conformal miscoverage (Bonferroni split with alpha_ret if desired)")
    p.add_argument("--use_rejection_module", action="store_true", help="Enable LLM rejection module in prediction sets")
    p.add_argument("--moncp_mlp", action="store_true", help="Use raw logits for MonCP (skip 1-NE adjustment)")
    p.add_argument("--moncp_weight", action="store_true", help="Use weighted quantiles in MonCP")
    p.add_argument("--plot_coverage", action="store_true", help="Plot expected vs empirical coverage curves")
    p.add_argument("--coverage_plot_points", type=int, default=20, help="Number of alpha grid points for coverage plots")
    p.add_argument("--retriever_coverage_plot_path", type=str, default="retriever_coverage_plot.png")
    p.add_argument("--llm_coverage_plot_path", type=str, default="llm_coverage_plot.png")

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
