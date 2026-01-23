
"""
HOW TO UNDERSTAND THE RETRIVER SET:
    Calibration stage:
        - Non-conformity score: similarity score between question and text in SearchResults (similarity calculated by inner product between embeddings calculated by Contriever), each question only has one non-conformity score in this stage
        for each question in Calibration dataset, calculate a Non-conformity score, get Non-conformity score sets: {s_i}_1^{n_cal},
        calculate a quantile for the Non-conformity score sets, get a threshold at 1-alpha: h=q_(1-alpha)({s_i}_1^{n_cal})
    Prediction stage:
        - for each question, calculate similarity scores between question and each retrieved text in ctxs, for each retrieved text with similarity score higher than h, include them in retriever set for the question.


"""


import os, re, math, json
from typing import List, Dict, Any, Tuple, Iterable
from collections import Counter

import numpy as np
from tqdm import tqdm
from openai import OpenAI

# plotting (matplotlib only)
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

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

client = OpenAI()


def norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_gold_aliases(ex: Dict[str, Any]) -> List[str]:
    ans = ex.get("Answer") or ex.get("answer") or {}
    aliases = []
    if isinstance(ans, dict):
        value = ans.get("Value") if "Value" in ans else ans.get("value")
        if value:
            aliases.append(norm_text(str(value)))
        for a in (ans.get("Aliases") if "Aliases" in ans else (ans.get("aliases") or [])):
            a2 = norm_text(str(a))
            if a2:
                aliases.append(a2)
    out, seen = [], set()
    for a in aliases:
        if a and a not in seen:
            out.append(a)
            seen.add(a)
    return out


def load_retriever_json(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise RuntimeError(f"Retriever JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # JSONL fallback
        rows = []
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
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise RuntimeError("Unsupported JSON format for retriever file.")

def rouge1_f1(a: str, b: str) -> float:
    ta = norm_text(a).split()
    tb = norm_text(b).split()
    if not ta or not tb:
        return 0.0
    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    prec = overlap / max(len(ta), 1)
    rec = overlap / max(len(tb), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)

def is_passage_relevant(passage: str, golds: List[str]) -> bool:
    p = norm_text(passage)
    if not p or not golds:
        return False
    return any(g in p for g in golds)

def quantile(scores: List[float], alpha: float) -> float:
    if not scores:
        return float("inf")
    s = np.sort(np.array(scores, dtype=np.float32))
    n = len(s)
    k = int(math.ceil((n + 1) * (1 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    return float(s[k])


def iter_ctxs(ex: Dict[str, Any], topk: int) -> Iterable[Tuple[str, float]]:
    ctxs = ex.get("ctxs") or []
    if not isinstance(ctxs, list):
        return []
    for i, ctx in enumerate(ctxs[:topk]):
        text = (ctx.get("text") or "").strip()
        if not text:
            continue
        score = ctx.get("score", 0.0)
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = 0.0
        yield text, score_f


def get_question(ex: Dict[str, Any]) -> str:
    return ex.get("Question") or ex.get("question") or ""


# -------------------------
# LLM sampling + clustering (ROUGE>0.7)
# -------------------------
def llm_sample_answers(question: str, passage: str, M: int, model: str) -> List[str]:
    prompt = (
        "Answer the question using ONLY the passage.\n"
        "If the passage does not contain the answer, output exactly: UNKNOWN\n\n"
        f"Question: {question}\n\n"
        f"Passage: {passage}\n\n"
        "Return ONLY a short answer string."
    )
    outs = []
    for _ in range(M):
        r = client.responses.create(
            model=model,
            input=prompt,
            temperature=1.0,
            max_output_tokens=16,  
        )
        a = (r.output_text or "").strip().splitlines()[0].strip()
        a = norm_text(a)
        if a:
            outs.append(a)
    return outs

def cluster_by_rouge(samples: List[str], thr: float = 0.7) -> List[List[str]]:
    clusters: List[List[str]] = []
    for s in samples:
        if s == "unknown":
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

def llm_prediction_set(samples: List[str], tau_llm: float, rouge_thr: float = 0.7) -> List[str]:
    M = max(len(samples), 1)
    clusters = cluster_by_rouge(samples, thr=rouge_thr)
    if not clusters:
        return []
    reps = []
    for c in clusters:
        conf = len(c) / M
        score = -conf
        if score <= tau_llm:
            reps.append(c[0])
    return reps

def llm_best_nonconformity_against_gold(samples: List[str], golds: List[str], rouge_thr: float = 0.7) -> float:
    """
    For plotting + calibration: return best (lowest) nonconformity among clusters
    whose representative matches gold by substring proxy.
    If none matches, return 0.0 (worst-ish for -conf in [-1,0]).
    """
    clusters = cluster_by_rouge(samples, thr=rouge_thr)
    if not clusters:
        return 0.0
    M = max(len(samples), 1)
    best = None
    for c in clusters:
        rep = c[0]
        conf = len(c) / M
        score = -conf
        if any(g in rep or rep in g for g in golds):
            best = score if best is None else min(best, score)
    return best if best is not None else 0.0

def save_single_boxplot(values: List[float], hline: float, title: str, ylabel: str, outpath: str) -> None:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure()
    plt.boxplot(values, vert=True, showfliers=True)
    plt.axhline(hline)  # default color
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks([1], ["calibration"])
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    RETRIEVER_JSON = "triviaqa.json"
    TOPK = 20
    ALPHA_RET = 0.1
    ALPHA_LLM = 0.1

    M = 30  # set 30 to match paper; smaller for cheaper demo
    GEN_MODEL = "gpt-4o-mini"

    N_CAL = 60
    N_TEST = 20

    examples = load_retriever_json(RETRIEVER_JSON)
    if not examples:
        raise RuntimeError(f"No examples found in {RETRIEVER_JSON}")

    n_cal = min(N_CAL, len(examples))
    n_test = min(N_TEST, max(len(examples) - n_cal, 0))
    calib_pool = examples[: max(n_cal * 3, n_cal)]

    calib_scores_ret = []
    calib_examples = []

    for ex in tqdm(calib_pool, desc="Collect calib examples"):
        q = get_question(ex)
        golds = get_gold_aliases(ex)
        if not golds:
            continue

        r_star = None
        for passage, score in iter_ctxs(ex, TOPK):
            if is_passage_relevant(passage, golds):
                r_star = float(score)
                break
        if r_star is None:
            continue

        calib_scores_ret.append(-r_star)  # nonconformity = -<q,p*>
        calib_examples.append((ex, q, golds))
        if len(calib_examples) >= N_CAL:
            break

    tau_ret = quantile(calib_scores_ret, ALPHA_RET)

    calib_scores_llm = []

    for (ex, q, golds) in tqdm(calib_examples, desc="Calibrate LLM"):
        ret_set = [p for p, sc in iter_ctxs(ex, TOPK) if -float(sc) <= tau_ret]
        if not ret_set:
            continue

        passage = ret_set[0]
        samples = llm_sample_answers(q, passage, M=M, model=GEN_MODEL)
        s_llm = llm_best_nonconformity_against_gold(samples, golds, rouge_thr=0.7)
        calib_scores_llm.append(s_llm)

    tau_llm = quantile(calib_scores_llm, ALPHA_LLM)

    
    save_single_boxplot(
        calib_scores_ret,
        hline=tau_ret,
        title="Retriever nonconformity on calibration",
        ylabel="score_ret = - <q, p*>",
        outpath="plots/retrieval_nonconformity_box.png",
    )
    save_single_boxplot(
        calib_scores_llm,
        hline=tau_llm,
        title="LLM nonconformity on calibration",
        ylabel="score_llm = - confidence(best correct cluster)",
        outpath="plots/llm_nonconformity_box.png",
    )

    # ---- test: CAgg(q) = union_{p in CRet(q)} CLLM(q,p) ----
    test = examples[n_cal : n_cal + n_test] if n_test > 0 else []
    covered = 0
    total = 0

    for ex in tqdm(test, desc="Test"):
        q = get_question(ex)
        golds = get_gold_aliases(ex)
        ret_set = [p for p, sc in iter_ctxs(ex, TOPK) if -float(sc) <= tau_ret]

        agg_answers = set()
        for passage in ret_set:
            samples = llm_sample_answers(q, passage, M=M, model=GEN_MODEL)
            ans_set = llm_prediction_set(samples, tau_llm=tau_llm, rouge_thr=0.7)
            for a in ans_set:
                agg_answers.add(a)

        hit = False
        if golds and agg_answers:
            for a in agg_answers:
                if any(g in a or a in g for g in golds):
                    hit = True
                    break

        covered += int(hit)
        total += 1

        print("\nQ:", q)
        print("CAgg:", list(agg_answers)[:10] if agg_answers else ["EMPTY"])
        print("gold aliases (first 5):", golds[:5])

    print("\n---")
    print(f"tau_ret = {tau_ret:.4f} | tau_llm = {tau_llm:.4f}")
    print(f"coverage_proxy = {covered}/{total} = {covered/max(total,1):.3f}")
    print("saved plots:")
    print(" - plots/retrieval_nonconformity_box.png")
    print(" - plots/llm_nonconformity_box.png")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY.")
    main()
