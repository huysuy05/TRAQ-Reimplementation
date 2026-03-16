import argparse
import hashlib
import json
import math
import os
import re
import string
import random
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from itertools import product as itertools_product
from tqdm import tqdm
from openai import OpenAI
from elasticsearch import Elasticsearch


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


# ══════════════════════════════════════════════════════════════════════
#  OpenAI Client
# ══════════════════════════════════════════════════════════════════════
class OpenAIClient:
    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        self.model = model
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str,
        n: int = 1,
        max_tokens: int = 16,
        temperature: float = 2.0,
        top_p: float = 0.95,
        frequency_penalty: float = 0.8,
        presence_penalty: float = 0.8,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop or None,
        )
        return [(c.message.content or "").strip() for c in resp.choices]

    def generate_greedy(self, prompt: str, max_tokens: int = 128,
                        stop: Optional[List[str]] = None) -> str:
        return self.generate(
            prompt,
            n=1,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
        )[0]


# ══════════════════════════════════════════════════════════════════════
#  Retriever Client
# ══════════════════════════════════════════════════════════════════════
class RetrieverClient:
    def __init__(self, es_url: str, es_index: str,
                 title_field: str = "title", text_field: str = "text",
                 username: Optional[str] = None,
                 password: Optional[str] = None):
        self.es_url = es_url.rstrip("/")
        self.es_index = es_index
        self.title_field = title_field
        self.text_field = text_field
        if username and password:
            self.es = Elasticsearch(self.es_url, basic_auth=(username, password))
        else:
            self.es = Elasticsearch(self.es_url)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": [self.title_field, self.text_field],
                    "type": "best_fields",
                }
            },
            "size": top_k,
        }
        resp = self.es.search(index=self.es_index, **body)
        hits = resp.get("hits", {}).get("hits", [])
        results: List[Dict] = []
        for h in hits:
            src = h.get("_source", {}) or {}
            title = str(src.get(self.title_field) or "").strip()
            txt = str(src.get(self.text_field) or "").strip()
            if not txt:
                continue
            results.append({
                "id": str(h.get("_id") or src.get("id") or ""),
                "title": title,
                "txt": txt,
                "score": float(h.get("_score") or 0.0),
            })
        return results

    def health(self) -> bool:
        try:
            return bool(self.es.ping())
        except Exception:
            return False


# ══════════════════════════════════════════════════════════════════════
#  Caching wrappers for LLM and Retriever
# ══════════════════════════════════════════════════════════════════════
def _cache_key(prefix: str, **kwargs) -> str:
    raw = json.dumps(kwargs, sort_keys=True)
    return f"{prefix}:" + hashlib.sha256(raw.encode()).hexdigest()


class CachedVLLMClient:
    def __init__(self, client: OpenAIClient, cache: Dict):
        self.client = client
        self.cache = cache
        self.model = client.model

    def generate(
        self,
        prompt: str,
        n: int = 1,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> List[str]:
        key = _cache_key(
            "llm",
            prompt=prompt,
            n=n,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        if key not in self.cache:
            kwargs = {}
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p
            if frequency_penalty is not None:
                kwargs["frequency_penalty"] = frequency_penalty
            if presence_penalty is not None:
                kwargs["presence_penalty"] = presence_penalty
            self.cache[key] = self.client.generate(
                prompt=prompt,
                n=n,
                stop=stop,
                **kwargs,
            )
        return self.cache[key]

    def generate_greedy(self, prompt: str, max_tokens: int = 128,
                        stop: Optional[List[str]] = None) -> str:
        return self.generate(
            prompt=prompt,
            n=1,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
        )[0]


class CachedRetrieverClient:
    def __init__(self, client: RetrieverClient, cache: Dict):
        self.client = client
        self.cache = cache

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        key = _cache_key("ret", query=query, top_k=top_k)
        if key not in self.cache:
            self.cache[key] = self.client.search(query, top_k)
        return self.cache[key]

    def health(self) -> bool:
        return self.client.health()


def load_cache(path: str) -> Dict:
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict, path: str):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, ensure_ascii=False)
    print(f"  Cache saved to: {path} ({len(cache)} entries)")


# ══════════════════════════════════════════════════════════════════════
#  Evaluation helpers (EM, F1) & answer normalization
# ══════════════════════════════════════════════════════════════════════
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = num_same / len(gold_tokens) if gold_tokens else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


# ══════════════════════════════════════════════════════════════════════
#  Answer parsing & passage formatting
# ══════════════════════════════════════════════════════════════════════
def parse_answer(text: str) -> Optional[str]:
    text = text.replace("</Step>", "").strip()
    for pattern in [
        r'[Tt]he answer is\s*:?\s*(.+?)(?:\.\s*$|$)',
        r'[Aa]nswer\s*:?\s*(.+?)(?:\.\s*$|$)',
    ]:
        match = re.search(pattern, text)
        if match:
            answer = match.group(1).strip().strip('"').strip("'").rstrip(".")
            if answer:
                return answer
    return None


def format_passages(passages: List[Dict]) -> str:
    if not passages:
        return ""
    lines = ["Retrieved Evidence:"]
    for i, p in enumerate(passages, 1):
        lines.append(f"[{i}] ({p['title']}) {p['txt']}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
#  Prompt templates
# ══════════════════════════════════════════════════════════════════════
DIRECT_ANSWER_PROMPT = "Question: {question}\nThe answer is"

DIRECT_ANSWER_WITH_CONTEXT_PROMPT = "{context} Question: {question}\n\nThe answer is"


# ══════════════════════════════════════════════════════════════════════
#  Transformer-based Answer Clustering
# ══════════════════════════════════════════════════════════════════════
class AnswerClusterer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 sim_threshold: float = 0.9):
        self.model = SentenceTransformer(model_name)
        self.sim_threshold = sim_threshold

    def cluster(self, answers: List[str]) -> Dict[int, List[str]]:
        if len(answers) <= 1:
            return {0: answers} if answers else {}
        embeddings = self.model.encode(answers, convert_to_numpy=True)
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - self.sim_threshold,
            metric="cosine", linkage="average",
        ).fit_predict(embeddings)
        clusters = defaultdict(list)
        for ans, label in zip(answers, labels):
            clusters[int(label)].append(ans)
        return dict(clusters)


# ══════════════════════════════════════════════════════════════════════
#  Normalized Entropy (NE) computation
# ══════════════════════════════════════════════════════════════════════
def compute_ne(answers: List[str], clusterer: AnswerClusterer
               ) -> Tuple[float, str, Dict[int, List[str]]]:
    """
    Returns: (ne_score, majority_answer, clusters)
    NE=0 if 1 cluster, NE=1 if no answers, else entropy/log(total).
    """
    if not answers:
        return 1.0, "", {}
    clusters = clusterer.cluster(answers)
    if not clusters:
        return 1.0, "", {}

    largest_cluster = max(clusters.values(), key=len)
    majority = Counter(largest_cluster).most_common(1)[0][0]

    if len(clusters) == 1:
        return 0.0, majority, clusters

    total = sum(len(v) for v in clusters.values())
    entropy = -sum((len(v) / total) * math.log(len(v) / total)
                   for v in clusters.values() if len(v) > 0)
    return entropy / math.log(total), majority, clusters


def sample_and_compute_ne(llm, prompt: str, n_samples: int,
                          clusterer: AnswerClusterer
                          ) -> Tuple[float, str, List[str], List[str], Dict]:
    """
    Sample n answers, parse, compute NE.
    Returns: (ne, majority_answer, raw_samples, parsed_answers, clusters)
    """
    raw_samples = llm.generate(
        prompt,
        n=n_samples,
        stop=["\n", "\n\n", "Question:", "Example"],
    )
    parsed = []
    for s in raw_samples:
        answer = s.strip().rstrip(".")
        extracted = parse_answer(f"The answer is {answer}")
        parsed.append(extracted or answer or "")

    valid = [a for a in parsed if a.strip()]
    if not valid:
        return 1.0, "", raw_samples, parsed, {}
    ne, majority, clusters = compute_ne(valid, clusterer)
    return ne, majority, raw_samples, parsed, clusters


# ══════════════════════════════════════════════════════════════════════
#  Gold-matching helpers
# ══════════════════════════════════════════════════════════════════════
def gold_in_samples(gold_answer: str, parsed_answers: List[str]) -> bool:
    gold_norm = normalize_answer(gold_answer)
    if not gold_norm:
        return False
    return any(gold_norm in normalize_answer(a) for a in parsed_answers if a)


def gold_freq_in_group(gold_answer: str, group: List[str]) -> float:
    gold_norm = normalize_answer(gold_answer)
    if not gold_norm:
        return 0.0
    valid = [a for a in group if a and a.strip()]
    if not valid:
        return 0.0
    return sum(1 for a in valid if gold_norm in normalize_answer(a)) / len(valid)


def get_gold_titles(rec: Dict) -> set:
    sf = rec.get("supporting_facts", {})
    titles = sf.get("title", []) if isinstance(sf, dict) else []
    return {t.strip().lower() for t in titles if t}


def count_gold_docs(passages: List[Dict], gold_titles: set) -> int:
    return sum(1 for p in passages
               if p.get("title", "").strip().lower() in gold_titles)


def gold_doc_scores(passages: List[Dict], gold_titles: set) -> List[float]:
    return [p.get("score", 0.0) for p in passages
            if p.get("title", "").strip().lower() in gold_titles]


# ══════════════════════════════════════════════════════════════════════
#  Data Loading & Splitting
# ══════════════════════════════════════════════════════════════════════
def load_data(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            records.append({
                "qid": r["qid"], "question": r["question"],
                "gold_answer": r["gold_answer"],
                "supporting_facts": r["supporting_facts"],
                "reasoning_steps": r.get("reasoning_steps", []),
            })
    return records


def split_data(records: List[Dict], cal_size: int = 300, seed: int = 42
               ) -> Tuple[List[Dict], List[Dict]]:
    shuffled = records.copy()
    random.seed(seed)
    random.shuffle(shuffled)
    return shuffled[:cal_size], shuffled[cal_size:]


# ══════════════════════════════════════════════════════════════════════
#  Conformal quantile
# ══════════════════════════════════════════════════════════════════════
def conformal_quantile(scores: List[float], alpha: float) -> float:
    if not scores:
        return 1.0
    n = len(scores)
    q_index = max(1, min(math.floor((n - 1) * alpha), n))
    return sorted(scores)[q_index - 1]
# def conformal_quantile(scores: List[float], alpha: float) -> float:
#     if not scores:
#         return 1.0
#     n = len(scores)
#     q_index = max(1, min(math.floor((n + 1) * alpha), n))
#     return sorted(scores)[q_index - 1]

# ══════════════════════════════════════════════════════════════════════
#  Shared: build prompt + retrieve for a given iteration
# ══════════════════════════════════════════════════════════════════════
def build_iteration_prompt(question: str, reasoning_steps: List[str],
                           iteration: int, retriever, retrieval_top_k: int,
                           q_hat_ret: float, all_passages: List[Dict]
                           ) -> Tuple[str, Optional[float], List[Dict]]:
    """
    Build the LLM prompt for a given iteration, performing retrieval if
    iteration > 0. Mutates all_passages in-place (appends new filtered docs).

    Returns: (prompt, top_retrieval_score, newly_retrieved_passages)
    """
    if iteration == 0:
        return (DIRECT_ANSWER_PROMPT.format(question=question),
                None, [])

    query = reasoning_steps[iteration - 1] if reasoning_steps else question
    new_passages = retriever.search(query, top_k=retrieval_top_k)
    top_score = max((p.get("score", 0.0) for p in new_passages),
                    default=None)

    filtered = [p for p in new_passages if p.get("score", 0.0) >= q_hat_ret]
    existing_ids = {p["id"] for p in all_passages}
    for p in filtered:
        if p["id"] not in existing_ids:
            all_passages.append(p)
            existing_ids.add(p["id"])

    prompt = DIRECT_ANSWER_WITH_CONTEXT_PROMPT.format(
        question=question, context=format_passages(all_passages))
    return prompt, top_score, new_passages


# ══════════════════════════════════════════════════════════════════════
#  Prediction set construction helpers
# ══════════════════════════════════════════════════════════════════════
def build_prediction_set(per_iter_parsed: List[List[str]],
                         per_iter_ne: List[float],
                         clusterer: AnswerClusterer,
                         q_hat_freq: float
                         ) -> Tuple[List[str], Dict[str, List[str]]]:
    """Cluster per-iteration samples; include clusters whose max
    per-iteration frequency (across iterations) >= threshold.
    Returns: (pred_set, cluster_map) where cluster_map maps each
    representative to its full list of cluster members."""
    all_valid = [a for g in per_iter_parsed for a in g]
    if not all_valid:
        return [], {}
    clusters = clusterer.cluster(all_valid)
    pred_set = []
    cluster_map = {}
    for members in clusters.values():
        member_set = set(members)
        representative = Counter(members).most_common(1)[0][0]
        max_freq = 0.0
        for t, iter_samples in enumerate(per_iter_parsed):
            if not iter_samples:
                continue
            count = sum(1 for a in iter_samples if a in member_set)
            freq = count / len(iter_samples) - 0.01 * per_iter_ne[t]
            max_freq = max(max_freq, freq)
        if max_freq >= q_hat_freq:
            pred_set.append(representative)
            cluster_map[representative] = list(members)
    return pred_set, cluster_map


# ══════════════════════════════════════════════════════════════════════
#  Post-hoc calibration trial (used by grid search)
# ══════════════════════════════════════════════════════════════════════
def _run_calibration_trial(states: List[Dict], alpha_per_hop: List[float],
                           max_iterations: int
                           ) -> Tuple[float, float, List[float], List[float]]:
    """
    Post-hoc sequential calibration with given alpha_per_hop.
    Returns: (avg_iteration, answer_rate, q_hat_list, c_ans_list)
    """
    n_cal = len(states)
    for s in states:
        s["active"] = True
        s["answered_at"] = None

    q_hat_list, c_ans_list = [], []

    for iteration in range(max_iterations):
        active_indices = [i for i, s in enumerate(states) if s["active"]]
        n_active = len(active_indices)
        n_answerable = sum(
            1 for idx in active_indices
            if any(states[idx][f"_correct_iter_{t}"]
                   for t in range(iteration + 1)))
        c_ans_list.append(n_answerable / n_active)

        alpha_t = alpha_per_hop[iteration]
        if alpha_t <= 0.0:
            q_hat_list.append(0.0)
            continue

        wrong_ne = [states[idx][f"_ne_iter_{iteration}"]
                    for idx in active_indices
                    if all(not states[idx][f"_correct_iter_{t}"]
                           for t in range(iteration + 1))]
        q_hat_i = conformal_quantile(wrong_ne, alpha_t)
        q_hat_list.append(q_hat_i)

        for idx in active_indices:
            if states[idx][f"_ne_iter_{iteration}"] <= q_hat_i:
                states[idx]["active"] = False
                states[idx]["answered_at"] = iteration

    answered = [s for s in states if s["answered_at"] is not None]
    n_answered = len(answered)
    answer_rate = n_answered / n_cal if n_cal else 0.0
    avg_iter = sum(
        s["answered_at"] if s["answered_at"] is not None else max_iterations
        for s in states
    ) / n_cal if n_cal else float(max_iterations)
    return avg_iter, answer_rate, q_hat_list, c_ans_list


# ══════════════════════════════════════════════════════════════════════
#  Retrieval threshold calibration (Phase 1)
# ══════════════════════════════════════════════════════════════════════
def calibrate_retrieval(cal_set: List[Dict], retriever, alpha: float = 0.1,
                        max_iterations: int = 3, retrieval_top_k: int = 5
                        ) -> float:
    all_gold_ret_scores = []
    total_gold_found, total_gold_possible = 0, 0
    total_queries, queries_with_hit = 0, 0
    all_passage_lists = []

    print(f"  Retrieval calibration: {len(cal_set)} questions x "
          f"{max_iterations - 1} retrieval iterations...")

    for rec in tqdm(cal_set, desc="Retrieval calibration"):
        question = rec["question"]
        reasoning_steps = rec.get("reasoning_steps", [])
        gold_titles = get_gold_titles(rec)

        for iteration in range(1, max_iterations):
            query = (reasoning_steps[iteration - 1] if reasoning_steps else question)
            passages = retriever.search(query, top_k=retrieval_top_k)
            all_passage_lists.append(passages)
            total_queries += 1

            g_scores = gold_doc_scores(passages, gold_titles)
            all_gold_ret_scores.extend(g_scores)
            total_gold_found += len(g_scores)
            total_gold_possible += len(gold_titles)
            if g_scores:
                queries_with_hit += 1

    orig_rate = total_gold_found / total_gold_possible if total_gold_possible else 0.0
    hit_ratio = queries_with_hit / total_queries if total_queries else 0.0
    q_hat_ret = conformal_quantile(all_gold_ret_scores, alpha)

    avg_filtered = (np.mean([sum(1 for p in plist if p["score"] >= q_hat_ret)
                             for plist in all_passage_lists])
                    if all_passage_lists else 0.0)

    print(f"    Gold docs found: {total_gold_found}/{total_gold_possible} "
          f"({orig_rate:.1%})")
    print(f"    Avg filtered set size (score >= q_hat_ret): {avg_filtered:.2f}")
    print(f"    Gold hit ratio: {queries_with_hit}/{total_queries} ({hit_ratio:.1%})")
    print(f"    q_hat_ret = {q_hat_ret:.4f}")
    return q_hat_ret


# ══════════════════════════════════════════════════════════════════════
#  NE Calibration (Phase 2)
# ══════════════════════════════════════════════════════════════════════
def calibrate(cal_set, llm, retriever, clusterer, alpha=0.1, n_samples=10,
              max_iterations=3, retrieval_top_k=5, q_hat_ret=0.0,
              grid_steps=20):
    """
    Phase 2: Sequential NE calibration + adaptive rejection thresholds.
    Returns: (q_hat_list, q_hat_freq, q_hat_one_minus_ne, q_hat_ret, cal_details)
    """
    n_cal = len(cal_set)
    states = [{"rec": rec, "per_iter_parsed": [], "all_passages": [],
               "iteration_log": [], "answered_at": None, "active": True}
              for rec in cal_set]

    print(f"  Sequential calibration: up to {max_iterations} iterations "
          f"for {n_cal} questions...")

    # ── Collect NE data for ALL questions through ALL iterations ──
    for iteration in range(max_iterations):
        for idx in tqdm(range(n_cal), desc=f"Cal iter {iteration} ({n_cal} qs)"):
            s = states[idx]
            rec = s["rec"]

            prompt, top_score, _ = build_iteration_prompt(
                rec["question"], rec.get("reasoning_steps", []),
                iteration, retriever, retrieval_top_k, q_hat_ret,
                s["all_passages"])

            ne, majority, raw, parsed, clusters = sample_and_compute_ne(
                llm, prompt, n_samples, clusterer)

            valid_parsed = [a for a in parsed if a and a.strip()]
            s["per_iter_parsed"].append(valid_parsed)
            correct = gold_in_samples(rec["gold_answer"], parsed)
            s[f"_ne_iter_{iteration}"] = ne
            s[f"_correct_iter_{iteration}"] = correct
            s["iteration_log"].append({
                "iteration": iteration, "ne": ne, "majority": majority,
                "n_clusters": len(clusters), "gold_in_samples": correct,
                "had_retrieval": iteration > 0, "samples": valid_parsed,
                "top_retrieval_score": top_score,
            })
        print(f"      Iter {iteration}: processed {n_cal} questions")

    # ── Grid search for alpha_per_hop ──
    # Compute c_ans_final using cluster-based gold detection across all iterations
    n_ever_correct = 0
    for idx in range(n_cal):
        s = states[idx]
        gold_norm = normalize_answer(s["rec"]["gold_answer"])
        found = False
        for t in range(max_iterations):
            iter_samples = s["per_iter_parsed"][t]
            if not iter_samples:
                continue
            clusters_dict = clusterer.cluster(iter_samples)
            for members in clusters_dict.values():
                if any(gold_norm in normalize_answer(a) for a in members):
                    found = True
                    break
            if found:
                break
        if found:
            n_ever_correct += 1
    c_ans_final = n_ever_correct / n_cal if n_cal else 0.0
    alpha_0 = 0.10
    n_free = max_iterations - 1

    print(f"\n  Post-hoc calibration with grid search...")
    print(f"    C_ans_final: {c_ans_final:.4f}, alpha_0: {alpha_0}, "
          f"free params: {n_free}")

    def _derive_alpha_last(alpha_free):
        """Derive alpha^last from error decomposition constraint.
        Returns alpha_last or None if infeasible."""
        trial_alphas = list(alpha_free) + [0.0]
        _, _, _, c_ans_trial = _run_calibration_trial(
            states, trial_alphas, max_iterations)
        c_ans_last = c_ans_trial[-1] if len(c_ans_trial) == max_iterations else 0.0
        budget_used = sum((1 - c_ans_trial[t]) * alpha_free[t]
                          for t in range(n_free))
        remaining = (1 - c_ans_final) * alpha_0 - budget_used
        denom = 1 - c_ans_last
        alpha_last = remaining / denom if denom > 0 else alpha_0
        if alpha_last < 0 or alpha_last > 1:
            return None, c_ans_trial
        return alpha_last, c_ans_trial

    if n_free > 0:
        grid_points = np.linspace(0.0, alpha_0, grid_steps)
        candidates = list(itertools_product(grid_points, repeat=n_free))
        print(f"    Grid: {grid_steps} steps per dim, "
              f"{len(candidates)} total candidates")

        best_obj, best_alpha_free = 1e6, [0.0] * n_free
        for alpha_free in tqdm(candidates, desc="Grid search"):
            alpha_last, _ = _derive_alpha_last(alpha_free)
            if alpha_last is None:
                continue
            full_alphas = list(alpha_free) + [alpha_last]
            avg_iter, answer_rate, _, _ = _run_calibration_trial(
                states, full_alphas, max_iterations)
            obj = avg_iter# + answer_rate
            if obj < best_obj:
                best_obj = obj
                best_alpha_free = list(alpha_free)

        print(f"    Grid best objective: {best_obj:.6f}")
        print(f"    Grid best alpha_free: "
              f"{[f'{a:.6f}' for a in best_alpha_free]}")

        # Derive alpha^last with best params
        alpha_last, _ = _derive_alpha_last(best_alpha_free)
        alpha_per_hop = best_alpha_free + [alpha_last if alpha_last is not None
                                           else alpha_0]
    else:
        alpha_per_hop = [alpha_0]

    # ── Final run with optimal alphas ──
    avg_iter, answer_rate, q_hat_list, c_ans = _run_calibration_trial(
        states, alpha_per_hop, max_iterations)

    print(f"    Final alpha_per_hop: {[f'{a:.6f}' for a in alpha_per_hop]}")
    print(f"    C_ans per hop: {[f'{c:.4f}' for c in c_ans]}")
    print(f"    Avg iteration: {avg_iter:.4f}, Answer rate: {answer_rate:.4f}")
    print(f"    q_hat_list: {[f'{q:.6f}' for q in q_hat_list]}")

    # ── Adaptive rejection calibration ──
    # Calibrate on ALL questions where gold answer appears in any
    # iteration's samples, regardless of early-stopping status.
    one_minus_ne_scores, freq_scores = [], []
    n_gold_appears, n_cant_answer = 0, 0

    for s in states:
        gold = s["rec"]["gold_answer"]
        # For early-stopped questions, only use iterations up to answered_at;
        # for non-early-stopped questions, use all iterations.
        n_iters_used = (s["answered_at"] + 1 if s["answered_at"] is not None
                        else len(s["per_iter_parsed"]))
        all_valid = [a for g in s["per_iter_parsed"][:n_iters_used] for a in g]

        # Compute per-cluster freq the same way as build_prediction_set:
        # overall freq across all_valid, with NE penalty.
        per_iter_ne = [e["ne"] for e in s["iteration_log"][:n_iters_used]]
        ar_ne = per_iter_ne[-1]
        s["ar_one_minus_ne"] = 1.0 - ar_ne

        # Compute the gold answer's frequency per iteration,
        # take the max across iterations. Also determines gold_ever.
        gold_norm = normalize_answer(gold)
        gold_cluster_freq = 0.0
        gold_ever = False
        for t in range(n_iters_used):
            iter_samples = s["per_iter_parsed"][t]
            if not iter_samples:
                continue
            iter_ne = s["iteration_log"][t]["ne"]
            clusters_dict = clusterer.cluster(iter_samples)
            for members in clusters_dict.values():
                member_set = set(members)
                if any(gold_norm in normalize_answer(a) for a in members):
                    gold_ever = True
                    count = sum(1 for a in iter_samples if a in member_set)
                    freq = count / len(iter_samples) - 0.01 * iter_ne
                    gold_cluster_freq = max(gold_cluster_freq, freq)

        s["ar_gold_ever"] = gold_ever
        if gold_ever:
            freq_scores.append(gold_cluster_freq)
            s["ar_score"] = gold_cluster_freq
            one_minus_ne_scores.append(1.0 - ar_ne)
            n_gold_appears += 1
        else:
            n_cant_answer += 1

        s["ar_gold_freqs"] = None  # no longer per-group

    q_hat_freq = conformal_quantile(freq_scores, alpha)
    n_total = n_gold_appears + n_cant_answer
    r_correct = n_gold_appears / n_total if n_total else 1.0
    
    alpha_1 = (((1 - c_ans_final) * alpha - (1-r_correct)*alpha_0)) / (c_ans_final * (1 - alpha))
    q_hat_one_minus_ne = conformal_quantile(one_minus_ne_scores, alpha_1)

    print(f"    Adaptive Rejection: {n_total} questions used for AR calibration")
    print(f"      Answerable: {n_gold_appears}, q_hat_freq = {q_hat_freq:.6f}")
    print(f"      Unanswerable: {n_cant_answer}, "
          f"r_correct = {r_correct:.4f}, alpha_1 = {alpha_1:.6f}")
    print(f"      q_hat_one_minus_ne = {q_hat_one_minus_ne:.6f}")
    print(f"    (q_hat_ret = {q_hat_ret:.4f})")
    print(f" c_ans_final={c_ans_final:.4f} and ")
    # ── Build cal_details ──
    cal_details = []
    for s in states:
        rec = s["rec"]
        detail = {
            "qid": rec["qid"], "question": rec["question"],
            "gold_answer": rec["gold_answer"],
            "iterations": s["iteration_log"],
            "answered_at": s["answered_at"],
            "ar_gold_freqs": s.get("ar_gold_freqs"),
            "ar_gold_ever": s.get("ar_gold_ever"),
            "ar_freq_score": s.get("ar_score"),
            "ar_one_minus_ne": s.get("ar_one_minus_ne"),
            "n_aggregated_samples": (sum(len(g) for g in s["per_iter_parsed"])
                                     if s["active"] else 0),
        }
        cal_details.append(detail)

    return q_hat_list, q_hat_freq, q_hat_one_minus_ne, q_hat_ret, cal_details


# ══════════════════════════════════════════════════════════════════════
#  Test-time iterative pipeline with adaptive rejection
# ══════════════════════════════════════════════════════════════════════
def evaluate_question(rec, llm, retriever, clusterer, q_hat_list, q_hat_freq,
                      q_hat_one_minus_ne, q_hat_ret=0.0, n_samples=10,
                      retrieval_top_k=5, max_iterations=3) -> Dict:
    question = rec["question"]
    reasoning_steps = rec.get("reasoning_steps", [])
    gold_titles = get_gold_titles(rec)

    iterations_log, per_iter_parsed, all_passages = [], [], []
    num_retrieval_calls, num_lm_calls = 0, 0
    total_gold_ret, total_gold_kept = 0, 0
    total_pass_ret, total_pass_kept = 0, 0

    # ── Run ALL iterations to collect data ──
    for iteration in range(max_iterations):
        prompt, top_score, raw_passages = build_iteration_prompt(
            question, reasoning_steps, iteration, retriever,
            retrieval_top_k, q_hat_ret, all_passages)

        if iteration > 0:
            num_retrieval_calls += 1
            n_retrieved = len(raw_passages)
            n_gold_iter = count_gold_docs(raw_passages, gold_titles)
            filtered = [p for p in raw_passages
                        if p.get("score", 0.0) >= q_hat_ret]
            n_kept = len(filtered)
            n_gold_kept_iter = count_gold_docs(filtered, gold_titles)
            total_gold_ret += n_gold_iter
            total_gold_kept += n_gold_kept_iter
            total_pass_ret += n_retrieved
            total_pass_kept += n_kept
        else:
            n_retrieved, n_kept = 0, 0
            n_gold_iter, n_gold_kept_iter = 0, 0

        ne, majority, raw, parsed, clusters = sample_and_compute_ne(
            llm, prompt, n_samples, clusterer)
        num_lm_calls += 1

        valid_parsed = [a for a in parsed if a and a.strip()]
        per_iter_parsed.append(valid_parsed)

        iterations_log.append({
            "iteration": iteration, "ne": ne, "majority": majority,
            "n_clusters": len(clusters), "had_retrieval": iteration > 0,
            "passages_used": len(all_passages), "samples": valid_parsed,
            "top_retrieval_score": top_score, "n_retrieved": n_retrieved,
            "n_kept_after_filter": n_kept,
            "gold_docs_retrieved": n_gold_iter,
            "gold_docs_kept": n_gold_kept_iter,
        })

    # ── Determine answerability using all iterations' data ──
    gold_norm = normalize_answer(rec["gold_answer"])
    gold_in_any = any(gold_norm in normalize_answer(a)
                      for g in per_iter_parsed for a in g if a)

    # ── Post-hoc early-stop check ──
    answered_at = None
    for iteration in range(max_iterations):
        ne = iterations_log[iteration]["ne"]
        if ne <= q_hat_list[iteration]:
            answered_at = iteration
            break

    if answered_at is not None:
        # Early-stopped: build prediction set from iterations up to answered_at
        used_parsed = per_iter_parsed[:answered_at + 1]
        used_ne = [e["ne"] for e in iterations_log[:answered_at + 1]]
        prediction_set, cluster_map = build_prediction_set(
            used_parsed, used_ne, clusterer, q_hat_freq)
        abstained = not prediction_set

        all_selected_members = [m for members in cluster_map.values()
                                for m in members]
        gold_in_pred = any(gold_norm in normalize_answer(a)
                           for a in all_selected_members)
        covered = gold_in_pred

        return {
            "qid": rec["qid"], "question": question,
            "gold_answer": rec["gold_answer"],
            "answered_at_iteration": answered_at,
            "abstained": abstained, "ar_used": False,
            "gold_in_any": gold_in_any,
            "prediction_set": prediction_set,
            "prediction_set_size": len(prediction_set),
            "covered": covered,
            "num_retrieval_calls": num_retrieval_calls,
            "num_lm_calls": num_lm_calls,
            "total_gold_docs_retrieved": total_gold_ret,
            "total_gold_docs_kept": total_gold_kept,
            "total_passages_retrieved": total_pass_ret,
            "total_passages_kept": total_pass_kept,
            "iterations": iterations_log,
        }

    # ── Adaptive Rejection stage ──
    ar_per_iter_ne = [e["ne"] for e in iterations_log]
    ar_ne = ar_per_iter_ne[-1] if ar_per_iter_ne else 1.0

    prediction_set = []
    # if (1.0 - ar_ne) <= q_hat_one_minus_ne:
    #     prediction_set.append("UNANSWERABLE")
    # else:
    prediction_set, cluster_map = build_prediction_set(
        per_iter_parsed, ar_per_iter_ne, clusterer, q_hat_freq)
    prediction_set.append("UNANSWERABLE")

    real_answers = [a for a in prediction_set if a != "UNANSWERABLE"]
    abstained = not real_answers

    all_selected_members = [m for members in cluster_map.values()
                            for m in members]
    gold_in_pred = any(gold_norm in normalize_answer(a)
                       for a in all_selected_members)
    covered = gold_in_pred or (not gold_in_any and "UNANSWERABLE" in prediction_set)

    best_answer = real_answers[0] if real_answers else ""
    all_valid = [a for g in per_iter_parsed for a in g]
    iterations_log.append({
        "iteration": "adaptive_rejection",
        "prediction_set": prediction_set,
        "prediction_set_size": len(prediction_set),
        "ar_one_minus_ne": 1.0 - ar_ne,
        "best_answer": best_answer,
        "n_aggregated_samples": len(all_valid),
        "covered": covered,
    })

    return {
        "qid": rec["qid"], "question": question,
        "gold_answer": rec["gold_answer"],
        "answered_at_iteration": max_iterations if not abstained else -1,
        "abstained": abstained, "ar_used": True,
        "gold_in_any": gold_in_any,
        "prediction_set": prediction_set,
        "prediction_set_size": len(prediction_set),
        "covered": covered,
        "num_retrieval_calls": num_retrieval_calls,
        "num_lm_calls": num_lm_calls,
        "total_gold_docs_retrieved": total_gold_ret,
        "total_gold_docs_kept": total_gold_kept,
        "total_passages_retrieved": total_pass_ret,
        "total_passages_kept": total_pass_kept,
        "iterations": iterations_log,
    }


# ══════════════════════════════════════════════════════════════════════
#  Main experiment runner
# ══════════════════════════════════════════════════════════════════════
def run_experiment(args):
    print(f"Loading data from {args.data_path}...")
    records = load_data(args.data_path)
    print(f"  Loaded {len(records)} records")

    cal_set, test_set = split_data(records, args.cal_size, args.seed)
    print(f"  Calibration: {len(cal_set)}, Test: {len(test_set)}")

    cache_path = os.path.join(args.output_dir, "cache.json")
    cache = load_cache(cache_path)
    print(f"  Cache: {len(cache)} entries loaded")

    print("Initializing LLM client...")
    api_key = (args.openai_api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required (or pass --openai_api_key).")
    model_name = (args.model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
    base_url = (args.openai_base_url or os.getenv("OPENAI_BASE_URL") or "").strip() or None
    raw_llm = OpenAIClient(model=model_name, api_key=api_key, base_url=base_url)
    llm = CachedVLLMClient(raw_llm, cache)
    print(f"  Model: {llm.model}")

    print("Initializing retriever...")
    raw_retriever = RetrieverClient(
        es_url=args.es_url,
        es_index=args.es_index,
        title_field=args.es_title_field,
        text_field=args.es_text_field,
        username=args.es_username,
        password=args.es_password,
    )
    assert raw_retriever.health(), "Elasticsearch is not healthy / reachable."
    retriever = CachedRetrieverClient(raw_retriever, cache)
    print(f"  Retriever: OK ({args.es_url}, index={args.es_index})")

    print(f"Loading sentence-transformer ({args.st_model})...")
    clusterer = AnswerClusterer(model_name=args.st_model,
                                sim_threshold=args.sim_threshold)

    # ── Phase 1: Retrieval threshold calibration ──
    print(f"\n{'='*70}")
    if args.use_conformal_retrieval:
        print(f"Phase 1: Retrieval Calibration (alpha={args.alpha})")
        q_hat_ret = calibrate_retrieval(
            cal_set, retriever, alpha=args.alpha,
            max_iterations=args.max_iterations,
            retrieval_top_k=args.retrieval_top_k)
    else:
        print("Phase 1: Retrieval Calibration — SKIPPED")
        q_hat_ret = 0.0

    # ── Phase 2: NE calibration ──
    print(f"\n{'='*70}")
    print(f"Phase 2: NE Calibration (alpha={args.alpha}, "
          f"n_samples={args.n_samples}, q_hat_ret={q_hat_ret:.4f})")
    print(f"{'='*70}")

    q_hat_list, q_hat_freq, q_hat_one_minus_ne, q_hat_ret, cal_details = calibrate(
        cal_set, llm, retriever, clusterer, alpha=args.alpha,
        n_samples=args.n_samples, max_iterations=args.max_iterations,
        retrieval_top_k=args.retrieval_top_k, q_hat_ret=q_hat_ret,
        grid_steps=args.grid_steps)

    # ── Test evaluation ──
    print(f"\n{'='*70}")
    print(f"Test Evaluation (n_test={len(test_set)})")
    print(f"{'='*70}")

    results = [
        evaluate_question(rec, llm, retriever, clusterer,
                          q_hat_list=q_hat_list, q_hat_freq=q_hat_freq,
                          q_hat_one_minus_ne=q_hat_one_minus_ne,
                          q_hat_ret=q_hat_ret, n_samples=args.n_samples,
                          retrieval_top_k=args.retrieval_top_k,
                          max_iterations=args.max_iterations)
        for rec in tqdm(test_set, desc="NE-Conformal Test")
    ]

    # ── Aggregate metrics ──
    n_total = len(results)
    n_answered = sum(1 for r in results if not r["abstained"])
    n_abstained = n_total - n_answered
    n_covered = sum(1 for r in results if r.get("covered", False))

    coverage_rate = n_covered / n_total
    answer_rate = n_answered / n_total
    abstention_rate = n_abstained / n_total
    avg_retrieval = sum(r["num_retrieval_calls"] for r in results) / n_total
    avg_pred_set_size = sum(r.get("prediction_set_size", 1) for r in results) / n_total

    ar_results = [r for r in results if r.get("ar_used")]
    avg_pred_set_size_ar = (
        sum(r.get("prediction_set_size", 1) for r in ar_results)
        / len(ar_results) if ar_results else 0.0)

    tot_gold_ret = sum(r.get("total_gold_docs_retrieved", 0) for r in results)
    tot_gold_kept = sum(r.get("total_gold_docs_kept", 0) for r in results)
    tot_pass_ret = sum(r.get("total_passages_retrieved", 0) for r in results)
    tot_pass_kept = sum(r.get("total_passages_kept", 0) for r in results)
    gold_retention = tot_gold_kept / tot_gold_ret if tot_gold_ret else 0.0
    pass_retention = tot_pass_kept / tot_pass_ret if tot_pass_ret else 0.0

    iter_counts = Counter()
    for r in results:
        if r["abstained"]:
            iter_counts["abstained"] += 1
        elif r["ar_used"]:
            iter_counts["ar_answered"] += 1
        else:
            iter_counts[f"iter_{r['answered_at_iteration']}"] += 1

    no_retrieval_rate = iter_counts.get("iter_0", 0) / n_total
    ar_save_rate = iter_counts.get("ar_answered", 0) / n_total

    # ── Print results ──
    print(f"\n{'='*70}")
    print("  RESULTS: NE-Based Conformal Prediction with Iterative Retrieval")
    print(f"{'='*70}")
    print(f"  alpha={args.alpha}, n_cal={len(cal_set)}, n_test={n_total}, "
          f"n_samples={args.n_samples}, max_iter={args.max_iterations}")
    print(f"  q_hat_list = [{', '.join(f'{q:.8f}' for q in q_hat_list)}]")
    print(f"  q_hat_freq={q_hat_freq:.6f}, q_hat_1-NE={q_hat_one_minus_ne:.6f}, "
          f"q_hat_ret={q_hat_ret:.4f}\n")

    metrics = [
        ("Coverage rate", f"{coverage_rate:.4f}"),
        ("Avg prediction set size", f"{avg_pred_set_size:.2f}"),
        ("Avg pred set size (AR)", f"{avg_pred_set_size_ar:.2f}"),
        ("Answer rate", f"{answer_rate:.4f}"),
        ("Abstention rate", f"{abstention_rate:.4f}"),
        ("Avg retrieval calls", f"{avg_retrieval:.4f}"),
        ("No-retrieval rate", f"{no_retrieval_rate:.4f}"),
        ("AR save rate", f"{ar_save_rate:.4f}"),
        ("Gold retention rate", f"{gold_retention:.4f}"),
        ("Passage retention rate", f"{pass_retention:.4f}"),
    ]
    for name, val in metrics:
        print(f"  {name:<30} {val:>10}")

    print("\n  Iteration distribution:")
    for key in sorted(iter_counts.keys()):
        count = iter_counts[key]
        print(f"    {key:<20}: {count:>4} ({count/n_total:.1%})")

    # ── Save results ──
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "ne_conformal_results.jsonl")
    summary_path = os.path.join(args.output_dir, "ne_conformal_summary.json")

    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "method": "NE-Conformal-IterativeRetrieval-AR",
        "config": {
            "alpha": args.alpha, "cal_size": len(cal_set),
            "test_size": n_total, "n_samples": args.n_samples,
            "max_iterations": args.max_iterations,
            "retrieval_top_k": args.retrieval_top_k,
            "sim_threshold": args.sim_threshold,
            "seed": args.seed, "model": llm.model, "st_model": args.st_model,
        },
        "calibration": {
            "q_hat_list": q_hat_list, "q_hat_freq": q_hat_freq,
            "q_hat_one_minus_ne": q_hat_one_minus_ne, "q_hat_ret": q_hat_ret,
        },
        "metrics": {
            "coverage_rate": round(coverage_rate, 4),
            "avg_prediction_set_size": round(avg_pred_set_size, 4),
            "avg_prediction_set_size_ar": round(avg_pred_set_size_ar, 4),
            "answer_rate": round(answer_rate, 4),
            "abstention_rate": round(abstention_rate, 4),
            "avg_retrieval_calls": round(avg_retrieval, 4),
            "no_retrieval_rate": round(no_retrieval_rate, 4),
            "ar_save_rate": round(ar_save_rate, 4),
            "gold_docs_retrieved": tot_gold_ret,
            "gold_docs_kept": tot_gold_kept,
            "gold_retention_rate": round(gold_retention, 4),
            "passages_retrieved": tot_pass_ret,
            "passages_kept": tot_pass_kept,
            "passage_retention_rate": round(pass_retention, 4),
        },
        "iteration_distribution": {str(k): v for k, v in sorted(iter_counts.items())},
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Results: {results_path}")
    print(f"  Summary: {summary_path}")
    save_cache(cache, cache_path)


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="NE-Based Conformal Prediction with Iterative Retrieval")
    p.add_argument("--data_path", default="./results_adaptive/adaptive_results.jsonl")
    p.add_argument("--openai_base_url", default=None,
                   help="Optional custom OpenAI-compatible base URL.")
    p.add_argument("--openai_api_key", default=None,
                   help="Optional API key override (defaults to OPENAI_API_KEY env).")
    p.add_argument("--model", default=None)
    p.add_argument("--es_url", default="http://localhost:9200")
    p.add_argument("--es_index", default="wikipedia")
    p.add_argument("--es_title_field", default="title")
    p.add_argument("--es_text_field", default="text")
    p.add_argument("--es_username", default=None)
    p.add_argument("--es_password", default=None)
    p.add_argument("--st_model", default="all-MiniLM-L6-v2")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--cal_size", type=int, default=300)
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--max_iterations", type=int, default=3)
    p.add_argument("--retrieval_top_k", type=int, default=10)
    p.add_argument("--sim_threshold", type=float, default=0.9)
    p.add_argument("--grid_steps", type=int, default=20,
                   help="Grid points per dimension for alpha calibration")
    p.add_argument("--use_conformal_retrieval", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42) #Change different seeds to enhance stronger claim
    p.add_argument("--output_dir", default="./results_ne_conformal")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
