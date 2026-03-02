import argparse
import csv
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer

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
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
        aliases = ans.get("Aliases") if "Aliases" in ans else (ans.get("aliases") or [])
        for alias in aliases or []:
            if alias:
                golds.append(str(alias))
    out: List[str] = []
    seen = set()
    for gold in golds:
        ng = norm_text(gold)
        if ng and ng not in seen:
            out.append(ng)
            seen.add(ng)
    return out


def any_gold_match(pred: str, golds_norm: List[str]) -> bool:
    p = norm_text(pred)
    if not p:
        return False
    return any((g in p) or (p in g) for g in golds_norm)


def exact_match_norm(pred: str, golds_norm: List[str]) -> bool:
    p = norm_text(pred)
    if not p:
        return False
    return p in golds_norm


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
                rank = int(sr["Rank"])
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_idx = i
            except Exception:
                pass
    sr0 = srs[best_idx]
    txt = searchresult_to_text(sr0).strip() if isinstance(sr0, dict) else str(sr0).strip()
    return (txt if txt else None), best_idx


class ContrieverEncoder:
    def __init__(self, model_name: str, device: Optional[str] = None):
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


def _normalize_few_shot_example(ex: Dict[str, Any]) -> Optional[Dict[str, str]]:
    q = ex.get("question") or ex.get("Question")
    c = ex.get("context") or ex.get("Context")
    a = ex.get("answer") or ex.get("Answer")
    if not (q and c and a):
        return None
    return {"question": str(q), "context": str(c), "answer": str(a)}


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


def build_prompt(question: str, context: str, few_shot: Sequence[Dict[str, str]]) -> str:
    if not few_shot:
        return ZERO_SHOT_TEMPLATE.format(question=question, context=context)
    parts = ["Answer the following question based on the given context; Answer the question shortly."]
    for ex in few_shot:
        parts.append("Question: {question}\nContext: {context}\nAnswer: {answer}".format(**ex))
    parts.append(f"Question: {question}\nContext: {context}\nAnswer:")
    return "\n".join(parts)


def cache_key_llm(question: str, passage: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(b"vanilla||")
    h.update(norm_text(question).encode("utf-8"))
    h.update(b"||")
    h.update(norm_text(passage)[:4000].encode("utf-8"))
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


def llm_answer(
    client: "OpenAI",
    model: str,
    question: str,
    passage: str,
    temperature: float,
    max_output_tokens: int,
    retries: int,
    retry_sleep_s: float,
    few_shot: Sequence[Dict[str, str]],
) -> str:
    prompt = build_prompt(question=question, context=passage, few_shot=few_shot)
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
    eval_start = n_cal if n_cal < n else 0

    enc = ContrieverEncoder(args.contriever_model, device=args.device)

    client = None
    cache: Dict[str, Any] = {}
    few_shot_examples: List[Dict[str, str]] = []
    if args.enable_llm:
        if OpenAI is None:
            raise RuntimeError("openai package not installed but --enable_llm was set.")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        client = OpenAI()
        cache = load_cache(args.llm_cache_path)
        few_shot_examples = load_few_shot_examples(args.few_shot_path)

    annotated: List[Dict[str, Any]] = []
    top1_hits = 0
    top1_total = 0
    top1_sims: List[float] = []
    exact_hits_eval = 0
    substr_hits_eval = 0
    eval_total = 0

    for row_idx, ex in enumerate(tqdm(rows, desc="Vanilla top-1 retrieval")):
        q = get_question(ex)
        if not q:
            ex["vanilla_passage"] = None
            ex["vanilla_answer"] = None
            annotated.append(ex)
            continue

        ctxs = ex.get("ctxs") or []
        if not isinstance(ctxs, list):
            ctxs = []

        combined: List[Dict[str, Any]] = []
        for ctx in ctxs[: args.topk_ctx]:
            if isinstance(ctx, dict):
                combined.append(dict(ctx))
        for txt in get_searchresults_texts(ex, args.topk_sr):
            if txt:
                combined.append({"text": txt, "source": "searchresult"})

        texts: List[str] = []
        valid_indices: List[int] = []
        for i, item in enumerate(combined):
            t = (item.get("text") or "").strip()
            if t:
                texts.append(t)
                valid_indices.append(i)

        best_text: Optional[str] = None
        best_sim: Optional[float] = None
        if texts:
            q_emb = enc.encode([q], batch_size=1, max_length=args.max_length)[0]
            p_embs = enc.encode(texts, batch_size=args.batch_size, max_length=args.max_length)
            sims = dot_sim(q_emb, p_embs)
            best_local = int(np.argmax(sims))
            best_idx = valid_indices[best_local]
            best_sim = float(sims[best_local])
            best_text = (combined[best_idx].get("text") or "").strip() or None
            top1_sims.append(best_sim)

        p_star_text, _ = pick_p_star_from_searchresults(ex)
        top1_hit = False
        if p_star_text:
            top1_total += 1
            if best_text and norm_text(best_text) == norm_text(p_star_text):
                top1_hit = True
                top1_hits += 1

        ex["vanilla_passage"] = best_text
        ex["vanilla_top1_sim"] = best_sim
        ex["vanilla_p_star_top1"] = top1_hit

        answer: Optional[str] = None
        if args.enable_llm and client is not None and best_text:
            key = cache_key_llm(q, best_text, args.llm_model)
            if key in cache:
                answer = str(cache[key]["answer"])
            else:
                answer = llm_answer(
                    client=client,
                    model=args.llm_model,
                    question=q,
                    passage=best_text,
                    temperature=args.llm_temperature,
                    max_output_tokens=args.max_output_tokens,
                    retries=args.retries,
                    retry_sleep_s=args.retry_sleep_s,
                    few_shot=few_shot_examples,
                )
                cache[key] = {"answer": answer}
        ex["vanilla_answer"] = answer

        golds = get_gold_aliases_norm(ex)
        exact_hit = None
        substr_hit = None
        if answer is not None and golds:
            exact_hit = exact_match_norm(answer, golds)
            substr_hit = any_gold_match(answer, golds)
            if row_idx >= eval_start:
                eval_total += 1
                exact_hits_eval += int(exact_hit)
                substr_hits_eval += int(substr_hit)
        ex["vanilla_exact_hit"] = exact_hit
        ex["vanilla_substr_hit"] = substr_hit
        annotated.append(ex)

    top1_rate = top1_hits / max(top1_total, 1)
    avg_top1_sim = float(np.mean(top1_sims)) if top1_sims else 0.0
    print(f"\nVanilla top-1 retrieval coverage (p* hit): {top1_hits}/{top1_total} = {top1_rate:.3f}")
    print(f"Avg top-1 retriever similarity: {avg_top1_sim:.3f}")

    if args.enable_llm and eval_total > 0:
        print(f"\n[Vanilla Eval on rows {eval_start}..{n-1}]")
        print(f"  Exact-match accuracy: {exact_hits_eval}/{eval_total} = {exact_hits_eval/max(eval_total, 1):.4f}")
        print(f"  Substring-match accuracy: {substr_hits_eval}/{eval_total} = {substr_hits_eval/max(eval_total, 1):.4f}")

    write_jsonl(args.output_path, annotated)
    print(f"\nSaved annotated JSONL: {args.output_path}")

    if args.summary_csv_path:
        rows_out: List[List[Any]] = []
        for ex in annotated:
            rows_out.append([
                get_question(ex),
                ex.get("vanilla_p_star_top1"),
                ex.get("vanilla_top1_sim"),
                ex.get("vanilla_passage"),
                ex.get("vanilla_answer"),
                ex.get("vanilla_exact_hit"),
                ex.get("vanilla_substr_hit"),
            ])
        write_csv(
            args.summary_csv_path,
            [
                "question",
                "top1_hit_p_star",
                "top1_similarity",
                "top1_passage",
                "vanilla_answer",
                "exact_hit",
                "substring_hit",
            ],
            rows_out,
        )
        print(f"Saved summary CSV: {args.summary_csv_path}")

    if args.enable_llm:
        save_cache(args.llm_cache_path, cache)

    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Vanilla RAG baseline: top-1 retrieval + single answer")
    p.add_argument("--input_path", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--max_rows", type=int, default=300)
    p.add_argument("--n_cal", type=int, default=None, help="Used only to define eval rows for reporting.")

    p.add_argument("--topk_sr", type=int, default=10)
    p.add_argument("--topk_ctx", type=int, default=10)

    p.add_argument("--contriever_model", type=str, default="facebook/contriever-msmarco")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=256)

    p.add_argument("--enable_llm", action="store_true")
    p.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    p.add_argument("--llm_temperature", type=float, default=0.0)
    p.add_argument("--max_output_tokens", type=int, default=16)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry_sleep_s", type=float, default=0.5)
    p.add_argument("--llm_cache_path", type=str, default="cache_llm_samples.json")
    p.add_argument("--few_shot_path", type=str, default=None)

    p.add_argument("--summary_csv_path", type=str, default="summary_vanilla.csv")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
