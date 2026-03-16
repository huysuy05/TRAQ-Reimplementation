#!/usr/bin/env python3
import json
import os
import random
from typing import Any, Dict, List

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INPUT_FILE = "triviaqa_full.json"
OUTPUT_FILE = "triviaqa_600_samples.jsonl"

SAMPLE_QUESTIONS = 600
TOP_K_CTXS = 20
SAMPLES_PER_PASSAGE = 15
SAMPLES_NO_CONTEXT = 15
SEED = 42

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

# OpenAI-first configuration.
# Optional OPENAI_BASE_URL is for custom gateways.
BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MODEL = (
    os.getenv("OPENAI_MODEL")
    or os.getenv("MODEL_NAME")
    or "gpt-4o-mini"
).strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

PROMPT_TEMPLATE = (
    "Answer the following question based on the given context; "
    "Context: {context}\n"
    "Answer the question shortly and output the answer directly and only.\n"
    "Question: {question}\n"
    "Answer:"
)
DIRECT_PROMPT_TEMPLATE = (
    "Answer the following question; "
    "Answer the question shortly and output the answer directly and only.\n"
    "Question: {question}\n"
    "Answer:"
)

if not API_KEY:
    raise RuntimeError("Missing API key. Set OPENAI_API_KEY in your shell or .env.")
if not MODEL:
    raise RuntimeError("Missing model name. Set OPENAI_MODEL or MODEL_NAME.")

if BASE_URL:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
else:
    client = OpenAI(api_key=API_KEY)
embedder = SentenceTransformer(EMBEDDING_MODEL)


def build_prompt(question: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(question=question, context=context)


def build_direct_prompt(question: str) -> str:
    return DIRECT_PROMPT_TEMPLATE.format(question=question)


def get_question(item: Dict[str, Any]) -> str:
    return str(item.get("question") or item.get("Question") or "").strip()


def get_question_id(item: Dict[str, Any]) -> str:
    return str(item.get("question_id") or item.get("QuestionId") or "").strip()


def get_gold_answer(item: Dict[str, Any]) -> Dict[str, Any]:
    ans = item.get("answer") or item.get("Answer")
    if isinstance(ans, str):
        return {"value": ans.strip(), "aliases": []}
    if isinstance(ans, dict):
        value = str(
            ans.get("value")
            or ans.get("Value")
            or ans.get("normalized_value")
            or ans.get("NormalizedValue")
            or ""
        ).strip()
        aliases = ans.get("aliases") or ans.get("Aliases") or ans.get("NormalizedAliases")
        if isinstance(aliases, list):
            aliases = [str(x).strip() for x in aliases if str(x).strip()]
        else:
            aliases = []
        return {"value": value, "aliases": aliases}
    return {"value": "", "aliases": []}


def get_text(obj: Any) -> str:
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, dict):
        for k in (
            "text",
            "passage",
            "content",
            "contents",
            "snippet",
            "Text",
            "Description",
            "Title",
        ):
            v = obj.get(k)
            if v:
                return str(v).strip()
    return ""


def get_passages(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    passages: List[Dict[str, Any]] = []
    seen = set()

    # Add rank-0 from search_results
    search_results = item.get("search_results") or item.get("SearchResults")
    if isinstance(search_results, list) and search_results:
        rank0 = get_text(search_results[0])
        if rank0:
            passages.append({
                "source": "search_results_rank0",
                "index": 0,
                "passage": rank0,
            })
            seen.add(rank0)

    # Add top-20 from ctxs
    ctxs = item.get("ctxs")
    if isinstance(ctxs, list):
        for i, ctx in enumerate(ctxs[:TOP_K_CTXS]):
            text = get_text(ctx)
            if text and text not in seen:
                passages.append({
                    "source": "ctxs",
                    "index": i,
                    "passage": text,
                })
                seen.add(text)

    return passages


def sample_answers(question: str, context: str, n: int) -> List[str]:
    prompt = build_prompt(question, context)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        top_p=0.95,
        max_tokens=32,
        n=n,
    )
    return [
        (choice.message.content or "").strip() if choice.message else ""
        for choice in resp.choices
    ]


def sample_direct_answers(question: str, n: int) -> List[str]:
    prompt = build_direct_prompt(question)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        top_p=0.95,
        max_tokens=32,
        n=n,
    )
    return [
        (choice.message.content or "").strip() if choice.message else ""
        for choice in resp.choices
    ]


def embed(text: str) -> List[float]:
    vec = embedder.encode(text, normalize_embeddings=False)
    return [float(x) for x in vec]


def negative_inner_product(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    return -sum(a[i] * b[i] for i in range(n))


def load_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    # Try normal JSON first.
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return [x for x in data["data"] if isinstance(x, dict)]
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass

    # Fallback: JSONL (one JSON object per line).
    rows: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            continue
    return rows


def main() -> None:
    data = load_data(INPUT_FILE)
    random.seed(SEED)
    if len(data) > SAMPLE_QUESTIONS:
        data = random.sample(data, SAMPLE_QUESTIONS)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for q_idx, item in enumerate(tqdm(data, desc="Questions")):
            question = get_question(item)
            if not question:
                continue
            question_id = get_question_id(item)
            gold_answer = get_gold_answer(item)

            passages = get_passages(item)

            q_embed = None
            q_embed_error = None
            try:
                q_embed = embed(question)
            except Exception as e:
                q_embed = None
                q_embed_error = str(e)

            per_passage_samples = []
            for p in passages:
                p_text = p["passage"]

                neg_ip = None
                neg_ip_error = None
                if q_embed is not None:
                    try:
                        p_embed = embed(p_text)
                        neg_ip = negative_inner_product(q_embed, p_embed)
                    except Exception as e:
                        neg_ip = None
                        neg_ip_error = str(e)
                else:
                    neg_ip_error = q_embed_error
                answers = sample_answers(question, p_text, SAMPLES_PER_PASSAGE)
                per_passage_samples.append({
                    "source": p["source"],
                    "index": p["index"],
                    "passage": p_text,
                    "negative_inner_product": neg_ip,
                    "negative_inner_product_error": neg_ip_error,
                    "answers": answers,
                })

            direct_question_answers = sample_direct_answers(question, SAMPLES_NO_CONTEXT)

            row = {
                "question_index": q_idx,
                "question_id": question_id,
                "question": question,
                "gold_answer": gold_answer,
                "per_passage_samples": per_passage_samples,
                "direct_question_answers": direct_question_answers,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
