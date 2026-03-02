"""
export_llm_samples.py
Exports LLM responses per (question, passage) into a readable JSON file.

Usage:
    python export_llm_samples.py \
        --jsonl labeled_v2.jsonl \
        --cache cache_llm_samples.json \
        --output llm_samples_readable.json \
        --M 15 \
        --model gpt-4o-mini
"""

import argparse
import hashlib
import json
import re


def norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def cache_key(question: str, passage: str, M: int, model: str) -> str:
    h = hashlib.sha256()
    h.update(norm_text(question).encode())
    h.update(b"||")
    h.update(norm_text(passage)[:4000].encode())
    h.update(b"||")
    h.update(str(M).encode())
    h.update(b"||")
    h.update(model.encode())
    return h.hexdigest()


def cache_key_question_only(question: str, M: int, model: str) -> str:
    h = hashlib.sha256()
    h.update(b"q_only||")
    h.update(norm_text(question).encode())
    h.update(b"||")
    h.update(str(M).encode())
    h.update(b"||")
    h.update(model.encode())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, default="labeled_v2.jsonl")
    parser.add_argument("--cache", type=str, default="cache_llm_samples.json")
    parser.add_argument("--output", type=str, default="llm_samples_readable.json")
    parser.add_argument("--M", type=int, default=15)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    with open(args.cache) as f:
        cache = json.load(f)
    print(f"Loaded cache: {len(cache)} entries")

    with open(args.jsonl) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded JSONL: {len(rows)} rows")

    output = []
    for row in rows:
        q = row.get("Question") or row.get("question") or ""
        gold = row.get("Answer", {})
        c_agg = row.get("C_agg", [])
        hit_gold = row.get("hit_gold")

        entry = {
            "question": q,
            "gold_answer": gold,
            "C_agg": c_agg,
            "hit_gold": hit_gold,
            "question_only_samples": [],
            "passage_samples": [],
        }

        # Question-only samples
        qkey = cache_key_question_only(q, args.M, args.model)
        if qkey in cache:
            entry["question_only_samples"] = cache[qkey].get("samples", [])

        # Per-passage samples (in_C_ret passages)
        for ctx in (row.get("ctxs") or []):
            if not isinstance(ctx, dict):
                continue
            if not ctx.get("in_C_ret"):
                continue
            passage = (ctx.get("text") or "").strip()
            if not passage:
                continue
            pkey = cache_key(q, passage, args.M, args.model)
            samples = cache.get(pkey, {}).get("samples", [])
            entry["passage_samples"].append({
                "passage": passage,
                "contriever_sim": ctx.get("contriever_sim"),
                "in_C_llm": ctx.get("in_C_llm"),
                "samples": samples,
            })

        output.append(entry)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(output)} entries to {args.output}")


if __name__ == "__main__":
    main()
