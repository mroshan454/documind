"""
DocuMind Eval Harness — Runner (Stage 1: capture)

This script does ONE job: run every query in the eval set through the live
DocuMind /query endpoint, and save what happened next to what *should* have
happened. It does NOT compute metrics yet — that's the next stage.

Think of it as the PyTorch validation loop's forward pass: run the (frozen)
system over held-out data and record predictions. Scoring comes after.

Prereq: DocuMind running locally ->  uvicorn app.main:app --reload
Run:     python eval/run_eval.py
Output:  eval/eval_results.json
"""

import json
import time
from pathlib import Path

import requests

# --- Config -----------------------------------------------------------------
API_URL = "http://localhost:8000/query"
EVAL_SET_PATH = Path(__file__).parent / "documind_eval_set.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"
TOP_K = 3                  # must match what you want to evaluate (your /query default)
REQUEST_TIMEOUT = 60       # seconds; LLM calls can be slow


def load_eval_set(path: Path) -> list[dict]:
    """Load the hand-authored ground-truth rows."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["rows"]


def run_one_query(query: str, k: int = TOP_K) -> dict:
    """
    Hit the real /query endpoint exactly as a user would (HTTP).
    Returns the parsed JSON: {"answer": ..., "sources": [{text, score, ...}]}.
    Raises on network / HTTP errors so the loop can record the failure.
    """
    resp = requests.post(
        API_URL,
        json={"question": query, "k": k},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    rows = load_eval_set(EVAL_SET_PATH)
    print(f"Loaded {len(rows)} eval rows. Hitting {API_URL} (k={TOP_K})...\n")

    results = []
    for i, row in enumerate(rows, start=1):
        rid = row["id"]
        query = row["query"]
        print(f"[{i}/{len(rows)}] {rid}: {query[:60]}...")

        record = {
            # --- authored in advance (ground truth) ---
            "id": rid,
            "query": query,
            "ground_truth_answer": row["ground_truth_answer"],
            "gold_context": row["gold_context"],
            "source_doc": row.get("source_doc"),
            "difficulty": row.get("difficulty"),
            # --- generated at eval time (filled below) ---
            "generated_answer": None,
            "retrieved_chunks": None,   # list of {text, score, source, page}
            "error": None,
        }

        try:
            t0 = time.time()
            response = run_one_query(query)
            latency = round(time.time() - t0, 2)

            record["generated_answer"] = response.get("answer")
            record["retrieved_chunks"] = response.get("sources", [])
            record["latency_seconds"] = latency
            print(f"    ok ({latency}s, {len(record['retrieved_chunks'])} chunks)")
        except Exception as e:
            record["error"] = str(e)
            print(f"    ERROR: {e}")

        results.append(record)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_ok = sum(1 for r in results if r["error"] is None)
    print(f"\nDone. {n_ok}/{len(results)} succeeded. Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()