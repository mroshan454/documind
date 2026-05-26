"""
RAGAS Faithfulness

Goal: score ONE row (neg_001) for faithfulness, to (a) confirm RAGAS is
performing correctly and (b) check our "I dont Know Answer"
scores HIGH faithfulness.
"""

import json
from pathlib import Path 

from datasets import Dataset 
from ragas import evaluate 
from ragas.metrics import faithfulness 

RESULTS_PATH = Path(__file__).parent / "eval_results.json"

def to_ragas_format(row):
    """Reshape on of our result rows into the keys RAGAS expects."""
    return { 
        "question": row["query"],
        "answer": row["generated_answer"],
        "ground_truth":row["ground_truth_answer"],
        "contexts": [chunk["text"] for chunk in row["retrieved_chunks"]],

    }

def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    # reshape all rows
    rows = [to_ragas_format(r) for r in results if r["generated_answer"] is not None]

    dataset = Dataset.from_dict({
        "question":     [r["question"] for r in rows],
        "answer":       [r["answer"] for r in rows],
        "contexts":     [r["contexts"] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })

    scores = evaluate(dataset, metrics=[faithfulness])
    print(scores)

    # per-row scores, paired with ids
    df = scores.to_pandas()
    for rid, f_score in zip([r["id"] for r in results if r["generated_answer"] is not None],
                            df["faithfulness"]):
        print(f"{rid:12} faithfulness = {f_score}")


if __name__ == "__main__":
   main() 