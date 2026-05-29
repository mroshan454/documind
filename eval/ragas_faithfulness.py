"""
RAGAS Faithfulness

Goal: score ONE row (neg_001) for faithfulness, to (a) confirm RAGAS is
performing correctly and (b) check our "I dont Know Answer"
scores HIGH faithfulness.
"""
import re 
import json
from pathlib import Path 

from datasets import Dataset 
from ragas import evaluate 
from ragas.metrics import faithfulness , answer_relevancy 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
ragas_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

RESULTS_PATH = Path(__file__).parent / "eval_results.json"


def strip_citations(text):
    # Remove "Source"
    text = re.sub(r'\(Source:.*?\)','',text)
    return text.strip()

def to_ragas_format(row):
    """Reshape on of our result rows into the keys RAGAS expects."""
    return { 
        "question": row["query"],
        "answer": strip_citations(row["generated_answer"]),
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

    scores = evaluate(dataset, 
                metrics=[faithfulness,answer_relevancy],
                llm=ragas_llm,
                embeddings=ragas_emb,)
    
    print(scores)

    # per-row scores, paired with ids
    df = scores.to_pandas()
    ids = [r["id"] for r in results if r["generated_answer"] is not None]
    for rid, f_score, a_score in zip(ids, df["faithfulness"], df["answer_relevancy"]):
        print(f"{rid:12} faithfulness = {f_score:.3f} answer_relevancy={a_score:.3f}")


if __name__ == "__main__":
   main() 