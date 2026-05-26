import json 
import re 
import unicodedata

def normalize(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '',text)
    text = re.sub(r'\s+', ' ',text).strip()
    return text 

def recall_at_k(gold_context,retrieved_chunks):
    """ 
    Gold Context : list of gold passage strings (What should be retrieved)
    retrieved Chunks: list of dicts , each with a "text" key (What was retrieved)
    Returns: recall score (float between 0 and 1)
    """
    
    denominator = len(gold_context)

    if denominator == 0:
        return None 

    found = 0 
    
    for gold in gold_context:
        for chunk in retrieved_chunks:
           if normalize(gold) in normalize(chunk["text"]): 
              found += 1 
              break 
    recall = found/denominator 
    return recall 
            
def evaluate_recall(results_path):
    with open(results_path) as f:
        results = json.load(f)
    scores = []
    for row in results:
        r = recall_at_k(row["gold_context"], row["retrieved_chunks"])
        print(f'{row["id"]:12} recall = {r}')
        if r is not None:
            scores.append(r)
    avg = sum(scores) / len(scores)
    print(f"\n Mean Recall@k over {len(scores)} scored rows: {avg:.3f}")
    return avg 

evaluate_recall("eval/eval_results.json")
