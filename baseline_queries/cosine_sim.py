#pip install -U sentence-transformers

import os
import json
import statistics
from sentence_transformers import SentenceTransformer

here = os.path.dirname(os.path.abspath(__file__))    

def cos_similarity(path: str):
    cos_scores = []
    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open(path, 'r') as file:
        articles = json.load(file)
        for url, dict in articles.items():

            y_pred = []
            y_true = []
            for source in dict['dr']:
                text = source['text']
                y_pred.append(text)

            for id, text in dict['truth'].items():
                y_true.append(text)
            
            if len(y_true) == 0:
                continue

            y_pred_embedding = model.encode(y_pred)
            y_true_embedding = model.encode(y_true)

            similarities = model.similarity(y_pred_embedding, y_true_embedding) #calculates the cosine similarity by default
            return similarities



if __name__ == "__main__":
    a = cos_similarity(os.path.join(here, "baseline_results", "retrieved_sources_test_set.json"))
    print(a)