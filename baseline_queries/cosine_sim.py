#pip install -U sentence-transformers

import os
import json
import statistics
import logging

from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment

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

            cost = similarities.numpy()
            row_ind, col_ind = linear_sum_assignment(cost, maximize=True) #hungarian matching
            
            total = cost[row_ind, col_ind].sum() #calculate the total cost of the matching

            cos_scores.append(total / min(len(y_true), len(y_pred))) #divide by the number of edges to get average cosine similarity. Append this to the list of cosine scores
            break
    
    return statistics.mean(cos_scores) #return the mean cosine similarity


if __name__ == "__main__":
    a = cos_similarity(os.path.join(here, "baseline_results", "retrieved_sources_test_set.json"))
    logging.info(f"The average cosine similarity is {a}")
    print(a)
    print("here")
    print(f"The average cosine similarity is {a}") 
