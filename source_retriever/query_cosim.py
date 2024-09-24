import os
import json
import statistics
import pandas as pd
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer
here = os.path.dirname(os.path.abspath(__file__))

model = SentenceTransformer("all-MiniLM-L6-v2")

query1 = {}

file = os.path.join(here, 'v2_search_res', 'v2_search_test_prompt1_all.json')
with open(file, 'r') as file:
    articles = json.load(file)
    for article in articles:
        url = article['url']
        q = article['query']
        query1[url] = q

file = os.path.join(here, 'queries', 'query2_v2_parsed.json')
with open(file, 'r') as file:
    query2 = json.load(file)

results = []
for url, q1 in tqdm(query1.items(), desc="Num Sources"):
    q2 = query2[url]
    
    embeddings1 = model.encode(q1)
    embeddings2 = model.encode(q2)
    # Compute cosine similarities
    similarities = model.similarity(embeddings1, embeddings2)
    sim = similarities.tolist()[0][0]
    results.append(sim)

print("THE COSIM BETWEEN Q1 and Q2 IS", statistics.mean(results))
    
