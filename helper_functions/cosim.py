import os
import json
import statistics
import numpy as np

from sentence_transformers import SentenceTransformer

here = os.path.dirname(os.path.abspath(__file__))    

def cosim_queries(iter_queries: list[list]):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    res = []
    for queries in iter_queries:
        embedding = model.encode(queries)
        similarities = model.similarity(embedding, embedding) #calculates the cosine similarity by default
        res.append(similarities)

    res = np.array(res)
    return res

def cosim_query_source(articles: list):
    """
    [
        {'query': query, 'texts': ['text1', 'text2']}
    ]
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    res = []
    for query, texts in articles.items():
        query_embedding = model.encode([query])
        text_embedding = model.encode(texts)
        sim = model.similarity(query_embedding, text_embedding)
        res.append(sim)
    
    res = np.array(res)
    return res


