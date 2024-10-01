import os
import json
import numpy as np

from sentence_transformers import SentenceTransformer
here = os.path.dirname(os.path.abspath(__file__))    
helper_dir = os.path.join(os.path.dirname(here), 'helper_functions')

from cosim import *

from collections import defaultdict
url_to_queries = defaultdict(list)

f = os.path.join(os.path.dirname(here), 'source_retriever', 'v2_search_res', 'v2_search_test_prompt1_all.json')
with open(f, 'r') as file:
    articles = json.load(file)

for article in articles:
    url = article['url']
    query1 = article['query']
    url_to_queries[url].append(query1)


for iteration in range(5):
    for i in range(0, 11500, 500):
        file = f"iter_{iteration}_search_results_v2_{i}_{i+500}.json"
        file = os.path.join(here, 'v2', file)
        with open(f, 'r') as file:
            articles = json.load(file)
        for article in articles:
            url = article['url']
            query = article['query']
            url_to_queries[url].append(query)

embed_queries = []
for url, query_list in url_to_queries.items():
    embed_queries.append(query_list)

query_cosim_matrix = cosim_queries(embed_queries)
np.save('query_cosim_matrix.npy', query_cosim_matrix)

query_text_embed = []
for i in range(0, 11500, 500):
    file = f"iter_4_search_results_v2_{i}_{i+500}.json"
    file = os.path.join(here, 'v2', file)
    with open(f, 'r') as file:
        articles = json.load(file)
    for article in articles:
        a = {}
        a['url'] = article['url']
        a['texts'] = []
        for source in article['dr_sources']:
            a['texts'].append(source['text'])
        
        query_text_embed.append(a)

query_text_cosim = cosim_query_source(query_text_embed)
np.save('query_text_cosim_matrix.npy', query_text_cosim)
