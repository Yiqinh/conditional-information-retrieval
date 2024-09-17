import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore
from tqdm import tqdm

import json


save_dir = "../trained_model"
dev_filename = "/scratch1/spangher/liheng/combined_test_prompt1_v2.json"
os.environ['KMP_DUPLICATE_LIB_OK']='True'

reloaded_retriever = DensePassageRetriever.load(load_dir=save_dir, document_store=InMemoryDocumentStore())
print("finished loading the retriever")

with open(dev_filename, 'r') as f:
    articles = json.load(f)

results = {}
for article in tqdm(articles):
    question = article['query']
    if question is None:
         print("This question is empty")
         continue
    results[question] = reloaded_retriever.retrieve(question, top_k=10)

print(results)


   
with open(f"/scratch1/spangher/conditional-information-retrieval/fine_tuning/test_result.json", 'w') as json_file:
        json.dump(results, json_file)

