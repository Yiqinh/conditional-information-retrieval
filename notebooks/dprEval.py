import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore
from tqdm import tqdm

import json


save_dir = "../trained_model"
dev_filename = "../source_summaries/v2_info_parsed/combined_test_prompt1_v2.json"
os.environ['KMP_DUPLICATE_LIB_OK']='True'

reloaded_retriever = DensePassageRetriever.load(load_dir=save_dir, document_store=InMemoryDocumentStore)
print("finished loading the retriever")

with open(dev_filename, 'r') as f:
    articles = json.load(f)

results = {}
for article in tqdm(articles):
    question = article['query']
    results[question] = reloaded_retriever.retrieve(question)

print(results)


   
with open(f"/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/test_result.json", 'w') as json_file:
        json.dump(results, json_file)

