from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore
import json


save_dir = "../saved_models/dpr"
dev_filename = "testingData.json"

reloaded_retriever = DensePassageRetriever.load(load_dir=save_dir)


with open(dev_filename, 'r') as f:
    articles = json.load(f)

results = {}
for article in articles:
    question = article['question']
    results[question] = reloaded_retriever.retrieve(question)


   
with open(f"/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/test_result.json", 'w') as json_file:
        json.dump(results, json_file)

