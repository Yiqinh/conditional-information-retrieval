import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from haystack.nodes import DensePassageRetriever, EmbeddingRetriever
from haystack.document_stores import InMemoryDocumentStore
from tqdm import tqdm

import json


save_dir = "../trained_model"
dev_filename = "/project/jonmay_231/spangher/Projects/conditional-information-retrieval/source_summaries/v2_info_parsed/combined_test_prompt1_v2.json"
os.environ['KMP_DUPLICATE_LIB_OK']='True'

print(dev_filename)
with open(dev_filename, 'r') as f:
    articles = json.load(f)

document_store = InMemoryDocumentStore()
documents = []
for article in articles:
    for source in article['sources']:
        content = {
            'id': article['url'] + "#" + source['Name'],
            'content': source['Information']
        }
        documents.append(content)

document_store.write_documents(documents)
document_store.update_embeddings(retriever="facebook/dpr-ctx_encoder-single-nq-base")


# retriever = EmbeddingRetriever(
#         document_store=document_store,
#         embedding_model=model, # this is my custom-trained model
#     )

reloaded_retriever = DensePassageRetriever.load(load_dir=save_dir, document_store=document_store)
print("finished loading the retriever")

results = {}
for article in tqdm(articles):
    question = article['query']
    # print(question)
    if question == "":
        print("This question is empty")
        continue
    results[question] = reloaded_retriever.retrieve(question, top_k=10)

print(results)


   
with open(f"/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/test_result.json", 'w') as json_file:
        json.dump(results, json_file)

