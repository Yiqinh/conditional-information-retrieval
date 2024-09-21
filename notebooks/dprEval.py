import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from haystack.nodes import DensePassageRetriever, EmbeddingRetriever
from haystack.document_stores import InMemoryDocumentStore, FAISSDocumentStore
from haystack.utils import convert_files_to_docs
from tqdm import tqdm
import json


save_dir = "../trained_model"
data_dir = "/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/docs"
dev_filename = "/project/jonmay_231/spangher/Projects/conditional-information-retrieval/source_summaries/v2_info_parsed/combined_test_prompt1_v2.json"
os.environ['KMP_DUPLICATE_LIB_OK']='True'


with open(dev_filename, 'r') as f:
    articles = json.load(f)

# document_store = InMemoryDocumentStore()
document_store = FAISSDocumentStore(sql_url="sqlite:///", faiss_index_factory_str="Flat")
file_idx = 0
for article in tqdm(articles, desc="creating source txt folder"):
    for source in article['sources']:
        source_text = source['Information']
        with open(f"{data_dir}/{file_idx}.txt", 'w') as source_file:
            source_file.write(source_text)
        file_idx += 1

print("converting files to docs...")
docs = convert_files_to_docs(dir_path=data_dir)
print("writing to document store...")
document_store.write_documents(docs)

retriever = EmbeddingRetriever(
    document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)
print("updating embeddings...")
document_store.update_embeddings(retriever)

print("creating index mapping...")
documents = document_store.get_all_documents()
mapping = {}
for document in documents:
    mapping[document.content] = document.id

with open(f"/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/index.json", 'w') as json_file:
    json.dump(mapping, json_file)


reloaded_retriever = DensePassageRetriever.load(load_dir=save_dir, document_store=document_store)
print("finished loading the retriever")

results = []
for article in tqdm(articles, desc="generating retrieval results"):
    question = article['query']
    if question == "":
        print("This question is empty")
        continue
    topk = reloaded_retriever.retrieve(question, top_k=10)
    # dr_result = []
    
    # for k in topk:
    #     try:
    #         url, name, text = k.content.split("###")
    #     except Exception as e:
    #         print(k.content)

    #     id = url + "#" + name
    #     curr_k = {}
    #     curr_k['id'] = id
    #     curr_k['text'] = text
    #     dr_result.append(curr_k)

    one_article = {}
    one_article['url'] = article['url']
    one_article['sources'] = article['sources']
    one_article['dr_sources'] = topk
    one_article['query'] = article['query']

    results.append(one_article)

   
with open(f"/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/test_result.json", 'w') as json_file:
    json.dump(results, json_file)

