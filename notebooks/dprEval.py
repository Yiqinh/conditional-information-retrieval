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
# documents = []
error_count = 0
for article in tqdm(articles, desc="creating source txt folder"):
    for source in article['sources']:
        source_name = source['Name']
        source_text = source['Information']
        url = source['url']
        file_name = source_name.replace(" ", "_")
        file_name = source_name.replace("/", "_")
        try:
            with open(f"{data_dir}/{file_name}.txt", 'w') as source_file:
                source_file.write(source_name + "###" + source_text)
        except Exception as e:
            print(f"An error occurred while writing the file: {e}")
            error_count += 1

print("error count", error_count)


print("converting files to docs...")
docs = convert_files_to_docs(dir_path=data_dir)
print("updating document store...")
document_store.write_documents(docs)


# document_store.write_documents(documents)

# embedding_model = EmbeddingRetriever(
#     document_store=document_store,
#     embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
#     use_gpu=True
# )

# print("updating embeddings")
# document_store.update_embeddings(retriever=embedding_model)
# print("embedding update completed")

# retriever = EmbeddingRetriever(
#         document_store=document_store,
#         embedding_model=model, # this is my custom-trained model
#     )


retriever = EmbeddingRetriever(
    document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)
print("updating embeddings...")
document_store.update_embeddings(retriever)
reloaded_retriever = DensePassageRetriever.load(load_dir=save_dir, document_store=document_store)
print("finished loading the retriever")

results = []
for article in tqdm(articles):
    question = article['query']
    # print(question)
    if question == "":
        print("This question is empty")
        continue
    topk = reloaded_retriever.retrieve(question, top_k=10)
    dr_result = {}
    
    for k in topk:
        url, name, text = k.content.split("###")
        id = url + "#" + name
        dr_result['id'] = id
        dr_result['text'] = text
    one_article = {}
    one_article['url'] = article['url']
    one_article['sources'] = article['sources']
    one_article['dr_sources'] = dr_result
    one_article['query'] = article['query']
    
    results.append(one_article)

   
with open(f"/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/test_result.json", 'w') as json_file:
    json.dump(results, json_file)

