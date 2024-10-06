import json
import numpy as np
import faiss
from tqdm import tqdm
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import FAISSDocumentStore
from haystack.utils import convert_files_to_docs
from pathlib import Path
from IPython import embed
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

doc_dir = "data/"
train_filename = 'train-2.json'
dev_filename = 'test-2.json'
train_data = load_data(doc_dir+train_filename)
test_data = load_data(doc_dir+dev_filename)

full_data = train_data+test_data
save_dir = "/nas04/tenghaoh/conditional-retrieval/notebooks/saved_models"

def convert_ctx2string(ctx):
    return ctx['title']+". "+ctx['text']

def write_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)

txts_dir = "data_txts"

idx = 0
doc_index_map = []
query_index_map = {}
qid2did = {}
did2doc = {}
for q_idx, data_item in enumerate(full_data):
    if len(data_item['question'])<5:
        continue
    qid2did[q_idx] = []
    query_index_map[q_idx] = data_item['question']
    for num, positive_ctx in enumerate(data_item['positive_ctxs']):
        # embed()
        # assert(False)
        qid2did[q_idx].append(idx + num)
        # Write content to a txt file
        write_to_file(f"{txts_dir}/{idx + num}.txt", convert_ctx2string(positive_ctx))

        doc_index_map.append(idx + num)
        did2doc[idx + num] = convert_ctx2string(positive_ctx)
    idx += len(data_item['positive_ctxs'])  # Update idx to continue from the last added item

import pickle
with open('qid2did.pkl', 'wb') as f:
    pickle.dump(qid2did, f)
with open('query_index_map.pkl', 'wb') as f:
    pickle.dump(query_index_map, f)
with open('doc_index_map.pkl', 'wb') as f:
    pickle.dump(doc_index_map, f)
with open('did2doc.pkl', 'wb') as f:
    pickle.dump(did2doc, f)
# Initialize document store and retriever
document_store = FAISSDocumentStore(sql_url="sqlite:///", faiss_index_factory_str="Flat")
reloaded_retriever = DensePassageRetriever.load(load_dir=save_dir, document_store=document_store)
dim = 768  # Dimension of the vectors
index = faiss.IndexFlatL2(dim)
index = faiss.IndexIDMap(index)

for id in tqdm(doc_index_map):
    with open(f"{txts_dir}/{id}.txt", 'r') as source_file:
        doc_vector = reloaded_retriever.embed_documents(convert_files_to_docs(file_paths=[Path(f"{txts_dir}/{id}.txt")]))
        index.add_with_ids(doc_vector, id)
index_file = "1005_full.index"
print("saving index...")
faiss.write_index(index, index_file)
