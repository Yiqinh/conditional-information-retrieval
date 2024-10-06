import json
import numpy as np
import faiss
from tqdm import tqdm
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import FAISSDocumentStore
from haystack.utils import convert_files_to_docs
from IPython import embed

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

index_file = "1005_full.index"
index = faiss.read_index(index_file)


save_dir = "/nas04/tenghaoh/conditional-retrieval/notebooks/saved_models"
doc_dir = "data/"
train_filename = 'train-2.json'
dev_filename = 'test-2.json'
train_data = load_data(doc_dir+train_filename)
test_data = load_data(doc_dir+dev_filename)

import pickle
with open('qid2did.pkl', 'rb') as f:
    qid2did = pickle.load(f)
with open('query_index_map.pkl', 'rb') as f:
    query_index_map = pickle.load(f)
with open('doc_index_map.pkl', 'rb') as f:
    doc_index_map = pickle.load(f)

question = train_data[0]['question']
reloaded_retriever = DensePassageRetriever.load(load_dir=save_dir, document_store=None)
query_vector = reloaded_retriever.embed_queries([question])
embed()
def search_vectors(index, query_vector, k):
    """Search the index for the k nearest vectors to the query."""
    D, I = index.search(np.array(query_vector, dtype=np.float32), k)  # Perform the search
    return D, I  # Distances and indices of the nearest vectors

i = search_vectors(index, query_vector, 10)
embed()