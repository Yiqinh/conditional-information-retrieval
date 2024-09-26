import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import json
import numpy as np
import faiss
from tqdm import tqdm
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import FAISSDocumentStore
from haystack.utils import convert_files_to_docs
from pathlib import Path


save_dir = "../trained_model"
data_dir = "/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/docs"
dev_filename = "/project/jonmay_231/spangher/Projects/conditional-information-retrieval/source_summaries/v2_info_parsed/combined_test_prompt1_v2.json"
index_file = "/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/test.index"
mapping_file = "/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/mapping.json"
# Load development data
with open(dev_filename, 'r') as f:
    articles = json.load(f)

# Initialize document store and retriever
document_store = FAISSDocumentStore(sql_url="sqlite:///", faiss_index_factory_str="Flat")
reloaded_retriever = DensePassageRetriever.load(load_dir=save_dir, document_store=document_store)
dim = 768  # Dimension of the vectors
index = faiss.IndexFlatL2(dim)
index = faiss.IndexIDMap(index)


# Write documents to the document store
file_idx = 0
mapping = {}
for article in tqdm(articles, desc="creating source txt folder"):
    for source in article['sources']:
        source_text = source['Information']
        doc_vector = reloaded_retriever.embed_documents(convert_files_to_docs(file_paths=[Path(f"{data_dir}/{file_idx}.txt")]))
        index.add_with_ids(doc_vector, file_idx)
        mapping[file_idx] = source_text
        with open(f"{data_dir}/{file_idx}.txt", 'w') as source_file:
            source_file.write(source_text)
            
        file_idx += 1

print("saving index...")
faiss.write_index(index, index_file)

print("saving mapping...")
with open(mapping_file, 'w') as f:
    json.dump(mapping, f, indent=4)

print("DONE")


# print("converting files to docs...")
# docs = convert_files_to_docs(dir_path=data_dir)
# print("writing to document store...")
# document_store.write_documents(docs)

# Embed documents
# tmp = reloaded_retriever.embed_documents(docs)

# def create_index(vector_dim):
#     """Create a FAISS index for the given vector dimension."""
#     index = faiss.IndexFlatL2(vector_dim)  # Using L2 distance for similarity
#     return index

# def add_vectors_to_index(index, vectors):
#     """Add vectors to the FAISS index."""
#     if isinstance(vectors, np.ndarray) and vectors.dtype != np.float32:
#         vectors = vectors.astype(np.float32)
#     index.add(vectors)  # Add vectors to the index
#     print("Vectors added to index!")

# def search_vectors(index, query_vector, k):
#     """Search the index for the k nearest vectors to the query."""
#     D, I = index.search(np.array([query_vector], dtype=np.float32), k)  # Perform the search
#     return D, I  # Distances and indices of the nearest vectors

# def get_index(source, index):
#     query_vector = reloaded_retriever.embed_queries([source])[0]
#     return index.search(np.array([query_vector], dtype=np.float32), 1)[0][0]

# # Set up the FAISS index




# # Adding vectors to the index
# # add_vectors_to_index(index, tmp)

# # Simulating a query vector (for example purposes)
# question = "what is the square root of 144?"
# query_vector = reloaded_retriever.embed_queries([question])[0]

# distances, indices = search_vectors(index, query_vector, 10)
# print("Nearest neighbors: ", indices)
# print("Distances: ", distances)

# faiss.write_index(index, index_file)