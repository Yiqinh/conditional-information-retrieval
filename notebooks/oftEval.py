import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

import json
import numpy as np
import faiss
from tqdm import tqdm
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import FAISSDocumentStore
from haystack.utils import convert_files_to_docs


save_dir = "../trained_model"
data_dir = "/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/docs_oft"
dev_filename = "/project/jonmay_231/spangher/Projects/conditional-information-retrieval/source_summaries/v2_info_parsed/combined_test_prompt1_v2.json"
index_file = "/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/test_oft.index"
# Load development data
with open(dev_filename, 'r') as f:
    articles = json.load(f)[:10]

reloaded_retriever = DensePassageRetriever(
    document_store=None,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=False,  # Set to False if you want to run on CPU
)


# Write documents to the document store
# print("creating source txt folder...")
# file_idx = 0
# for article in tqdm(articles, desc="creating source txt folder"):
#     for source in article['sources']:
#         source_text = source['Information']
#         with open(f"{data_dir}/{file_idx}.txt", 'w') as source_file:
#             source_file.write(source_text)
#         file_idx += 1

# print("converting files to docs...")
# docs = convert_files_to_docs(dir_path=data_dir)
# print("writing to document store...")
# document_store.write_documents(docs)

# # Embed documents
# tmp = reloaded_retriever.embed_documents(docs)

def create_index(vector_dim):
    """Create a FAISS index for the given vector dimension."""
    index = faiss.IndexFlatL2(vector_dim)  # Using L2 distance for similarity
    return index

def add_vectors_to_index(index, vectors):
    """Add vectors to the FAISS index."""
    if isinstance(vectors, np.ndarray) and vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    index.add(vectors)  # Add vectors to the index
    print("Vectors added to index!")

def search_vectors(index, query_vector, k):
    """Search the index for the k nearest vectors to the query."""
    D, I = index.search(np.array([query_vector], dtype=np.float32), k)  # Perform the search
    return D, I  # Distances and indices of the nearest vectors

def get_index(source, index):
    if not source or source == "":
        return 0
    query_vector = reloaded_retriever.embed_queries([source])[0]
    return index.search(np.array([query_vector], dtype=np.float32), 1)[0][0]

# Set up the FAISS index
# dim = 768  # Dimension of the vectors
# index = create_index(dim)


# # Adding vectors to the index
# add_vectors_to_index(index, tmp)

# # Simulating a query vector (for example purposes)
# question = "what is the square root of 144?"
# query_vector = reloaded_retriever.embed_queries([question])[0]

# distances, indices = search_vectors(index, query_vector, 10)
# print("Nearest neighbors: ", indices)
# print("Distances: ", distances)
mapping_file = "/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/mapping_oft.json"
with open(mapping_file, 'r') as f:
    mapping = json.load(f)

reverse_mapping = {v: k for k, v in mapping.items()}

index = faiss.read_index(index_file)


def process_batch(batch):
    batch_results = []
    questions = [article['query'] for article in batch if article['query'] != ""]
    query_vectors = reloaded_retriever.embed_queries(questions)
    distances, dr_indices = [], []
    for query_vector in query_vectors:
        i = search_vectors(index, query_vector, 10)[1]
        curr_indices = i[0].tolist()
        curr_indices = [int(i) for i in curr_indices]
        # dr_indices.append([mapping[str(i)] for i in curr_indices])
        dr_indices.append(curr_indices)

    gt_sources = []
    for article in batch:
        curr_sources = []
        for source in article['sources']:
            if source['Information'] == "" or not source['Information']:
                continue
            # curr_sources.append(source['Information'])
            curr_sources.append(reverse_mapping[source['Information']])
        gt_sources.append(curr_sources)
        #     source_vector = reloaded_retriever.embed_queries([source['Information']])[0]
        #     sourceid = search_vectors(index, source_vector, 1)[1]
        #     curr_indices.append(sourceid[0][0])
        # curr_indices = [int(i) for i in curr_indices]
        # gt_indices.append(curr_indices)
    
    
    for question, gt, dr in zip(questions, gt_sources, dr_indices):
        one_article = {}
        one_article['query'] = question
        one_article['sources'] = gt
        one_article['dr_sources'] = dr
        batch_results.append(one_article)
    return batch_results


batch_size = 10
results = []

for i in tqdm(range(0, len(articles), batch_size), desc="Generating retrieval results in batches"):
    batch = articles[i:min(i + batch_size, len(articles))]
    results.extend(process_batch(batch))


print(results)
with open(f"/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/test_result_oft.json", 'w') as json_file:
    json.dump(results, json_file)

print("DONE!!!")

