import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import statistics
from tqdm import tqdm
import numpy as np


# mapping_file = "/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/mapping.json"
# with open(mapping_file, 'r') as f:
#     mapping = json.load(f)

# mapping_file = "/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/mapping_oft.json"
# with open(mapping_file, 'r') as f:
#     mapping_oft = json.load(f)



def get_scores(path: str):
        
    precision_list = []
    recall_list = []
    f1_list = []

    with open(path, 'r') as file:
        articles = json.load(file)
        for article in tqdm(articles):
            y_pred = set([str(i) for i in article['dr_sources']])
            y_true = set(article['sources'])


            # for dr_source, source in zip(article['dr_sources'], article['sources']):
            #     y_pred.add(dr_source)
            #     y_true.add(source)
            
            if len(y_true) == 0:
                continue
            
            true_pos = set.intersection(y_pred, y_true)
            n = len(true_pos)
            # print(n, len(y_true), len(y_pred))

            recall = n / len(y_true)
            precision = n / len(y_pred)

            if (recall + precision) == 0:
                f1 = 0

            else:
                f1 = (2 * precision * recall) / (precision + recall)
            
            recall_list.append(recall)
            precision_list.append(precision)
            f1_list.append(f1)
    
    avg_prec = statistics.mean(precision_list)
    avg_rec = statistics.mean(recall_list)
    avg_f1 = statistics.mean(f1_list)

    print("average precision:", avg_prec)
    print("average recall:", avg_rec)
    print("average f1:", avg_f1)



get_scores("/project/jonmay_1426/spangher/conditional-information-retrieval/fine_tuning/test_result_combined.json")
# get_scores("/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/test_result_oft.json")





# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# # Initialize a pre-trained sentence transformer model for embedding
# model = SentenceTransformer('all-MiniLM-L6-v2')
# # Example dictionary of documents
# documents = {
#     "doc1": "Text of the first document.",
#     "doc2": "Text of the second document.",
#     "doc3": "Text of the third document."
# }
# # Convert documents to vectors
# doc_ids = list(documents.keys())
# doc_texts = list(documents.values())
# doc_vectors = model.encode(doc_texts)
# # Dimension of the vectors
# d = doc_vectors.shape[1]
# # Creating a FAISS index
# index = faiss.IndexFlatL2(d)  # Using L2 distance for similarity
# index.add(doc_vectors)
# # Query to ground truth mapping
# query_ground_truth = {
#     "relevant information about the second document": ["doc2", "doc3"]
# }
# # Function to evaluate retrieval
# def evaluate_retrieval(query, ground_truth_ids, k=2):
#     query_vector = model.encode([query])
#     distances, indices = index.search(query_vector, k)
#     retrieved_ids = [doc_ids[idx] for idx in indices[0]]
#     precision = len(set(retrieved_ids).intersection(ground_truth_ids)) / len(retrieved_ids)
#     recall = len(set(retrieved_ids).intersection(ground_truth_ids)) / len(ground_truth_ids)
#     f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
#     return {
#         "query": query,
#         "retrieved_ids": retrieved_ids,
#         "precision": precision,
#         "recall": recall,
#         "f1_score": f1_score
#     }
# # Evaluate each query
# results = []
# for query, gt_ids in query_ground_truth.items():
#     result = evaluate_retrieval(query, gt_ids, k=2)
#     results.append(result)
# # Print results
# for result in results:
#     print("Query:", result["query"])
#     print("Retrieved IDs:", result["retrieved_ids"])
#     print("Precision:", result["precision"])
#     print("Recall:", result["recall"])
#     print("F1 Score:", result["f1_score"])
#     print("-" * 30)