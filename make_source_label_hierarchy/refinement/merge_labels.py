import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import numpy as np
import faiss
import torch
import pandas as pd


##
## kmeans clustering
##

def train_kmeans_clustering(x, ncentroids=1024, niter=50, verbose=True, downsample_to=None, save_path=None):
    """
    Perform K-means clustering on the input data.

    Args:
        x (numpy.ndarray): Input data matrix of shape (n_samples, n_features).
        ncentroids (int): Number of centroids.
        niter (int): Number of iterations for training.
        verbose (bool): Whether to print verbose information.

    Returns:
        numpy.ndarray: Cluster labels for the input data.
    """
    d = x.shape[1]
    if downsample_to is not None:
        x = x[:downsample_to]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=torch.cuda.is_available())
    kmeans.train(x)
    if save_path is not None:
        with open(save_path, 'wb') as f:
            np.save(f, kmeans.centroids)
    return kmeans


def assign_kmeans_clusters(embs, save_path=None, kmeans=None):
    """
    Assigns K-means clusters to the given embeddings.

    Args:
        embs (numpy.ndarray): Embeddings to be clustered.
        save_path (str): Path to the saved centroids.

    Returns:
        numpy.ndarray: Cluster indices for the embeddings.
    """
    if kmeans is None:
        centroids = np.load(save_path)
        n, d = centroids.shape
        kmeans = faiss.Kmeans(d, n, verbose=True, gpu=True, niter=0, nredo=0)
        kmeans.train(embs, init_centroids=centroids) # this ensures that kmeans.index is created
        assert np.sum(kmeans.centroids - centroids) == 0, "centroids are not the same" # sanity check
    
    cluster_distances, cluster_indices = kmeans.assign(embs)
    return cluster_indices



## 
## old methods
def get_clusters_from_large_embedding_matrix(embedding_matrix, chunk_size=1000, threshold=0.5):
    """
    This function takes a large embedding matrix and returns all clusters of rows with similarity > threshold.
    The computation is done in chunks to handle large matrices that do not fit into memory.
    
    Args:
    embedding_matrix (np.ndarray): A 2D numpy array representing the embedding matrix.
    chunk_size (int): The size of chunks to process at a time.
    threshold (float): A threshold value to determine if two rows are similar.
    
    Returns:
    list of lists: A list where each sublist contains the indices of rows that form a cluster.
    """
    num_embeddings = embedding_matrix.shape[0]
    rows, cols, data = [], [], []
    
    # Compute the similarity matrix in chunks and sparsify the adjacency matrix
    for start_idx in tqdm(range(0, num_embeddings, chunk_size), desc="Computing similarity matrix"):
        end_idx = min(start_idx + chunk_size, num_embeddings)
        chunk = embedding_matrix[start_idx:end_idx]
        chunk_normed = chunk / np.linalg.norm(chunk, axis=1, keepdims=True)
                
        # Precompute the norms of the inner chunks to avoid redundant calculations
        inner_chunk_norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        # Compute only the lower triangle of the similarity matrix
        for inner_start_idx in range(0, start_idx + chunk_size, chunk_size):
            inner_end_idx = min(inner_start_idx + chunk_size, num_embeddings)
            inner_chunk = embedding_matrix[inner_start_idx:inner_end_idx]
            inner_chunk_normed = inner_chunk / inner_chunk_norms[inner_start_idx:inner_end_idx]
            
            similarity_chunk = np.dot(chunk_normed, inner_chunk_normed.T)
            if inner_start_idx == start_idx:
                # Zero out the upper triangle including the diagonal
                similarity_chunk[np.triu_indices_from(similarity_chunk)] = 0
            adjacency_chunk = similarity_chunk > threshold
            
            # Store the indices and values of the non-zero entries in the adjacency matrix
            adjacency_indices = np.argwhere(adjacency_chunk)
            rows.extend(start_idx + adjacency_indices[:, 0])
            cols.extend(inner_start_idx + adjacency_indices[:, 1])
            data.extend([1] * len(adjacency_indices))
    
    # Create a sparse adjacency matrix
    sparse_matrix = csr_matrix((data, (rows, cols)), shape=(num_embeddings, num_embeddings))
    
    # Find connected components in the graph represented by the sparse matrix
    n_components, labels = connected_components(csgraph=sparse_matrix, directed=False, return_labels=True)
    
    # Group row indices by their component labels to form clusters
    clusters = [[] for _ in range(n_components)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    
    return clusters



def get_clusters_from_similarity_matrix(similarity_matrix=None, embedding_matrix=None, threshold=0.5):
    """
    This function takes a similarity matrix and returns all clusters of rows with similarity > threshold.
    
    Args:
    similarity_matrix (np.ndarray): A 2D numpy array representing the similarity matrix.
    threshold (float): A threshold value to determine if two rows are similar.
    
    Returns:
    list of lists: A list where each sublist contains the indices of rows that form a cluster.
    """
    if similarity_matrix is None:
        if embedding_matrix is None:
            raise ValueError("Either similarity_matrix or embedding_matrix must be provided.")
        
        # Compute pairwise cosine similarity from the embedding matrix
        normed_embeddings = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        similarity_matrix = np.dot(normed_embeddings, normed_embeddings.T)


    # Create a boolean adjacency matrix where entries are True if similarity > threshold
    adjacency_matrix = similarity_matrix > threshold
    
    # Convert the boolean adjacency matrix to a sparse matrix
    sparse_matrix = csr_matrix(adjacency_matrix)
    
    # Find connected components in the graph represented by the sparse matrix
    n_components, labels = connected_components(csgraph=sparse_matrix, directed=False, return_labels=True)
    
    # Group row indices by their component labels to form clusters
    clusters = [[] for _ in range(n_components)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    
    return clusters



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cluster rows based on similarity matrix or embeddings.")
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--embeddings', type=str, required=False, help='Path to the embeddings file.')
    parser.add_argument('--method', type=str, choices=['kmeans', 'adjacency'], required=True, help='Clustering method to use: "kmeans" or "adjacency".')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output clusters.')
    
    args = parser.parse_args()

    # Load data
    if args.embeddings:
        embeddings = np.load(args.embeddings)
    else:
        embeddings = None
    
    # Load data
    data = pd.read_csv(args.input_csv)

    if args.method == "kmeans":
        kmeans = train_kmeans_clustering(embeddings)
        clusters = assign_kmeans_clusters(embeddings, kmeans=kmeans)
    elif args.method == "adjacency":
        clusters = get_clusters_from_large_embedding_matrix(embeddings)
    else:
        raise ValueError(f"Invalid clustering method: {args.method}")
        
    source_df = pd.read_csv(args.input_csv)
    source_df["cluster"] = clusters
    source_df.to_csv(args.output_path, index=False)
    