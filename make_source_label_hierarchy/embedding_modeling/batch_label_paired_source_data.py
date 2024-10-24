#!/usr/bin/env python
# coding: utf-8

import os
import re
import glob
import json
import argparse
import pandas as pd
import numpy as np
import jsonlines
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai  # Ensure you have installed the OpenAI library
import sys
sys.path.append('..')
from utils_client import write_prompts_to_files, process_batches_with_openai, download_and_process_outputs, process_input_output_data_from_openai_files
from utils_basic import batchify, process_source_data

# Set environment variable
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def compute_embeddings(
        source_df,
        embedding_model_name, 
        col_name='Narrative Function', 
        err_col_name='Is_Error',
        max_returned=None
):
    """
    Compute embeddings for the specified column in the DataFrame using a SentenceTransformer model.
    
    Parameters:
    - source_df: DataFrame containing the source data.
    - embedding_model_name: Name of the SentenceTransformer model to use for computing embeddings.
    - col_name: Name of the column to compute embeddings for (default is 'Narrative Function').
    - err_col_name: Name of the column indicating errors (default is 'Is_Error').
    - max_returned: Maximum number of samples to return (optional).
    
    Returns:
    - embeddings: Computed embeddings for the specified column.
    - idx_of_df: Index of the DataFrame rows used for computing embeddings.
    """
    model = SentenceTransformer(embedding_model_name)
    if err_col_name in source_df.columns:
        source_df = source_df.loc[source_df[err_col_name] == 'No', col_name]
    if max_returned:
        source_df = source_df.sample(n=max_returned)

    narrative_functions = source_df[col_name].dropna()
    texts = narrative_functions.str.split(':').str.get(0).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    idx_of_df = narrative_functions.index
    return embeddings, idx_of_df


def compute_high_similarity_pairs(embeddings, idx_of_df, sim_threshold=0.3, sample_size=2_000_000, batch_size=1000):
    """
    Identify pairs of data points with high similarity based on cosine similarity of embeddings.
    This version processes embeddings in batches to handle large datasets efficiently.
    
    Parameters:
    - embeddings: Embeddings of the data points.
    - idx_of_df: Index of the DataFrame rows corresponding to the embeddings.
    - sim_threshold: Threshold for considering a pair as highly similar (default is 0.3).
    - sample_size: Number of high similarity pairs to sample (default is 2,000,000).
    - batch_size: Number of embeddings to process in each batch (default is 1000).
    
    Returns:
    - high_sim_sample: DataFrame containing sampled high similarity pairs.
    """
    num_embeddings = len(embeddings)
    high_sim_pairs = []

    for start in tqdm(range(0, num_embeddings, batch_size), total=int(np.ceil(num_embeddings / batch_size))):
        end = min(start + batch_size, num_embeddings)
        batch_embeddings = embeddings[start:end]
        batch_idx = idx_of_df[start:end]

        # Compute cosine similarity for the current batch
        similarity_matrix = cosine_similarity(batch_embeddings, embeddings)
        
        # Zero out lower triangle and diagonal for the current batch
        for i in range(end - start):
            similarity_matrix[i, start:start + i + 1] = 0

        # Convert to DataFrame and filter based on similarity threshold
        sim_df = pd.DataFrame(similarity_matrix, index=batch_idx, columns=idx_of_df)
        sim_df = sim_df.stack().reset_index()
        sim_df.columns = ['level_0', 'level_1', 'similarity']
        batch_high_sim_pairs = sim_df.loc[
            (sim_df['similarity'] > sim_threshold) &
            (sim_df['similarity'] < 0.99999)
        ]
        high_sim_pairs.append(batch_high_sim_pairs)

    # Concatenate all high similarity pairs from each batch
    high_sim_pairs = pd.concat(high_sim_pairs, ignore_index=True)
    high_sim_pairs = high_sim_pairs.loc[high_sim_pairs['level_0'] != high_sim_pairs['level_1']]
    high_sim_sample = high_sim_pairs.sample(n=min(sample_size, len(high_sim_pairs)))

    return high_sim_sample


def create_high_similarity_samples(text_df, high_sim_sample, text_col_name='Narrative Function'):
    """
    Create a DataFrame with pairs of texts that have high similarity.
    
    Parameters:
    - text_df: DataFrame containing the text data.
    - high_sim_sample: DataFrame containing high similarity pairs.
    - text_col_name: Name of the column containing the text data (default is 'Narrative Function').
    
    Returns:
    - high_sim_pairwise_samples_to_evaluate: DataFrame with pairs of high similarity texts.
    """
    narrative_functions = text_df[text_col_name]
    high_sim_pairwise_samples_to_evaluate = (
        pd.concat([
            narrative_functions.loc[high_sim_sample['level_0']].reset_index(drop=True).rename('source_1'),
            narrative_functions.loc[high_sim_sample['level_1']].reset_index(drop=True).rename('source_2'),
        ], axis=1)
        .dropna()
        .drop_duplicates()
    )
    return high_sim_pairwise_samples_to_evaluate


def generate_prompts(high_sim_samples, k=5):
    """
    Generate prompts for the OpenAI API based on high similarity samples.
    
    Parameters:
    - high_sim_samples: DataFrame containing high similarity text pairs.
    - k: Number of pairs to include in each prompt (default is 5).
    
    Returns:
    - all_prompts: List of generated prompts for the OpenAI API.
    """
    prompt_template = (f"""I will show you {k} pairs of sources, all from different news articles.

Are the two sources in each pair playing similar narrative roles in their respective articles?
Think broadly about the role the source is playing, given the description. Don't pay attention to the specific events of each story.
Answer with "Yes" or "No".

""" + 
'\n'.join([f'{i}. Source 1: {{source_1_{i}}}, Source 2: {{source_2_{i}}}' for i in range(1, k+1)]) +

"""

Answer each sequentially and number them with 1., 2., 3., etc.""")

    all_prompts = []
    total_batches = int(len(high_sim_samples) / k)
    for t in tqdm(batchify(high_sim_samples.iterrows(), k), total=total_batches):
        input_dict = {}
        for i, (_, row) in enumerate(t, 1):
            input_dict[f'source_1_{i}'] = row['source_1']
            input_dict[f'source_2_{i}'] = row['source_2']
        p = prompt_template.format(k=k, **input_dict)
        all_prompts.append(p)
    return all_prompts



def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process narrative function similarity.')
    parser.add_argument('--data_dir', type=str, default='../data/v2_narr_keywords', help='Directory containing the input data files.')
    parser.add_argument('--output_dir', type=str, default='../data/openai-batches/narr-role-similarity', help='Directory to save output files.')
    parser.add_argument('--openai_api_key', type=str, required=True, help='Your OpenAI API key.')
    parser.add_argument('--model_name', type=str, default='gpt-4', help='OpenAI model name to use.')
    parser.add_argument('--embedding_model_name', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name for embeddings.')
    parser.add_argument('--sim_threshold', type=float, default=0.3, help='Similarity threshold for selecting pairs.')
    parser.add_argument('--sample_size', type=int, default=2000000, help='Number of sample pairs to process.')
    parser.add_argument('--k', type=int, default=5, help='Number of pairs per prompt.')
    parser.add_argument('--batch_size', type=int, default=40000, help='Number of prompts per batch file.')
    parser.add_argument('--completion_window', type=str, default='24h', help='Completion window for OpenAI batch processing.')
    args = parser.parse_args()

    # Process source data
    source_df = process_source_data(data_dir=args.data_dir)

    # 
    # Step 1: create data for training
    # 
    # Compute embeddings
    embeddings, idx_of_df = compute_embeddings(source_df, args.embedding_model_name)
    high_sim_sample = compute_high_similarity_pairs(embeddings, idx_of_df, args.sim_threshold, args.sample_size)
    high_sim_samples = create_high_similarity_samples(source_df, high_sim_sample)


    # 
    # Step 2: Generate prompts for prompting an LLM to label the data
    #
    all_prompts = generate_prompts(high_sim_samples, k=args.k)
    batch_files = write_prompts_to_files(all_prompts, args.output_dir, batch_size=args.batch_size, model_name=args.model_name)
    batch_ids = process_batches_with_openai(batch_files, args.openai_api_key, args.model_name, completion_window=args.completion_window)
    all_data = download_and_process_outputs(batch_ids, args.output_dir, args.openai_api_key)
    input_files = batch_files
    output_files = [os.path.join(args.output_dir, f"{batch_id}_output.jsonl") for batch_id in batch_ids]
    full_data_exp_df = process_input_output_data_from_openai_files(input_files, output_files)

    # Save the processed data
    output_csv = os.path.join(args.output_dir, 'processed_narrative_roles.csv')
    full_data_exp_df.to_csv(output_csv, index=False)
    print(f"Processing complete. Data saved to '{output_csv}'.")

if __name__ == "__main__":
    main()
