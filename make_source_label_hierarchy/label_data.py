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

# Set environment variable
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def parse_sources(input_string):
    """
    Parse the sources from a given input string.
    """
    import re
    # Remove any starting text before the first 'Name:'
    match = re.search(r'\bName:', input_string)
    if match:
        input_string = input_string[match.start():]
    else:
        # No 'Name:' found, return empty list
        return []
        
    # Split the input string into blocks separated by two or more newlines
    blocks = re.split(r'\n\s*\n', input_string)
    source_list = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        # Initialize a dictionary to store fields
        source_dict = {}
        current_field = None
        # Split the block into lines
        lines = block.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Match any line that looks like 'Field Name: Value'
            m = re.match(r'^([^:]+):\s*(.*)', line)
            if m:
                field_name = m.group(1).strip()
                field_name = re.sub(r'\s*(\(?\d+\)?[:.\s]*)', '', field_name).strip()
                field_name = re.sub(r'-|\**|Grouped', '', field_name).strip()
                field_value = m.group(2).strip()
                field_value = re.sub(r'\(.*\)', '', field_value).strip()
                source_dict[field_name] = field_value
                current_field = field_name
            else:
                # If the line doesn't start with a field name, it's part of the previous field
                if current_field:
                    source_dict[current_field] += ' ' + line
        # Only add the source if it contains at least one field
        if source_dict:
            source_list.append(source_dict)
    return source_list

def batchify(iterable, n=1):
    """
    Yield successive n-sized chunks from an iterable.
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def format_batch_prompt(prompts, model_name):
    """
    Format a batch of prompts for the OpenAI API.
    """
    output_ps = []
    for i, p in enumerate(prompts):
        message_body = {
            "custom_id": f"request-{i}", 
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {
                "model": model_name, 
                "messages": [{
                    "role": "system", 
                    "content": "You are a helpful journalist's assistant."
                },{
                    "role": "user", 
                    "content": p
                }],
                "max_tokens": 1000
            }
        }
        output_ps.append(message_body)
    return output_ps

def read_keyword_files(data_dir):
    """
    Read JSON lines files from a directory and concatenate them into a DataFrame.
    """
    files = glob.glob(os.path.join(data_dir, '*'))
    dfs = []
    for f in files:
        dfs.append(pd.read_json(f, lines=True))
    df = pd.concat(dfs, ignore_index=True)
    return df

def process_source_data(df):
    """
    Process the DataFrame to extract and structure source data.
    """
    source_df = (
        df
        .assign(parsed_sources=lambda df: df['response'].apply(parse_sources))
        .explode('parsed_sources')
        .dropna()
    )
    source_df = (source_df[['url', 'parsed_sources']]
        .pipe(lambda df: pd.concat([
            df['url'].reset_index(drop=True),
            pd.DataFrame(df['parsed_sources'].tolist())
        ], axis=1))
    )
    cols_to_keep = ['url', 'Name', 'Original Name', 'Narrative Function', 'Is_Error']
    source_df = source_df[cols_to_keep]
    return source_df

def compute_embeddings(source_df, embedding_model_name):
    """
    Compute embeddings for the 'Narrative Function' field using a SentenceTransformer model.
    """
    model = SentenceTransformer(embedding_model_name)
    narrative_functions = source_df.loc[source_df['Is_Error'] == 'No', 'Narrative Function'].dropna()
    texts = narrative_functions.str.split(':').str.get(0).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    idx_of_df = narrative_functions.index
    return embeddings, idx_of_df

def compute_high_similarity_pairs(embeddings, idx_of_df, sim_threshold=0.3, sample_size=2000000):
    """
    Compute pairs of high similarity based on cosine similarity of embeddings.
    """
    similarity_matrix = cosine_similarity(embeddings, embeddings)
    tril_indices = np.tril_indices_from(similarity_matrix)
    similarity_matrix[tril_indices] = 0  # Zero out lower triangle and diagonal
    sim_df = pd.DataFrame(similarity_matrix, index=idx_of_df, columns=idx_of_df)
    sim_df = sim_df.stack().reset_index()
    sim_df.columns = ['level_0', 'level_1', 'similarity']
    high_sim_pairs = sim_df.loc[
        (sim_df['similarity'] > sim_threshold) &
        (sim_df['similarity'] < 0.99999)
    ]
    high_sim_pairs = high_sim_pairs.loc[high_sim_pairs['level_0'] != high_sim_pairs['level_1']]
    high_sim_sample = high_sim_pairs.sample(n=sample_size)
    return high_sim_sample

def create_high_similarity_samples(source_df, high_sim_sample):
    """
    Create DataFrame with pairs of 'Narrative Function' texts for high similarity samples.
    """
    narrative_functions = source_df['Narrative Function']
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

def write_prompts_to_files(all_prompts, output_dir, batch_size=40000, model_name='gpt-4'):
    """
    Write the prompts to JSON Lines files in batches.
    """
    os.makedirs(output_dir, exist_ok=True)
    prompt_batches = []
    for prompt_batch in batchify(all_prompts, batch_size):
        batch_to_write = format_batch_prompt(prompt_batch, model_name)
        prompt_batches.append(batch_to_write)
    batch_files = []
    for i, b in enumerate(prompt_batches):
        batch_file = os.path.join(output_dir, f'narr-role-similarity-batch-{i}.jsonl')
        with jsonlines.open(batch_file, 'w') as f:
            f.write_all(b)
        batch_files.append(batch_file)
    return batch_files

def process_batches_with_openai(batch_files, openai_api_key, model_name, max_tokens=1000, completion_window='24h'):
    """
    Process the batches using OpenAI's batch processing API.
    """
    openai.api_key = openai_api_key
    batch_ids = []
    for batch_file in tqdm(batch_files):
        with open(batch_file, 'rb') as f:
            batch_input_file = openai.File.create(file=f, purpose="batch")
        batch_input_file_id = batch_input_file.id
        # Create a batch processing job
        batch_response = openai.Batch.create(
            input_file=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata={
                "description": "narrative role similarity evaluation"
            }
        )
        batch_ids.append(batch_response['id'])
    return batch_ids

def download_and_process_outputs(batch_ids, output_dir, openai_api_key):
    """
    Download the outputs from OpenAI batch processing and process them.
    """
    openai.api_key = openai_api_key
    all_data = []
    for batch_id in tqdm(batch_ids):
        batch_info = openai.Batch.retrieve(id=batch_id)
        output_file_id = batch_info['output_file_id']
        if output_file_id:
            output_file = openai.File.download(id=output_file_id)
            output_file_path = os.path.join(output_dir, f"{batch_id}_output.jsonl")
            with open(output_file_path, 'w') as f:
                f.write(output_file.decode('utf-8'))
            # Read and process the output file
            with jsonlines.open(output_file_path) as reader:
                for obj in reader:
                    all_data.append(obj)
    # Further processing of all_data can be done here
    return all_data

def process_input_output_data_from_files(input_files, output_files):
    """
    Process the input prompts and OpenAI responses from files to extract relevant data.
    """
    all_input_data = []
    for f in input_files:
        data = list(jsonlines.open(f))
        data_df = pd.DataFrame(data)
        data_df['filename'] = os.path.basename(f)
        all_input_data.append(data_df)

    all_output_data = []
    for f in output_files:
        data = list(jsonlines.open(f))
        data_df = pd.DataFrame(data)
        data_df['filename'] = os.path.basename(f)
        all_output_data.append(data_df)

    all_input_data_df = pd.concat(all_input_data)
    all_output_data_df = pd.concat(all_output_data)

    # Merge input and output data
    all_input_data_df['input'] = all_input_data_df['body'].str.get('messages').str.get(1).str.get('content')
    all_input_data_df['f_key'] =  all_input_data_df['filename'].str.replace('.jsonl', '')
    all_output_data_df['output'] = all_output_data_df['response'].str.get('body').str.get('choices').str.get(0).str.get('message').str.get('content')
    all_output_data_df['f_key'] = all_output_data_df['filename'].str.split('__').str.get(0)

    full_data_df = all_input_data_df[['custom_id', 'f_key', 'input']].merge(
        all_output_data_df[['custom_id', 'f_key', 'output']], on=['custom_id', 'f_key']
    )

    # Further processing as in your original code
    full_data_df['input'] = full_data_df['input'].str.split('\n').apply(lambda x: list(filter(lambda y: re.search(r'^\d\.', y), x)))
    full_data_df['output'] = full_data_df['output'].str.split('\n')
    full_data_df = full_data_df.loc[lambda df: df['output'].str.len() == 5]
    full_data_exp_df = full_data_df.explode(['input', 'output'])
    full_data_exp_df['input_chunks'] = full_data_exp_df['input'].str.split(r'Source \d\:', regex=True)

    full_data_exp_df = (full_data_exp_df
        .assign(source_1=lambda df: df['input_chunks'].str.get(1).str.strip())
        .assign(source_2=lambda df: df['input_chunks'].str.get(2).str.strip())
        .drop(columns=['input', 'input_chunks'])
    )

    full_data_exp_df['output'] = full_data_exp_df['output'].str.replace(r'\d\.', '', regex=True).str.strip()
    full_data_exp_df = full_data_exp_df.reset_index(drop=True)

    return full_data_exp_df

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

    # Read keyword files
    df = read_keyword_files(args.data_dir)

    # Process source data
    source_df = process_source_data(df)

    # Compute embeddings
    embeddings, idx_of_df = compute_embeddings(source_df, args.embedding_model_name)

    # Compute high similarity pairs
    high_sim_sample = compute_high_similarity_pairs(
        embeddings, idx_of_df, args.sim_threshold, args.sample_size
    )

    # Create high similarity samples to evaluate
    high_sim_samples = create_high_similarity_samples(source_df, high_sim_sample)

    # Generate prompts
    all_prompts = generate_prompts(high_sim_samples, k=args.k)

    # Write prompts to files
    batch_files = write_prompts_to_files(
        all_prompts, args.output_dir, batch_size=args.batch_size, model_name=args.model_name
    )

    # Process batches with OpenAI
    batch_ids = process_batches_with_openai(
        batch_files, args.openai_api_key, args.model_name, completion_window=args.completion_window
    )

    # Download and process outputs
    all_data = download_and_process_outputs(batch_ids, args.output_dir, args.openai_api_key)

    # Process input and output data from files
    input_files = batch_files
    output_files = [os.path.join(args.output_dir, f"{batch_id}_output.jsonl") for batch_id in batch_ids]
    full_data_exp_df = process_input_output_data_from_files(input_files, output_files)

    # Save the processed data
    output_csv = os.path.join(args.output_dir, 'processed_narrative_roles.csv')
    full_data_exp_df.to_csv(output_csv, index=False)
    print(f"Processing complete. Data saved to '{output_csv}'.")

if __name__ == "__main__":
    main()
