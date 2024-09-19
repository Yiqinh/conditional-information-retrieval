import os 
import jsonlines
import tqdm
import openai
from utils_basic import batchify
import re
import openai
import pandas as pd


# os.environ['OPENAI_API_KEY'] = open()
client = openai.OpenAI()

# 
# hit OpenAI API 
# 
def call_openai_completions(prompt, model_name='gpt-4o', max_tokens=1000):
    """
    Call the OpenAI completions API with a given prompt.
    
    Args:
    prompt (str): The prompt to send to the API.
    model_name (str): The name of the model to use.
    max_tokens (int): The maximum number of tokens to generate.
    
    Returns:
    dict: The response from the OpenAI API.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful journalist assistant."},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content



# 
# batch API data processing 
#
def format_batch_prompt(prompts, model_name):
    """
    Format a list of prompts for the OpenAI Batch API.
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


def write_prompts_to_files(all_prompts, output_dir, batch_size=40000, model_name='gpt-4'):
    """
    Take a list of the text-prompts, format them for the OpenAI Batch API, and write them to JSON Lines files (in batches).
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


def process_batches_with_openai(batch_files, completion_window='24h'):
    """
    Upload them to OpenAI's batch processing API and create batch processing jobs.
    """
    batch_ids = []
    for batch_file in tqdm(batch_files):
        with open(batch_file, 'rb') as f:
            batch_input_file = client.files.create(file=f, purpose="batch")
        batch_input_file_id = batch_input_file.id
        # Create a batch processing job
        batch_response = client.batches.create(
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
    all_data = []
    for batch_id in tqdm(batch_ids):
        batch_info = client.batches.retrieve(id=batch_id)
        output_file_id = batch_info['output_file_id']
        if output_file_id:
            output_file = client.files.content(id=output_file_id)
            output_file_path = os.path.join(output_dir, f"{batch_id}_output.jsonl")
            with open(output_file_path, 'w') as f:
                f.write(output_file.decode('utf-8'))
            # Read and process the output file
            with jsonlines.open(output_file_path) as reader:
                for obj in reader:
                    all_data.append(obj)
    # Further processing of all_data can be done here
    return all_data


def process_input_output_data_from_openai_files(input_files, output_files):
    """
    Join the batches of input prompts with OpenAI's responses.

    Parameters:
        input_files (list): List of input prompt files.
        output_files (list): List of output response files.

    Returns:
        pd.DataFrame: Processed data.
    """
    # read input files
    all_input_data = []
    for f in input_files:
        data = list(jsonlines.open(f))
        data_df = pd.DataFrame(data)
        data_df['filename'] = os.path.basename(f)
        all_input_data.append(data_df)

    # read output files
    all_output_data = []
    for f in output_files:
        data = list(jsonlines.open(f))
        data_df = pd.DataFrame(data)
        data_df['filename'] = os.path.basename(f)
        all_output_data.append(data_df)

    # concatenate input and output data
    all_input_data_df = pd.concat(all_input_data)
    all_output_data_df = pd.concat(all_output_data)

    # Merge input and output data
    all_input_data_df['input'] = all_input_data_df['body'].str.get('messages').str.get(1).str.get('content')
    all_input_data_df['f_key'] =  all_input_data_df['filename'].str.replace('.jsonl', '')
    all_output_data_df['output'] = all_output_data_df['response'].str.get('body').str.get('choices').str.get(0).str.get('message').str.get('content')
    all_output_data_df['f_key'] = all_output_data_df['filename'].str.split('__').str.get(0)

    # merge input and output data
    full_data_df = all_input_data_df[['custom_id', 'f_key', 'input']].merge(
        all_output_data_df[['custom_id', 'f_key', 'output']], on=['custom_id', 'f_key']
    )

    # further processing 
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