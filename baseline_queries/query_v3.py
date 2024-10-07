import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os
import sys
import json
import torch
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
here = os.path.dirname(os.path.abspath(__file__))
helper_dir = os.path.join(os.path.dirname(here), 'helper_functions')
sys.path.append(helper_dir)
from vllm_functions import load_model, infer
here = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument('--hf_config', type=str, default=os.path.join(os.path.dirname(here), 'config.json'))
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--end_idx", type=int)
    args = parser.parse_args()

    config_data = json.load(open(args.hf_config))
    os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

    articles = []
    directory = '/pool001/spangher/alex/conditional-information-retrieval/data/v3_source_summaries/article_text'
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            for line in file:
                # Parse the JSON string into a dictionary directly
                json_obj = json.loads(line)
                if len(json_obj['response']) > 300:
                    articles.append(json_obj)
    
    print(len(articles))
    print(articles[50])

    # urls = []
    # messages = []

    # for i in range(args.start_idx, args.end_idx):
    #     article_text = articles[i]['response']

    #     prompt = f"""
    #             Output one sentence only. I have pasted a news article below. State the preliminary question the news article answers. 
    #             Incorporate the initial story lead and the reason why the journalist started investigating this topic. Please output this one question only.
    #             Do not output "here is..."
                    
    #             {one_article_text}

    #             """
        
    #     message = [
    #         {
    #             "role": "system",
    #             "content": "You are an experienced journalist",
    #         },

    #         {
    #             "role": "user",
    #             "content": prompt
    #         },
    #     ]

                
    #Load the LLM
    #my_model = load_model(args.model)
    #response = infer(model=my_model, messages=messages, model_id=args.model, batch_size=100)

