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

    #LENGTH ARTICLES IS: 239535
    urls = []
    all_messages = []

    for i in range(args.start_idx, args.end_idx):
        article_text = articles[i]['response']
        prompt = f"""
            I am a journalist and I just wrote an article. You will try to guess the initial question I had before I started investigating this topic. 
            This is the article I wrote:

            ```{article_text}```

            Again, try to guess the initial question or angle I decided to investigate when I started to pursue this article, before interviewing any other sources. 
            Be specific. If there is a specific event, person, fact, or company that seems like it lead to the idea, you can reference it. 
            Pay attention to the angle I took and incorporate that into your question. Please output your answer under the label "ANSWER"

                """
        
        message = [
            {
                "role": "system",
                "content": "You are an experienced journalist",
            },

            {
                "role": "user",
                "content": prompt
            },
        ]
        urls.append(articles[i]['url'])
        all_messages.append(message)

    #Load the LLM
    my_model = load_model(args.model)
    response = infer(model=my_model, messages=all_messages, model_id=args.model, batch_size=100)

    queries = []
    for url, output in zip(urls, response):
        one_query = {
             'url': url,
             'query': output
        }
        queries.append(one_query)

    with open(f"v3_query_{args.start_idx}_{args.end_idx}", 'w') as json_file:
        json.dump(queries, json_file, indent=4)

