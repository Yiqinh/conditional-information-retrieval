from datasets import load_from_disk
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import json
import logging
import os
import sys

import csv

here = os.path.dirname(os.path.abspath(__file__))
helper_dir = os.path.join(os.path.dirname(here), 'helper_functions')
sys.path.append(helper_dir)

from vllm_functions import load_model, infer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

here = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument("--hf_home", type=str, default="/project/jonmay_231/spangher/huggingface_cache")
    parser.add_argument('--hf_config', type=str, default=os.path.join(os.path.dirname(here), 'config.json'))
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(here), 'data'))
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)

    args = parser.parse_args()

    config_data = json.load(open(args.hf_config))
    os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
    os.environ['HF_HOME'] = args.hf_home
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    #load in the data
    data_dir = args.data_dir

    article_d = load_from_disk('all-coref-resolved')

    candidate_df = pd.read_csv('data/news_articles_covering_the_same_press_releases.csv')
    pr_url_set = set(candidate_df['Target URL'].drop_duplicates().values)
    filtered_pr_d = article_d.filter(lambda x: x['article_url'] in pr_url_set, num_proc=10)
    filtered_pr_df = filtered_pr_d.to_pandas()

    # store each message/prompt
    messages = []
    urls = []
    for i in range(len(filtered_pr_df)):
        one_article_text = "".join(filtered_pr_df.iloc[i]['sent_lists'])
        one_article_url = filtered_pr_df.iloc[i]['article_url']


        prompt = ("A press release is a communication announcing a story to the public, which is deliberately sent to journalists or media publishers in the hope they will publish the news contained in it."
                  "I will paste a press release below. In a well formed paragraph, provide a concise and comprehensive summary of the given press release." 
                  
                  "Here is the press release: \n" 
                  f"{one_article_text} \n")
                  
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
        messages.append(message)
        urls.append(one_article_url)


    # load the model and infer to get article summaries. 
    my_model = load_model(args.model)
    response = infer(model=my_model, messages=messages, model_id=args.model, batch_size=100)

    data = [["article_url", "pr_summary"]]

    for url, output in zip(urls, response):
        if output != "":
            output = output.split('\n\n')[-1]
            data.append([url, output])

    # store each summary/query in a csv
    save_path = os.path.join(here, 'pr_summary.csv')
    with open(save_path, 'w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        # Write the data rows
        writer.writerows(data)
