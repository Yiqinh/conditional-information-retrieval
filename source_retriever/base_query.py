from datasets import load_from_disk
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import json
import logging
import os
import sys

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

    #load in the data
    source_df = pd.read_json(os.path.join(args.data_dir, 'full-source-scored-data.jsonl'), nrows=args.end_idx, lines=True).iloc[args.start_idx:]
    article_d = load_from_disk('all-coref-resolved')

    # process the data into right format: article with annotated sentences
    a_urls_lookup = set(source_df['article_url'])
    filtered_article_d = article_d.filter(lambda x: x['article_url'] in a_urls_lookup, num_proc=10)

    all_articles = filtered_article_d.to_pandas().merge(source_df, on='article_url')

    # store each message/prompt
    messages = []
    urls = []
    for i in range(len(all_articles)):
        one_article_text = all_articles.iloc[i]['article_text']
        one_article_url = all_articles.iloc[i]['article_url']
        prompt = f"""
                        Summarize this article in one sentence. Do NOT contain any source information. Just provide the main idea of the article. Output ONE sentence only.

                        Here is the news article:
                        ```
                        {one_article_text}
                        ```
                    """
        
        message = [
            {
                "role": "system",
                "content": "You are an experienced journalist that can summarize news articles.",
            },

            {
                "role": "user",
                "content": prompt
            },
        ]
        messages.append(message)
        urls.append(one_article_url)

    # load the model and infer
    my_model = load_model(args.model)
    response = infer(model=my_model, messages=messages, model_id=args.model)

    queries = []
    for url, output in zip(urls, response):
        one_query = {
             'url': url,
             'summary': output
        }
        queries.append(one_query)

    if args.start_idx is None:
        args.start_idx = 0
    if args.end_idx is None:
        args.end_idx = len(urls)

    fname = f'article_sum__{args.start_idx}_{args.end_idx}.json'
    fname = os.path.join(os.path.dirname(here), 'baseline_queries', fname)
    with open(fname, 'w') as json_file:
        json.dump(queries, json_file)
