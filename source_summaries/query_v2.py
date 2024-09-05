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
    #parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--hf_home", type=str, default="/project/jonmay_231/spangher/huggingface_cache")
    parser.add_argument('--hf_config', type=str, default=os.path.join(os.path.dirname(here), 'config.json'))
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(here), 'data'))

    args = parser.parse_args()

    config_data = json.load(open(args.hf_config))
    os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
    os.environ['HF_HOME'] = args.hf_home
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    #load in the data
    source_df = pd.read_json(os.path.join(args.data_dir, 'full-source-scored-data.jsonl'), lines=True)
    article_d = load_from_disk(os.path.join(args.data_dir, 'all-coref-resolved'))

    # process the data into right format: article with annotated sentences
    a_urls_lookup = set(source_df['article_url'])
    filtered_article_d = article_d.filter(lambda x: x['article_url'] in a_urls_lookup, num_proc=10)

    all_articles = filtered_article_d.to_pandas().merge(source_df, on='article_url')

    #find urls/articles in test set
    info_dir = os.path.join(here, "v2_info_parsed")
    f = open(os.path.join(info_dir, "v2_test_set.json"))
    test_data = json.load(f)

    test_urls = set()
    for article in test_data:
        test_urls.add(article['url'])

    # store each message/prompt
    messages = []
    urls = []

    for i in range(len(all_articles)):
        if len(messages) == 501:
            break

        one_article_text = all_articles.iloc[i]['article_text'].replace("\n", "")
        one_article_url = all_articles.iloc[i]['article_url']

        if one_article_url not in test_urls:
            continue

        prompt = f"""
                    Output one sentence only. I have pasted a news article below. State the preliminary question the news article answers. 
                    Incorporate the initial story lead and the reason why the journalist started investigating this topic. Please output this one question only.
                    Do not output "here is..."
                    
                    {one_article_text}

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
        messages.append(message)
        urls.append(one_article_url)

    # load the model and infer to get article summaries. 
    my_model = load_model(args.model)
    response = infer(model=my_model, messages=messages, model_id=args.model, batch_size=100)

    queries = []
    print("num urls:", len(urls))
    print("num responses:", len(response))

    for url, output in zip(urls, response):
        one_query = {
             'url': url,
             'query': output
        }
        queries.append(one_query)


    fname = f'article_query_500sample.json'
    fname = os.path.join(here, 'v2_queries', fname)
    with open(fname, 'w') as json_file:
        json.dump(queries, json_file)