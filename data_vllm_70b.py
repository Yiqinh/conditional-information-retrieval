from datasets import load_from_disk
import pandas as pd
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import unicodedata

import os
import json
import torch
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

from vllm import LLM,  SamplingParams
from transformers import AutoTokenizer
import os
HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"
config_data = json.load(open('config.json'))
os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
os.environ['HF_HOME'] = HF_HOME

BATCH_SIZE = 100

def load_model(model: str):
    torch.cuda.memory_summary(device=None, abbreviated=False)
    model = LLM(
        model,
        dtype=torch.float16,
        tensor_parallel_size=torch.cuda.device_count(),
        download_dir=HF_HOME, # sometimes the distributed model doesn't pay attention to the 
        enforce_eager=True
    )
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    args = parser.parse_args()


    #load in the data
    source_df = pd.read_json(
        f'{args.data_dir}/full-source-scored-data.jsonl', nrows=args.end_idx, lines=True
    ).iloc[args.start_idx:]
    article_d = load_from_disk('all-coref-resolved')

    # process the data into right format: article with annotated sentences
    a_urls_lookup = set(source_df['article_url'])
    filtered_article_d = article_d.filter(lambda x: x['article_url'] in a_urls_lookup, num_proc=10)
    disallowed_quote_types = set(['Other', 'Background/Narrative', 'No Quote'])
    disallowed_sources = set(['journalist', 'passive-voice'])
    sentences_with_quotes = (
        filtered_article_d
            .to_pandas()
            .merge(source_df, on='article_url')
            [['article_url', 'attributions', 'quote_type', 'sent_lists',]]
            .explode(['attributions', 'quote_type', 'sent_lists'])
    )

    sentences_with_quotes = (sentences_with_quotes
        .assign(attributions=lambda df:
            df.apply(lambda x:
                       x['attributions'] if (
                               (len(x['attributions']) < 50) or
                               (x['quote_type'] not in disallowed_quote_types) or
                               (x['attributions'] not in disallowed_sources)) else np.nan, axis=1)
        )
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    # store each article_url, annoted_sentences pair
    # hold the batches
    message_batches = []
    url_batches = []
    # each batch 
    messages = []
    urls = []
    for url in sentences_with_quotes['article_url'].unique():
        one_article = (
            sentences_with_quotes
                .loc[lambda df: df['article_url'] == url]
                .reset_index(drop=True)
                )

        json_str = (
            one_article[['sent_lists', 'attributions']]
            .rename(columns={'sent_lists': 'sentence', 'attributions': 'source'})
            .to_json(lines=True, orient='records')
        )

        prompt = f"""
                        Here is a news article, with each sentence annotated according to the source of it's information:
                        ```
                        {json_str}
                        ```

                        Please summarize each source, based on our source annotations. 
                        Tell me in one paragraph per source: (1) who the source is (2) what informational content they provide to the article. 
                        Only rely on the annotations I have provided, don't identify additional sources. 
                        Generate only ONE summary per source. Group sources that are clearly the same but named slightly differently.
                        That is, summarize the SAME source if it occurs in multiple source annotations. 
                    """
        message = [
            {
                "role": "system",
                "content": "You are an experienced journalist.",
            },

            {
                "role": "user",
                "content": prompt
            },
        ]
        formatted_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        messages.append(formatted_prompt)
        urls.append(url)

        if len(messages) >= BATCH_SIZE:
            message_batches.append(messages)
            url_batches.append(urls)
            messages = []
            urls = []

    # load the model
    model = load_model(args.model)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    if args.start_idx is None:
        args.start_idx = 0
    if args.end_idx is None:
        args.end_idx = len(urls)

    # generate the summaries
    start_idx = args.start_idx
    end_idx = start_idx + BATCH_SIZE
    for messages, urls in zip(tqdm(message_batches), url_batches):
        fname = f'sources_data_70b__{start_idx}_{end_idx}.txt'
        outputs = model.generate(messages, sampling_params)
        with open(fname, 'wb') as file:
            for url, output in zip(urls, outputs):
                response = output.outputs[0].text
                response = unicodedata.normalize('NFKC', response)
                if response and url:
                    file.write(url.encode('utf-8')
                    file.write(b'\n')
                    file.write(b'{')
                    file.write(response.encode('utf-8'))
                    file.write(b'}')
                    file.write(b'\n')
                    file.write(b'\n')
        start_idx = end_idx
        end_idx = start_idx + BATCH_SIZE
