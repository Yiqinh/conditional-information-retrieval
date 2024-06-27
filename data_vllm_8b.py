from datasets import load_from_disk
import pandas as pd
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import os
import json
import torch

from vllm import LLM,  SamplingParams
from transformers import AutoTokenizer

def load_model(model: str):
    config_data = json.load(open('config.json'))
    os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
    os.environ['HF_HOME'] = "/project/jonmay_231/spangher/huggingface_cache"
    torch.cuda.memory_summary(device=None, abbreviated=False)
    model = LLM(model, dtype=torch.float16)
    return model



def infer(model, messages):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    formatted_prompt =  tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    output = model.generate(formatted_prompt, sampling_params)

    return output[0].outputs[0].text

if __name__ == "__main__":

    #load in the data
    data_dir = './data'
    source_df = pd.read_json(f'{data_dir}/full-source-scored-data.jsonl.gz', lines=True, compression='gzip', nrows=100)
    article_d = load_from_disk('all-coref-resolved')

    #process the data into right format: article with annotated sentences
    filtered_article_d = article_d.filter(lambda x: x['article_url'] in set(source_df['article_url']), num_proc=10)
    disallowed_quote_types = set(['Other', 'Background/Narrative', 'No Quote'])
    sentences_with_quotes = (
        filtered_article_d
            .to_pandas()
            .merge(source_df, on='article_url')
            [['article_url', 'attributions', 'quote_type', 'sent_lists',]]
            .explode(['attributions', 'quote_type', 'sent_lists'])
    )

    sentences_with_quotes = (sentences_with_quotes
        .assign(attributions=lambda df:
                df.apply(lambda x: x['attributions'] if x['quote_type'] not in disallowed_quote_types else np.nan, axis=1)
        )
    )

    #store each article_url, annoted_sentences pair
    articles = []
    for url in sentences_with_quotes['article_url'].unique():
        one_article = (
            sentences_with_quotes
                .loc[lambda df: df['article_url'] == url]
                .reset_index(drop=True)
                )

        json_str = one_article[['sent_lists', 'attributions']].to_json(lines=True, orient='records')
        articles.append((url, json_str))

    #load the model
    model = load_model("meta-llama/Meta-Llama-3-8B-Instruct")

    # loop through and create prompts for each article
    for article in articles:
        json_str = article[1]
        url = article[0]

        prompt = f"""
                Here is a news article, with each sentence annotated according to the source of it's information:
                ```
                {json_str}
                ```

                Please summarize each of our source annotations. Tell me in one paragraph per source: (1) who the source is (2) what informational content they provide to the article. 
                Only rely on the annotations I have provided, don't identify additional sources. Generate only ONE summary per source. That is, summarize the SAME source if it occurs in multiple source annotations.
            """

        message = [
            {
                "role": "system",
                "content": "You are an experienced journalist.",
            },

            {"role": "user",
            "content": prompt},
            ]

        response = infer(model, message)

        with open('sources_data_8b.txt', 'a') as file:
            file.write(url)
            file.write('\n')
            file.write('{')
            file.write(response)
            file.write('}')
            file.write('\n')
            file.write('\n')

