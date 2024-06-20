from datasets import load_from_disk
import pandas as pd
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          pipeline,
                          set_seed)
import os
import json
import torch

def load_model(model_id):
    config_data = json.load(open('config.json'))
    os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto")

    return model, tokenizer


def infer(model, tokenizer, messages):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
        ).to(model.device)
        
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
)
    response = outputs[0][input_ids.shape[-1]:-1]
    return tokenizer.decode(response, skip_special_tokens=True)

if __name__ == "__main__":

    #load in the data
    data_dir = './data'
    source_df = pd.read_json(f'{data_dir}/full-source-scored-data.jsonl.gz', lines=True, compression='gzip', nrows=100)
    article_d = load_from_disk(f'{data_dir}/all-coref-resolved')
    
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

    #store each article in a map
    article_map = {}
    url_map = {}
    for i in range(len(sentences_with_quotes.index.unique())):
    #for i in range(2):
        article_map[i] = sentences_with_quotes.loc[sentences_with_quotes.index == i][['sent_lists', 'attributions']].to_json(lines=True, orient='records')
        url_map[i] = sentences_with_quotes.loc[99].iloc[0]['article_url']

    
    #load the model
    model, tokenizer = load_model("meta-llama/Meta-Llama-3-8B-Instruct")

    # loop through and create prompts for each article
    for i in range(len(article_map)):
        article = article_map[i]

        prompt = f"""
                Here is a news article, with each sentence annotated according to the source of it's information:
                ```
                {article}
                ```

                Please summarize each of our source annotations. Tell me in one paragraph per source: (1) who the source is (2) what informational content they provide to the article. 
                Only rely on the annotations I have provided, don't identify additional sources.
            """

        message = [
            {
                "role": "system",
                "content": "You are an experienced journalist.",
            },

            {"role": "user",
            "content": prompt},
            ]
        
        response = infer(model, tokenizer, message)
        with open('output.txt', 'w') as file:
            file.write(url_map[i])
            file.write('\n')
            file.write(response)
            file.write('\n')
