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
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

BATCH_SIZE = 50
INFO_PROMPT = """
Here is a news article, with each sentence annotated according to the source of it‛s information:
    ```{json_str}```

    Please summarize each source. Include unnamed sources (e.g. "witnesses"). Include any facts that might have come from the source, even if we didn't label it.
    Generate only ONE summary per source. Group sources that are clearly the same but named slightly differently. For example: "Andrew Dresden" and "Dresden" should be grouped together as one source. "Lao Diplomats" and "Laotian Diplomats" should be grouped together as one source.
    Split source annotations that refer to multiple sources into separate summaries. For example: if the annotation is "John and Jane", generate two separate summaries, one for "John" and one for "Jane". 
    
    For each source, provide the following information:
        (1) Name: who the source is.
        (2) Original Name: What their original name(s) are in our annotations.
        (3) Information: Restate the facts provided by the source. Be as SPECIFIC and be as VERBOSE as possible. 
            Contextualize all events the source describes, names mentioned and everything the source says with AS MUCH BACKGROUND INFORMATION relating to the main events of the article so I can fully understand the information the source is giving.
            For example, don't just say "the crash", say "the plane crash carrying key senior Laotian government officials"
            Or for another example, don't just say "There were about 20 passengers on board", say "there were 20 passengers on board the plane crash carrying key senior Laotian government officials."  
    Output the summary in a list of python dictionaries with one key per number. Don't say anything else.
"""

NARRATIVE_PROMPT = """
Here is a news article, with each sentence annotated according to the source of it‛s information:
    ```{json_str}```

    Please describe the role each source plays. Include unnamed sources (e.g. "witnesses"). Include any facts that might have come from the source, even if we didn't label it.
    Generate only ONE summary per source. Group sources that are clearly the same but named slightly differently. For example: "Andrew Dresden" and "Dresden" should be grouped together as one source. "Lao Diplomats" and "Laotian Diplomats" should be grouped together as one source.
    Split source annotations that refer to multiple sources into separate summaries. For example: if the annotation is "John and Jane", generate two separate summaries, one for "John" and one for "Jane". 
    
    For each source, provide the following information:
        (1) Name: who the source is.
        (2) Original Name: What their original name(s) are in our annotations.
        (3) Narrative function: What is their narrative function in the article? (1-2 sentences)
        (4) Perspective: What is their perspective on the main events of the article? Choose from "Authoritative", "Informative", "Supportive", "Skeptical", "Against", "Neutral".
        (5) Centrality: How central is this source to the main events of the article? Choose from "High", "Medium", "Low".
        (6) JUSTIFY your choice for (4) and (5) in 1-2 sentences. 
    Output the summary in a list of python dictionaries with one key per number. Output only the python dictionary.
"""

#
# (3) Information: Restate the informational content they provide to the article.  Be specific about the facts provided. Make sure you includ to include all the information attributable to that source. (3-4 sentences).


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
    parser.add_argument('--input_data_file', type=str, default=None)
    parser.add_argument('--id_col', type=str, default='article_url')
    parser.add_argument('--source_col', type=str, default='source')
    parser.add_argument('--sent_col', type=str, default='sent_lists')
    parser.add_argument('--output_file', type=str, default='sources_data_70b.txt')
    args = parser.parse_args()

    if args.input_data_file is None:
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
    else:
        sentences_with_quotes = pd.read_csv(args.input_data_file, index_col=0)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    # store each article_url, annoted_sentences pair
    # hold the batches
    message_batches = []
    url_batches = []
    # each batch 
    messages = []
    urls = []
    for url in sentences_with_quotes[args.id_col].unique():
        one_article = (
            sentences_with_quotes
                .loc[lambda df: df[args.id_col] == url]
                .reset_index(drop=True)
        )

        json_str = (
            one_article[[args.sent_col, args.source_col]]
            .rename(columns={args.sent_col: 'sentence', args.source_col: 'source'})
            .to_json(lines=True, orient='records')
        )

        message = [
            {
                "role": "system",
                "content": "You are an experienced journalist.",
            },

            {
                "role": "user",
                "content": PROMPT.format(json_str=json_str)
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
        fname, fext = os.path.splitext(args.output_file)
        fname = f'{fname}_{start_idx}_{end_idx}{fext}'
        outputs = model.generate(messages, sampling_params)
        with open(fname, 'wb') as file:
            for url, output in zip(urls, outputs):
                response = output.outputs[0].text
                response = unicodedata.normalize('NFKC', response)
                if response and url:
                    output = {}
                    output['url'] = str(url)
                    output['response'] = str(response)
                    file.write(json.dumps(output).encode('utf-8'))
                    file.write(b'\n')
        start_idx = end_idx
        end_idx = start_idx + BATCH_SIZE



"""
import attrdict
args = attrdict.AttrDict()
args.id_col = 'doc_id'
args.source_col = 'head'
args.sent_col = 'sent'
args.output_file = 'annotated_sources_summarized.txt'
args.input_data_file = 'full-training-df.csv'
args.model = 'meta-llama/Meta-Llama-3-70B-Instruct'
args.start_idx = None
args.end_idx = None


    python data_vllm_70b.py \
      --start_idx 0 \
      --end_idx 5 \
      --id_col  doc_id \
      --source_col  head \
      --sent_col  sent \
      --output_file  annotated_sources_summarized.txt \
      --input_data_file  full-training-df.csv


"""

"""
original prompt:

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