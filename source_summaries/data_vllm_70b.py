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
proj_dir = '/project/jonmay_231/spangher/Projects/conditional-information-retrieval'
config_data = json.load(open(f'{proj_dir}/config.json'))
os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
os.environ['HF_HOME'] = HF_HOME
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

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
        (6) Is_Error: Did we annotate this source in error? This can happen for many reasons, including if a sentence from the webpage was included in the story unintentionally. Answer with "Yes" or "No".
        (7) JUSTIFY your choice for (4) and (5) in 1-2 sentences. 
    Output the summary in a list of python dictionaries with one key per number. Output only the python dictionary.
"""

NARRATIVE_KEYWORD_PROMPT = """
You will receive a news article with each source annotated. 
Your task is to identify a generalizable label that can characterize the narrative role of each source.
The label must be as generalizable as possible and should not be topic-specific. Here are some examples, in short-hand:

[Examples]
Example 1:

Summary: "Is Bumble's initial public offering worth the buzz, and can it compete with industry leader Match Group?"
Source: "Match Group: Match Group is a $45 billion dating conglomerate that runs Match.com, Tinder, and Hinge. The company is valued at $46 billion, or roughly eight times Bumble's current valuation. This source provides a comparison to Bumble and information about the dating app industry."
Your response: 
"Counterpoint": This source is used to compare to the main actor in the news article and provide grounding.

Example 2:
Summary: "What is the significance of Emirates' massive order of 150 Boeing 777X aircraft and how will it impact the airline industry?"
Source: "Dubai Airshow: The deal, worth $56bn (PS33bn) at list prices, was agreed at the Dubai Airshow in November. This source provides context for the deal agreement."
Your response:
"More Context": This source is used to further expand the context offered and offer a visual setting.

Example 3:
Summary: "What major companies were affected by the collapse of Silicon Valley Bank and how much of their assets were tied up with the failed lender?"
Source: "Regulators: Took the unusual step of guaranteeing all deposits at Silicon Valley Bank on Sunday, allowing companies to transfer deposits out of the bank. This source takes action to guarantee deposits"
Your response:
"More Context": This source provides more context for events happening at the time so we can better understand the impacts.

[Instructions]
For each source, describe the role each source plays and determine if it contributes information to the story, or if it was annotated in error.  Include unnamed sources (e.g. "witnesses") if they contribute information.
Generate only ONE summary per source. Group sources that are clearly the same but named slightly differently. For example: "Andrew Dresden" and "Dresden" should be grouped together as one source. "Lao Diplomats" and "Laotian Diplomats" should be grouped together as one source.
Split source annotations that refer to multiple sources into separate summaries. For example: if the annotation is "John and Jane", generate two separate summaries, one for "John" and one for "Jane". 

For each source, provide the following information:
    (1) Name: who the source is.
    (2) Original Name: What their original name(s) are in our annotations.
    (3) Narrative function: Come up with generic keyword label to categorize the narrative function the source playes in the article, and describe it. Return in the format: "LABEL": DESCRIPTION.
    (3) Is_Error: Did we annotate this source in error? This can happen for many reasons, including if a sentence from the webpage was included in the story unintentionally. Answer with "Yes" or "No".

Here's an news article with all of it's sources:

[Article]
```{json_str}```

Your response:
"""

ERROR_PROMPT = """
Here is a news article, with each sentence annotated according to the source of it‛s information:
    ```{json_str}```

    For each source, determine if it contributes information to the story, or if it was annotated in error. 
    Include unnamed sources (e.g. "witnesses") if they contribute information.
    Generate only ONE summary per source. Group sources that are clearly the same but named slightly differently. For example: "Andrew Dresden" and "Dresden" should be grouped together as one source. "Lao Diplomats" and "Laotian Diplomats" should be grouped together as one source.
    Split source annotations that refer to multiple sources into separate summaries. For example: if the annotation is "John and Jane", generate two separate summaries, one for "John" and one for "Jane". 
    
    For each source, provide the following information:
        (1) Name: who the source is.
        (2) Original Name: What their original name(s) are in our annotations.
        (3) Is_Error: Did we annotate this source in error? This can happen for many reasons, including if a sentence from the webpage was included in the story unintentionally. Answer with "Yes" or "No".
"""

#
# (3) Information: Restate the informational content they provide to the article.  Be specific about the facts provided. Make sure you includ to include all the information attributable to that source. (3-4 sentences).

def format_prompt(prompt: str, json_str: str) -> str:
    message = [
        {
            "role": "system",
            "content": "You are an experienced journalist.",
        },

        {
            "role": "user",
            "content": prompt.format(json_str=json_str)
        },
    ]
    formatted_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return formatted_prompt

def load_model(model_name: str):
    torch.cuda.memory_summary(device=None, abbreviated=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LLM(
        model_name,
        dtype=torch.float16,
        tensor_parallel_size=torch.cuda.device_count(),
        download_dir=HF_HOME, # sometimes the distributed model doesn't pay attention to the 
        enforce_eager=True,
        max_model_len=60_000
    )
    return tokenizer, model


def load_full_dataset_from_disk(args):
    # load in the data
    source_df = pd.read_json(
        f'{args.data_dir}/{args.source_data_file}', nrows=args.end_idx, lines=True
    ).iloc[args.start_idx:]
    article_d = load_from_disk(f'{args.data_dir}/all-coref-resolved')

    # process the data into right format: article with annotated sentences
    a_urls_lookup = set(source_df['article_url'])
    filtered_article_d = article_d.filter(lambda x: x['article_url'] in a_urls_lookup, num_proc=10)
    disallowed_quote_types = set(['Other', 'Background/Narrative', 'No Quote'])
    # disallowed_sources = set(['journalist', 'passive-voice'])
    # disallowed_sources = set(['passive-voice'])
    sentences_with_quotes = (
        filtered_article_d
        .to_pandas()
        .merge(source_df, on='article_url')
        [['article_url', 'attributions', 'quote_type', 'sent_lists', ]]
        .explode(['attributions', 'quote_type', 'sent_lists'])
    )

    sentences_with_quotes = (
        sentences_with_quotes.assign(
            attributions=lambda df: df.apply(lambda x:
                x['attributions'] if (
                    (len(x['attributions']) < 50)
                    or (x['quote_type'] not in disallowed_quote_types)
                    # or (x['attributions'] not in disallowed_sources)
            ) else np.nan, axis=1)
        )
    )
    return sentences_with_quotes


def write_to_file(fname, urls, outputs):
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument('--data_dir', type=str, default=f'{proj_dir}/data')
    parser.add_argument('--source_data_file', type=str, default='full-source-scored-data.jsonl')
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--input_data_file', type=str, default=None)
    parser.add_argument('--id_col', type=str, default='article_url')
    parser.add_argument('--source_col', type=str, default='attributions')
    parser.add_argument('--sent_col', type=str, default='sent_lists')
    parser.add_argument('--output_file', type=str, default='sources_data_70b.txt')
    parser.add_argument('--do_info_prompt', action='store_true')
    parser.add_argument('--do_narr_prompt', action='store_true')
    parser.add_argument('--do_narr_key_prompt', action='store_true')
    parser.add_argument('--do_error_prompt', action='store_true')
    args = parser.parse_args()

    if args.input_data_file is None:
        sentences_with_quotes = load_full_dataset_from_disk(args)
    else:
        sentences_with_quotes = pd.read_csv(args.input_data_file, index_col=0)

    tokenizer, model = load_model(args.model)
    # store each article_url, annoted_sentences pair
    # hold the batches
    url_batches, message_batches = [], []
    # each batch
    urls, info_messages, narr_messages, narr_keyword_messages, error_messages = [], [], [], [], []
    for url in sentences_with_quotes[args.id_col].unique():
        one_article = (
            sentences_with_quotes
                .loc[lambda df: df[args.id_col] == url]
                .reset_index(drop=True)
        )

        json_str = (
            one_article[[args.sent_col, args.source_col]]
            .rename(columns={args.sent_col: 'sentence', args.source_col: 'source'})
            .explode(['sentence', 'source'])
            .to_json(lines=True, orient='records')
        )

        info_messages.append(format_prompt(INFO_PROMPT, json_str))
        narr_messages.append(format_prompt(NARRATIVE_PROMPT, json_str))
        narr_keyword_messages.append(format_prompt(NARRATIVE_KEYWORD_PROMPT, json_str))
        error_messages.append(format_prompt(ERROR_PROMPT, json_str))
        urls.append(url)

        if len(info_messages) >= BATCH_SIZE:
            message_batches.append((
                info_messages, 
                narr_messages,
                narr_keyword_messages,
                error_messages
            ))
            url_batches.append(urls)
            info_messages, narr_messages = [], []
            urls = []

    # load the model
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    if args.start_idx is None:
        args.start_idx = 0
    if args.end_idx is None:
        args.end_idx = len(urls)

    # generate the summaries
    start_idx = args.start_idx
    end_idx = start_idx + BATCH_SIZE
    for (info_messages, narr_messages, narr_keyword_messages, error_messages), urls in zip(tqdm(message_batches), url_batches):
        dirname = os.path.dirname(args.output_file)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        fname, fext = os.path.splitext(args.output_file)
        info_fname = f'{fname}__info__{start_idx}_{end_idx}{fext}'
        narr_fname = f'{fname}__narr__{start_idx}_{end_idx}{fext}'
        narr_key_fname = f'{fname}__narr-key__{start_idx}_{end_idx}{fext}'
        err_fname = f'{fname}__err__{start_idx}_{end_idx}{fext}'

        # generate the informational summaries
        if args.do_info_prompt and not os.path.exists(info_fname):
            info_outputs = model.generate(info_messages, sampling_params)
            write_to_file(info_fname, urls, info_outputs)
        
        # generate the narrative summaries            
        if args.do_narr_prompt and not os.path.exists(narr_fname):
            narr_outputs = model.generate(narr_messages, sampling_params)
            write_to_file(narr_fname, urls, narr_outputs)
        
        # generate the narrative keyword summaries
        if args.do_narr_key_prompt and not os.path.exists(narr_key_fname):
            narr_key_outputs = model.generate(narr_keyword_messages, sampling_params)
            write_to_file(narr_key_fname, urls, narr_key_outputs)
        
        # generate the error summaries
        if args.do_error_prompt and not os.path.exists(err_fname):
            err_outputs = model.generate(error_messages, sampling_params)
            write_to_file(err_fname, urls, err_outputs)
        
        # update the indices
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