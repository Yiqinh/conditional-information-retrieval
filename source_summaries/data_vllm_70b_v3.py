from datasets import load_from_disk
import pandas as pd
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import unicodedata

import os
import json
import torch
import ast
import logging
import re 

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

from vllm import LLM,  SamplingParams
from transformers import AutoTokenizer
import os
here = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f'{here}/../config.json'))
os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

BATCH_SIZE = 500
CLEAN_ARTICLE_TEXT_PROMPT = """
You are a helpful editor assistant. I scraped this news article from the web and removed HTML tags, however, 
there is a lot of extraneous text from the HTML page that remains. Please only extract text related to the news article.
Do NOT miss any article text. Copy over the text exactly. Here is the news article:

```{article_text}```

Return only the text of the news article, nothing else.
"""

SOURCE_EXTRACTION_PROMPT = """
You are a helpful news assistant. Here is a news article:

```{news_article}```

Please summarize each informational source providing information in the article. 
Include unnamed or passively expressed sources (e.g. "witnesses", "price signals") if there is information attributable to them.
Include any facts that might have come from the source.
Make sure each source you return refer to just one source. For example: if "John and Jane" both contribute the same information, generate two separate summaries, one for "John" and one for "Jane". 
Generate only ONE summary per source.

For each source, provide the following information:
    (1) Name: just the name of the source.
    (2) Biography: A brief biography of the source mentioned in the article.
    (3) Information: Restate the facts provided by the source. Be as SPECIFIC and be as VERBOSE as possible. 
        Contextualize ALL the information the source describes. State the full names of all people, places, events and ideas mentioned and
        everything the source says with AS MUCH BACKGROUND INFORMATION from the article so I can fully understand the information
        the source is giving. I will look at each source independently without looking at any others, so help me understand the context.

Here are some examples:
example 1:
{{ "Name": "Supermarkets around the country",
   "Biography": "Retail stores that sell food and other household items",
   "Information": "Supermarkets around the country alerted shoppers that prices are likely to continue going up due to the avian flu outbreak, with eggs now average $2.88 per dozen, up 52% since the first confirmed case of avian influenza in February."
}}

example 2:
{{
  "Name": "The article's author (unnamed)",
  "Biography": "The author of the article",
  "Information": "The author stated that Wing, which is collaborating with FedEx and Walgreens on drone delivery, was the first to receive a limited Part 135 certificate. Wing is launching operations in Virginia this month, and the Standard certification allows UPS to send an unlimited number of drones to the skies, for their cargo load to exceed 55 pounds and for them to fly at night."
}}

example 3:
{{
   "Name": "Delta's customers",
   "Biography": "People who travel with Delta Air Lines",
   "Information": "Delta's customers suggested that they preferred more space on flights amid the COVID-19 pandemic, and they continue to tell Delta that more space provides more peace of mind."
}}

example 4:
{{
   "Name": "European Union countries",
   "Biography": "Countries that are part of the European Union",
   "Information": "European Union countries are working on adopting copyright rules that allow news companies and publishers to negotiate payments with large tech companies like Facebook, Microsoft and Google that use their content on their platforms."
}}

Output the summary in a list of python dictionaries as in the examples. Don't say anything else.
"""


NARRATIVE_KEYWORD_PROMPT = """
You will receive a news article and a set of sources to examine in that article.

For each source in the list, provide the following information, once per source:
    (1) Name: Exactly copy the name of the source.
    (2) Narrative Function: Give a generic keyword label to categorize the narrative role the source playes in the article. 
    Infer why the author used the source, and a generalizable statement about the role they play in the article.
    Don't just summarize their identity. Return in the format: "LABEL": DESCRIPTION.

Here are example outputs. Again, your main task here is to identify a generalizable label that can characterize the narrative role of each source and why the author used them. 

[Examples]
Example 1:

{{
    "Name": "Match Group",
    "Narrative Function": "\"Counterpoint\": This source is used to compare to the main actor in the news article and provide grounding."
}}

Example 2:

{{
    "Name": "Dubai Airshow",
    "Narrative Function": "\"More Context\": This source is used to further expand the context offered and offer a visual setting."
}}

Example 3:
{{

    "Name": "Ann Gough",
    "Narrative Function": "\"Victim\": This source provides the voice of a user for the product, giving us a personal view of the harm caused by the event.
}}

[Instructions]

Now it's your turn. Here is a news article:

```{news_article}```

Please examine the narrative role of each of the following sources: 

```[{target_sources}]```

For each source, answer the questions above. Output the summary in a list of python dictionaries as in the examples. Don't say anything else.
"""


CENTRALITY_AND_PERSPECTIVE_PROMPT = """
You will receive a news article and a set of sources to examine in that article.
    
    For each source, provide the following information:
        (1) Name: who the source is.
        (2) Perspective: What is their perspective on the main events of the article? Choose as many labels as fit from: ["Authoritative", "Informative", "Supportive", "Skeptical", "Against", "Neutral"].
        (3) Centrality: How central is this source to the main events of the article? Choose from "High", "Medium", "Low".
        (4) Is_Error: Did we annotate this source in error? This can happen for many reasons, including if a sentence from the webpage was included in the story unintentionally. Answer with "Yes" or "No".

Here is a news article:

```{news_article}```

Please examine the role of each of the following sources: 

```[{target_sources}]```

For each source, answer the questions above. Output the summary in a list of python dictionaries as in the examples. Don't say anything else.
"""


def robust_extract_json_str(lm_string):
    
    if not lm_string:
        return None
    # Use regular expressions to search for list brackets across multiple lines
    match = re.search(r'\[.*?\]', lm_string, re.DOTALL)
    if match:
        lm_string = match.group(0)
    try:
        return json.loads(lm_string)
    except:
        try:
            return ast.literal_eval(lm_string)
        except:
            pass
    logging.error(f"Could not extract json string from: {lm_string}")
    return None


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
    return (
        filtered_article_d
        .to_pandas()
        .merge(source_df, on='article_url')
        [['article_url', 'article_text']]
    )


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


def batchify_dataframe(df, batch_size):
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
    return [df.iloc[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]


def get_text_and_sources(text_fname, summaries_fname):
    cleaned_articles = pd.read_json(text_fname, lines=True).rename(columns={'response': 'article_text'})
    source_summaries = pd.read_json(summaries_fname, lines=True).rename(columns={'response': 'source_summaries'})
    combined_df = cleaned_articles.merge(source_summaries, on='url')
    combined_df['target_sources'] = (
        combined_df['source_summaries']
            .apply(robust_extract_json_str)
            .apply(lambda x: [s['Name'] for s in x] if x else [])
            .apply(lambda x: ', '.join(map(lambda y: '"%s"' % y , x)))
    )
    return combined_df



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # meta-llama/Llama-3.1-8B-Instruct
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument('--data_dir', type=str, default=f'{here}/../data')
    parser.add_argument('--source_data_file', type=str, default='full-source-scored-data.jsonl')
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--input_data_file', type=str, default=None)
    parser.add_argument('--id_col', type=str, default='article_url')
    parser.add_argument('--text_col', type=str, default='article_text')
    parser.add_argument('--output_file', type=str, default='sources_data_70b.txt')
    # prompts
    parser.add_argument('--do_article_gen', action='store_true')
    parser.add_argument('--do_source_summ', action='store_true')
    parser.add_argument('--do_narr_key_prompt', action='store_true')
    parser.add_argument('--do_cent_prompt', action='store_true')

    args = parser.parse_args()

    if args.input_data_file is None:
        article_df = load_full_dataset_from_disk(args)
    else:
        article_df = pd.read_csv(args.input_data_file, index_col=0)

    if args.start_idx is None:
        args.start_idx = 0
    if args.end_idx is None:
        args.end_idx = len(article_df)
    
    # load the model
    sampling_params = SamplingParams(temperature=0.1, max_tokens=4096)
    tokenizer, model = load_model(args.model)
    # store each article_url, annoted_sentences pair
    # hold the batches
    url_batches, message_batches = [], []
    df_batches = batchify_dataframe(article_df, BATCH_SIZE)

    # generate the summaries
    start_idx = args.start_idx
    end_idx = start_idx + BATCH_SIZE
    for df in tqdm(df_batches):
        dirname = os.path.dirname(args.output_file)
        if (dirname != '') and not os.path.exists(dirname):
            os.makedirs(dirname)

        out_dirname, out_fname = os.path.split(args.output_file)
        if (out_dirname != '') and not os.path.exists(out_dirname):
            os.makedirs(out_dirname)
            if args.do_article_gen:
                os.makedirs(f'{out_dirname}/article_text', exist_ok=True)
            if args.do_source_summ:
                os.makedirs(f'{out_dirname}/summaries', exist_ok=True)
            if args.do_narr_key_prompt:
                os.makedirs(f'{out_dirname}/narrative-keyword', exist_ok=True)
            if args.do_cent_prompt:
                os.makedirs(f'{out_dirname}/centrality-perspective-v2', exist_ok=True)

        fname, fext = os.path.splitext(out_fname)
        text_fname = f'{out_dirname}/article_text/{fname}__article_text__{start_idx}_{end_idx}{fext}'
        summaries_fname = f'{out_dirname}/summaries/{fname}__summaries__{start_idx}_{end_idx}{fext}'
        narr_key_fname = f'{out_dirname}/narrative-keyword/{fname}__narrative-keyword__{start_idx}_{end_idx}{fext}'
        cent_persp_fname = f'{out_dirname}/centrality-perspective-v2/{fname}__centrality-perspective__{start_idx}_{end_idx}{fext}'

        # clean the article text
        if args.do_article_gen and not os.path.exists(text_fname):
            logging.info(f"Generating cleaned article text for batch {start_idx} to {end_idx}")
            clean_prompts = df[args.text_col].apply(lambda x: CLEAN_ARTICLE_TEXT_PROMPT.format(article_text=x)).tolist()
            cleaned_article_outputs = model.generate(clean_prompts, sampling_params)
            write_to_file(text_fname, df[args.id_col], cleaned_article_outputs)

        # generate the informational summaries
        if args.do_source_summ and not os.path.exists(summaries_fname):
            logging.info(f"Generating source summaries for batch {start_idx} to {end_idx}")
            if not os.path.exists(text_fname):
                raise ValueError("You need to generate the cleaned article text first.")
            cleaned_articles = pd.read_json(text_fname, lines=True)
            source_prompts = cleaned_articles['response'].apply(lambda x: SOURCE_EXTRACTION_PROMPT.format(news_article=x)).tolist()
            source_outputs = model.generate(source_prompts, sampling_params)
            write_to_file(summaries_fname, cleaned_articles['url'], source_outputs)

        # generate the narrative keyword summaries
        if args.do_narr_key_prompt and not os.path.exists(narr_key_fname):
            logging.info(f"Generating narrative keyword summaries for batch {start_idx} to {end_idx}")
            if not  os.path.exists(text_fname) and os.path.exists(summaries_fname):
                raise ValueError("You need to generate the cleaned article text and source summaries first.")
            combined_df = get_text_and_sources(text_fname, summaries_fname)
            narr_key_prompts = (
                combined_df
                    .apply(lambda x: NARRATIVE_KEYWORD_PROMPT.format(news_article=x['article_text'], target_sources=x['target_sources']), axis=1)
                    .tolist()
            )
            narr_key_outputs = model.generate(narr_key_prompts, sampling_params)
            write_to_file(narr_key_fname, combined_df['url'], narr_key_outputs)
        
        # generate the centrality and perspective annotations
        if args.do_cent_prompt and not os.path.exists(cent_persp_fname):
            logging.info(f"Generating centrality and perspective annotations for batch {start_idx} to {end_idx}")
            if not  os.path.exists(text_fname) and os.path.exists(summaries_fname):
                raise ValueError("You need to generate the cleaned article text and source summaries first.")
            combined_df = get_text_and_sources(text_fname, summaries_fname)
            cent_prompts = (
                combined_df
                    .apply(lambda x: CENTRALITY_AND_PERSPECTIVE_PROMPT.format(news_article=x['article_text'], target_sources=x['target_sources']), axis=1)
                    .tolist()
            )
            cent_persp_outputs = model.generate(cent_prompts, sampling_params)
            write_to_file(cent_persp_fname, combined_df['url'], cent_persp_outputs)
        
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


    python data_vllm_70b_v3.py \
      --start_idx 0 \
      --end_idx 50 \
      --id_col article_url \
      --text_col article_text \
      --data_dir ../data \
      --source_data_file   full-source-scored-data.jsonl.gz \
      --output_file  ../data/v3_source_summaries/test_sources.txt \
      --do_article_gen \
      --do_source_summ \
      --do_narr_key_prompt \
      --do_cent_prompt \
      --model meta-llama/Llama-3.1-8B-Instruct
      --model meta-llama/Meta-Llama-3.1-8B-Instruct


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

"""
# count completed files on disk

import glob
import pandas as pd

summ_df = pd.Series(glob.glob('test_sources__summaries__*')).to_frame('filename')
summ_df = (
    summ_df
        .assign(file_chunks=lambda df: df['filename'].str.split('__').str.get(2).str.replace('.txt', ''))
        .assign(start_idx=lambda df: df['file_chunks'].str.split('_').str.get(0).astype(int))
        .assign(end_idx=lambda df: df['file_chunks'].str.split('_').str.get(1).astype(int))
)
expected_starts = pd.Series(list(range(0, 350_000, 500))).to_frame('expected_starts')

summ_df = summ_df.merge(expected_starts, left_on='start_idx', right_on='expected_starts', how='right')

"""