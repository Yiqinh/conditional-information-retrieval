import json
import sys
import os
import logging
import argparse
from tqdm.auto import tqdm
import gzip

here = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_config', type=str, default=os.path.join(os.path.dirname(here), 'config.json'), help="The path to the json file containing HF_TOKEN")
    parser.add_argument("--index_name", type=str, help="Name of the index to load", default="v3_SFR_MERGED_index")
    parser.add_argument("--retriv_cache_dir", type=str, default=here, help="Path to the directory containing indices")
    parser.add_argument("--iterations", type=int, help="Number of iterations to augment query and retrieve sources", default=10)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")    
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--end_idx", type=int)
    args = parser.parse_args()

    #set huggingface token
    config_data = json.load(open(args.hf_config))
    os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
    #set the proper huggingface cache directory
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    # LOAD THE LLM
    helper_dir = os.path.join(os.path.dirname(here), 'helper_functions')
    sys.path.append(helper_dir)
    from vllm_functions import load_model, infer

    url_to_pred = {}
    file_path = '/pool001/spangher/alex/conditional-information-retrieval/LLM_pooling/oracle_pooling_example.json'
    with open(file_path, 'r') as file:
        articles = json.load(file)
        for article in articles:
            url_to_pred[article['url']] = article['truth']

    url_to_story_lead = {}
    test_set_file = '/pool001/spangher/alex/conditional-information-retrieval/interleaving/article_data_v3/v3_combined_TEST.json'
    with open(test_set_file, 'r') as file:
        articles = json.load(file)
        for i in range(args.start_idx, args.end_idx):
            article = articles[i]
            url = article['url']
            initial_query = article['query']
            if initial_query == "":
                continue
            url_to_story_lead[url] = initial_query
    
    from collections import defaultdict, Counter
    url_to_oracle = defaultdict(list)
    text_to_oracle = {}
    
    oracle_file = '/pool001/spangher/alex/conditional-information-retrieval/interleaving/v3_combined_ALL_with_oracle.json.gz'
    with gzip.open(oracle_file, 'r') as file:
        test_set = json.load(file)
        for article in test_set:
            url = article['url']
            for doc in article['truth']:
                text = doc['Text_embed']
                oracle_label = doc['llama_label']
                url_to_oracle[url].append(oracle_label)
                text_to_oracle[text] = oracle_label
    
    url_to_source_mappings = {}
    article_order = [url for url, val in url_to_pred.items()]
    for url in article_order:
        pred_sources = ""
        index = 1
        for s in url_to_pred[url]:
            pred_sources += f"Source {index}: {s['Text_embed']} "
            pred_sources += f"Narrative role: {text_to_oracle[text]} "
            pred_sources += "\n"
            index += 1
        url_to_source_mappings[url] = pred_sources
    
    messages = []
    for url in article_order:
        counter = Counter(url_to_oracle[url])
        # Format the output to display each source cluster and its count in the article
        formatted_oracles = ""
        for item, count in counter.items():
            formatted_oracles += f'"{item}": {count}\n'

        prompt = f"""
            You are a journalist writing a news article. 
            The main story of the article is:
            {url_to_story_lead[url]}
            
            You are looking for diverse sources to tell a persuasive story.
            Sources serve various narrative roles. I have defined some narrative roles below:

            'Main Actor',
            'Analysis',
            'Background Information',
            'Subject',
            'Expert',
            'Data Resource',
            'Confirmation and Witness',
            'Anecdotes, Examples and Illustration',
            'Counterpoint',
            'Broadening Perspective'

            I have included a large list of sources and their narrative roles.
            Please pick out sources from this list according to this distribution of narrative roles:
            {formatted_oracles}

            Here is the list of sources to pick from:
            {url_to_source_mappings[url]}

            Please output your selection of sources in a Python list of source numbers under the label "OUTPUT".
            For example: 

            OUTPUT: ["Source 1", "Source 2", "Source 3"]

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
        if url == article_order[0]:
            print(prompt)

    LLM_model = load_model(args.model)
    print("Loaded the LLM Model...")
    response = infer(model=LLM_model, messages=messages, model_id=args.model, batch_size=100)

    res = []
    for url, output in zip(article_order, response):
        one_article = {}
        one_article['url'] = url
        one_article['dr_sources'] = url_to_pred[url]
        one_article['story_lead'] = url_to_story_lead[url]
        one_article['LLM_pool'] = output.split("OUTPUT:")[-1]
        one_article['mappings'] = url_to_source_mappings[url]
        res.append(one_article)
    
    with open(os.path.join(here, f"v3_llm_pooling.json"), 'w') as json_file:
        json.dump(res, json_file, indent=4)
    