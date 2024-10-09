import json
import gzip
import sys
import os
import logging
import argparse
from tqdm.auto import tqdm

"""
    Starting from the initial query, returns json files storing the augmented queries 
    and corresponding source retrievals for each iteration.
"""

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

    # BEGIN INTERLEAVE EXPERIMENT SETUP
    url_to_story_lead = {} 
    url_to_truth = {} 
    url_to_searched_docs = {} 
    url_to_past_queries = {} 

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
            url_to_past_queries[url] = []
            url_to_searched_docs[url] = []

    # Add all source documents from each article to the TOTAL pool of sources
    included_docs = set() 
    
    all_articles_file = '/pool001/spangher/alex/conditional-information-retrieval/interleaving/article_data_v3/v3_combined_ALL.json'
    with open(all_articles_file, 'r') as file:
        articles = json.load(file)
        for article in articles:
            url = article['url']
            for doc in article['truth']:
                id = url + "#" + doc["Name"]
                included_docs.add(id)

    print(f"A TOTAL OF {len(included_docs)} INCLUDED IN THE SEARCH")
    #ORACLE SETUP
    from collections import defaultdict
    oracle_to_docs = defaultdict(list)
    url_to_oracle = defaultdict(list)
    oracle_ordering = {'Main Actor': 0, 'Analysis': 1, 'Background Information': 2, 'Subject': 3, 'Expert': 4, 'Data Resource': 5, 
                       'Confirmation and Witness': 6, 'Anecdotes, Examples and Illustration': 7, 'Counterpoint': 8, 'Broadening Perspective': 9, 'Start' : -1}
    
    oracle_file = '/pool001/spangher/alex/conditional-information-retrieval/interleaving/v3_combined_ALL_with_oracle.json.gz'
    with gzip.open(oracle_file, 'r') as file:
        articles = json.load(file)
        for article in articles:
            url = article['url']
            url_to_oracle[url].append('Start')
            for doc in article['truth']:
                id = url + "#" + doc["Name"]
                oracle_label = doc['llama_label']
                oracle_to_docs[oracle_label].append(id)
                url_to_oracle[url].append(oracle_label)
            url_to_oracle[url].sort(key=lambda x: oracle_ordering.get(x, 100))

    # LOAD THE DENSE RETRIEVER
    sys.path.append(os.path.join(os.path.dirname(here), "source_retriever"))
    # needs to be imported here to make sure the environment variables are set before the retriv library sets certain defaults
    from dense_retriever import MyDenseRetriever
    # Sets the retriv base path
    retriv_cache_dir = args.retriv_cache_dir
    logging.info(f"Setting environment variables: RETRIV_BASE_PATH={retriv_cache_dir}")
    os.environ['RETRIV_BASE_PATH'] = retriv_cache_dir

    dr = MyDenseRetriever.load(args.index_name) # DENSE RETRIEVER
    print("loaded the Dense Retriever...")

    # LOAD THE LLM
    helper_dir = os.path.join(os.path.dirname(here), 'helper_functions')
    sys.path.append(helper_dir)
    from vllm_functions import load_model, infer

    LLM_model = load_model(args.model)
    print("Loaded the LLM Model...")

    article_order = [url for url, val in url_to_story_lead.items()] #ordering of URLS
    # BEGIN INTERLEAVING EXPERIMENT
    for i in range(args.iterations):
        messages = []
        for url in article_order:
            retrieved_str = ""
            dr_list = url_to_searched_docs[url]
            index = 0
            for source_dict in dr_list:
                source_text = source_dict['text']
                retrieved_str += f"Source {index}: "
                retrieved_str += source_text
                retrieved_str += '\n'
                index += 1
            
            index = 0
            past_queries = ""
            for old_query in url_to_past_queries[url]:
                past_queries += f"{index}. "
                past_queries += old_query
                past_queries += '\n'
                index += 1

            story_lead = url_to_story_lead[url]
            if i < len(url_to_oracle[url]):
                cur_oracle = url_to_oracle[url][i]
            else:
                cur_oracle = "None"

            prompt = f"""
                You are helping me find relevant and diverse sources for a news article I am working on.

                Here is the question we started out asking at the beginning of our investigation:
                ```{story_lead}```

                We've already interviewed these sources and they've given us this information:
                {retrieved_str}

                We've already considered these questions:
                {past_queries}

                Take into account: We are specifically looking for:
                ```{cur_oracle}```

                Please write a 1-sentence query to help us find our next source. Let's think about this step-by-step:
                1. What information has already been gathered? What information is missing?
                2. What angles have already been explored? What angles are missing?
                3. What kinds of sources would fulfill these missing informational needs?

                Finally, after answering these questions, write the one-sentence query under the label "NEW QUERY".
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
        
        # Infer the new query using LLM agent
        url_to_new_query = {}
        if i != 0:
            response = infer(model=LLM_model, messages=messages, model_id=args.model, batch_size=100)
            for url, output in zip(article_order, response):
                url_to_new_query[url] = output.split("NEW QUERY:")[-1]
        if i == 0:
            for url in article_order:
                url_to_new_query[url] = url_to_story_lead[url]
        print(f"Query augmentation {i} has been completed")

        interleave_result = []

        print(f"Starting another round of DR search for augmented query {i}")
        for url in tqdm(article_order):
            new_query = url_to_new_query[url]
            article_seen_ids = [d['id'] for d in url_to_searched_docs[url]] #current retrieval pool for this article. Do not include these in search
            
            if i < len(url_to_oracle[url]):
                cur_oracle = url_to_oracle[url][i]
            else:
                cur_oracle = "None"

            included_id_list = [id for id in oracle_to_docs.get(cur_oracle, []) if id not in article_seen_ids]
            if i == 0:
                included_id_list = list(included_docs)

            dr_result = dr.search(
                    query=new_query,
                    return_docs=True,
                    include_id_list=included_id_list,
                    cutoff=10)

            # Only taking the top 10 scores from last two retrievals
            combined = list(dr_result)
            combined.extend(url_to_searched_docs[url]) # last 10 sources + new 10 sources retrieved
            combined.sort(key=lambda x: -float(x['score']))
            new_top_k = combined #[:10]
            for source in new_top_k:
                source["score"] = str(source["score"]) #convert to string to write to json file.
            
            one_article = {}
            one_article['url'] = url
            one_article['query'] = new_query
            one_article['dr_sources'] = new_top_k
            one_article['oracle'] = cur_oracle
            
            interleave_result.append(one_article)
            url_to_searched_docs[url] = new_top_k
            url_to_past_queries[url].append(new_query)
        
        print(f"DR search for round {i} complete")
        # write to json file with RESULTS from iteration i
        fname = os.path.join(here, f"ORACLE_iter_{i}_SFR_V3_{args.start_idx}_{args.end_idx}.json")
        with open(fname, 'w') as json_file:
            json.dump(interleave_result, json_file, indent=4)