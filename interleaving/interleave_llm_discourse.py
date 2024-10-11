import json
import gzip
import sys
import os
import logging
import argparse
from tqdm.auto import tqdm
import difflib

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
from Levenshtein import distance as levenshtein_distance
def find_most_overlapping_string(input_string, string_list):
    best_match = "None"
    best_score = float('inf')
    for string in string_list:
        score = levenshtein_distance(input_string, string)
        if score < best_score:
            best_score = score
            best_match = string
    return best_match

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
    text_to_oracle_label = {}
    oracle_file = '/pool001/spangher/alex/conditional-information-retrieval/interleaving/v3_combined_ALL_with_oracle.json.gz'
    with gzip.open(oracle_file, 'r') as file:
        articles = json.load(file)
        for article in articles:
            url = article['url']
            for doc in article['truth']:
                id = url + "#" + doc["Name"]
                oracle_label = doc['llama_label']
                oracle_to_docs[oracle_label].append(id)
                text = doc['Text_embed']
                text_to_oracle_label[text] = oracle_label
    
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

    cluster_list = ['Main Actor', 'Analysis', 'Background Information', 'Subject', 'Expert', 'Data Resource', 'Confirmation and Witness', 'Anecdotes, Examples and Illustration', 'Counterpoint', 'Broadening Perspective', "None"]
    article_order = [url for url, val in url_to_story_lead.items()] #ordering of URLS
    url_to_query_clusters = defaultdict(list)
    # BEGIN INTERLEAVING EXPERIMENT
    for i in range(args.iterations):
        cluster_messages = []
        for url in article_order:
            retrieved_str = ""
            dr_list = url_to_searched_docs[url]
            for source_dict in dr_list:
                source_text = source_dict['text']
                source_type = text_to_oracle_label[source_text]
                retrieved_str += f"{source_type}: "
                retrieved_str += source_text
                retrieved_str += '\n'
            
            index = 0
            past_queries = ""
            for old_query in url_to_past_queries[url]:
                past_queries += f"{index}. "
                past_queries += old_query
                past_queries += '\n'
                index += 1

            story_lead = url_to_story_lead[url]

            prompt = f"""
                Role: You are helping to identify the next source we should consult for a news article.

                We began this investigation with the following question:
                {story_lead}

                We have already consulted a range of sources from different categories, and they have provided us with the following information:
                {retrieved_str}

                These are the questions we have already explored:
                {past_queries}

                Now, we need to determine the next source we should consult, following these steps:
                1. Assess the information already gathered and identify what is still missing.
                2. Review the angles we have explored and determine which perspectives or areas are yet to be covered.
                3. Consider what type of source will help address these informational gaps.
                
                Here are the types of sources in our database:

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

                Identify a source type from the list above that we should consult next.
                Please write the source type under the label "NEW SOURCE:".
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
            cluster_messages.append(message)

        url_to_new_cluster = {}
        if i != 0:
            cluster_response = infer(model=LLM_model, messages=cluster_messages, model_id=args.model, batch_size=100)
            for url, output in zip(article_order, cluster_response):
                llm_cluster = output.split("NEW SOURCE: ")[-1]
                cur_cluster = find_most_overlapping_string(llm_cluster, cluster_list)
                url_to_new_cluster[url] = cur_cluster
                
        query_messages = []
        for url in article_order:
            retrieved_str = ""
            dr_list = url_to_searched_docs[url]
            for source_dict in dr_list:
                source_text = source_dict['text']
                source_type = text_to_oracle_label[source_text]
                retrieved_str += f"{source_type}: "
                retrieved_str += source_text
                retrieved_str += '\n'
            
            index = 0
            past_queries = ""
            for old_query in url_to_past_queries[url]:
                past_queries += f"{index}. "
                past_queries += old_query
                past_queries += '\n'
                index += 1
                
            story_lead = url_to_story_lead[url]
            cluster = url_to_new_cluster.get(url, "None")

            prompt = f"""
                Role: You are helping to identify the next source we should consult for a news article.

                We began this investigation with the following question:
                {story_lead}

                We have already consulted a range of sources, and they have provided us with the following information:
                {retrieved_str}

                These are the questions we have already explored:
                {past_queries}

                Now, we need to determine our next step. Please craft a one-sentence query for the next source we should consult, following these steps:
                1. Assess the information already gathered and identify what is still missing.
                2. Review the angles we have explored and determine which perspectives or areas are yet to be covered.
                3. Consider what type of source will help address these informational gaps.

                We are now particularly interested in this source type:
                {cluster}

                Please provide the one-sentence query under the label "NEW QUERY:".  
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
            query_messages.append(message)
        # Infer the new query using LLM agent
        url_to_new_query = {}
        if i != 0:
            response = infer(model=LLM_model, messages=query_messages, model_id=args.model, batch_size=100)
            for url, output in zip(article_order, response):
                url_to_new_query[url] = output.split("NEW QUERY: ")[-1]
        if i == 0:
            for url in article_order:
                url_to_new_query[url] = url_to_story_lead[url]
        print(f"Query augmentation {i} has been completed")

        interleave_result = []
        print(f"Starting another round of DR search for augmented query {i}")
        for url in tqdm(article_order):
            new_query = url_to_new_query[url]
            article_seen_ids = [d['id'] for d in url_to_searched_docs[url]] #current retrieval pool for this article. Do not include these in search
            cur_oracle = url_to_new_cluster.get(url, "None")

            if i == 0:
                included_id_list = list(included_docs)
            else:
                included_id_list = [id for id in oracle_to_docs[cur_oracle] if id not in article_seen_ids]

            dr_result = dr.search(
                    query=new_query,
                    return_docs=True,
                    include_id_list=included_id_list,
                    cutoff=10)

            combined = list(dr_result)
            combined.extend(url_to_searched_docs[url])
            combined.sort(key=lambda x: -float(x['score']))
            new_top_k = combined
            for source in new_top_k:
                source["score"] = str(source["score"]) #convert to string to write to json file.

            url_to_searched_docs[url] = new_top_k
            url_to_past_queries[url].append(new_query)
            url_to_query_clusters[url].append(cur_oracle)

            one_article = {}
            one_article['url'] = url
            one_article['initial_story'] = url_to_story_lead[url]
            one_article['queries'] = url_to_past_queries[url]
            one_article['query_clusters'] = url_to_query_clusters[url]
            one_article['dr_sources'] = new_top_k
            
            interleave_result.append(one_article)
        
        print(f"DR search for round {i} complete")
        # write to json file with RESULTS from iteration i
        fname = os.path.join(here, f"LLM_discourse2_{i}_SFR_{args.start_idx}_{args.end_idx}.json")
        with open(fname, 'w') as json_file:
            json.dump(interleave_result, json_file, indent=4)