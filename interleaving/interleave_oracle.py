import json
import sys
import os
import logging
import argparse
from tqdm.auto import tqdm

"""
Starting from the initial query, returns json files storing the augmented queries and corresponding source retrievals for each iteration.

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
    parser.add_argument("--index_name", type=str, help="Name of the index to load")
    parser.add_argument("--retriv_cache_dir", type=str, default=here, help="Path to the directory containing indices")
    parser.add_argument("--iterations", type=int, help="Number of iterations to augment query and retrieve sources", default=10)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--end_idx", type=int)
    args = parser.parse_args()

    #set huggingface token
    config_data = json.load(open(args.hf_config))
    os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]

    #set the proper huggingface cache directory
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    # BEGIN INTERLEAVE EXPERIMENT SETUP
    url_to_categories = {}
    file_path = os.path.join(os.path.dirname(here), "source_summaries", "v2_info_parsed", "combined_test_prompt1_v2__with_oracle.json")
    with open(file_path, 'r') as file:
        articles = json.load(file)
        for article in articles:
            url = article['url']
            category_string = ""
            for source in article['sources']:
                if source.get('oracle_label') != None:
                    category_string += "* "
                    category_string += source['oracle_label']
                    category_string += "\n"
            url_to_categories[url] = category_string

    url_to_story_lead = {} # url to the initial query / initial story lead
    url_to_truth = {} # url to the ground truth sources
    url_to_searched_docs = {} # url to the first set of 10 sources retrieved using the INITIAL query
    url_to_past_queries = {} #url to the previous queries used

    file_path = os.path.join(os.path.dirname(here), "source_summaries", "v2_info_parsed", "combined_test_prompt1_v2.json")
    with open(file_path, 'r') as file:
        articles = json.load(file)
        for i in range(args.start_idx, args.end_idx):
            article = articles[i]
            url = article['url']
            initial_query = article['query']
            if (initial_query == "") or (url not in url_to_categories):
                continue

            url_to_story_lead[url] = initial_query
            url_to_past_queries[url] = []
            url_to_searched_docs[url] = []

    print(f"DOING EXPERIMENT ON {len(url_to_story_lead)} ARTICLES")

    #add all source documents from each article to the TOTAL pool of sources
    included_docs = set() 
    file_path = os.path.join(os.path.dirname(here), "source_summaries", "v2_info_parsed", "combined_train_prompt1_v2.json")
    with open(file_path, 'r') as file:
        articles = json.load(file)
        for article in articles:
            url = article['url']
            for doc in article['sources']:
                id = url + "#" + doc["Name"]
                included_docs.add(id)
    
    file_path = os.path.join(os.path.dirname(here), "source_summaries", "v2_info_parsed", "combined_test_prompt1_v2.json")
    with open(file_path, 'r') as file:
        articles = json.load(file)
        for article in articles:
            url = article['url']
            for doc in article['sources']:
                id = url + "#" + doc["Name"]
                included_docs.add(id)

    print(f"A TOTAL OF {len(included_docs)} INCLUDED IN THE SEARCH")

    #LOAD THE DENSE RETRIEVER
    sys.path.append(os.path.join(os.path.dirname(here), "source_retriever"))
    # needs to be imported here to make sure the environment variables are set before
    # the retriv library sets certain defaults
    from dense_retriever import MyDenseRetriever
    #sets the retriv base path
    retriv_cache_dir = args.retriv_cache_dir
    logging.info(f"Setting environment variables: RETRIV_BASE_PATH={retriv_cache_dir}")
    os.environ['RETRIV_BASE_PATH'] = retriv_cache_dir

    dr = MyDenseRetriever.load("v2-ALL-dense-index") # DENSE RETRIEVER
    print("loaded the Dense Retriever...")

    #LOAD THE LLM
    helper_dir = os.path.join(os.path.dirname(here), 'helper_functions')
    sys.path.append(helper_dir)
    from vllm_functions import load_model, infer

    LLM_model = load_model(args.model)
    #response = infer(model=my_model, messages=messages, model_id=args.model, batch_size=100)
    print("Loaded the LLM Model...")

    article_order = [url for url, val in url_to_story_lead.items()] #ordering of URLS

    #BEGIN INTERLEAVING EXPERIMENT
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
                past_queries += f"Angle {i}: "
                past_queries += old_query
                past_queries += '\n'
                index += 1

            story_lead = url_to_story_lead[url]
            categories = url_to_categories[url]

            prompt = f"""I am a journalist and you are my helpful assistant.
        We need to think about sources to interview to complete the article. I am providing you with:
        * the initial story lead we are working on.
        * the sources we have already interviewed and the information they provided to the story.
        * the angles and queries we have already investigated.
        I want you to write a new query to help us think about the next informational needs of the story.

        KEEP IN MIND:
        Stories like this tend to use sources in the following categories: 
        {categories}

        Here are the definitions of the categories:
        "Credibility and Engagement": These sources contribute to building trust, influence audience understanding, and enhancing the reliability of information by providing personal accounts, insights and warnings.
        "Authoritative Source": These sources provide verified, expert-backed information that enhances the trustworthiness and reliability of the content they support.
        "Background Information": These sources provide broader context to events, helping readers understand the main topic in the context of what is going on and grasp peripheral details.
        "Analysis and Criticism": These sources are experts and insiders. They offer deep insights, critiques, and multifaceted interpretations of complex issues and decisions across diverse fields.
        "Central Figure": These sources are the main individual, company, or entity featured in news articles, serving as the focal point by providing important information, insights, actions, decisions, and statements.
        "Examples and Illustration": These sources provide background information in different ways, enhancing the main content for a more comprehensive grasp of topics discussed.
        "Alternative Viewpoints": These sources offer diverse perspectives, critical analyses, and opposing opinions to provide a more balanced understanding and challenge dominant views across various topics.
        "Expert Analysis": These sources offer a broad and detailed range of expert analyses and insights, enriching the article's depth and credibility.
        "Expert Insights": These sources collectively provide essential data, context, analysis, and credibility across diverse topics, enhancing depth and understanding in journalistic articles and ensuring comprehensive and accurate reporting.
        "Author Perspective": These sources that offer analysis and personal insights, usually from the author of the article.
        "Anecdotes": These sources offer diverse and specific examples that highlight different responses, adaptations, successes, challenges, innovations, and impacts across sectors, providing insights into broader themes and concepts.
        "Peripheral Context": These sources enrich the main content by offering deeper context than typical background sources.

        Please format this query in one sentence. Think about this step by step. Keep in mind the source categories that we mentioned above:
        - What are these sources that we have already interviewed providing to the story?
        - What are we missing from the story?
        - What other angles could we look at for this story?
        - What kinds of sources would complete our informational needs?
        You may include your thinking in the output.

        We are working on this story:
        {story_lead}
        We have already considered these angles:
        {past_queries}
        We have already interviewed these sources:
        {retrieved_str}
        Please write the 1 sentence query under the label 'NEW QUERY' below your thinking."""

            message = [
                {
                    "role": "system",
                    "content": "I am a journalist and you are my helpful assistant.",
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ]
            messages.append(message)
        
        #Infer AUGMENTED query using LLM agent
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

            included_id_list = [id for id in included_docs if id not in article_seen_ids]

            dr_result = dr.search(
                    query=new_query,
                    return_docs=True,
                    include_id_list=included_id_list,
                    cutoff=10)

            #only taking the top 10 scores from last two retrievals
            combined = list(dr_result)
            combined.extend(url_to_searched_docs[url]) # last 10 sources + new 10 sources retrieved
            combined.sort(key=lambda x: -float(x['score']))

            new_top_k = combined[:10]

            for source in new_top_k:
                source["score"] = str(source["score"]) #convert to string to write to json file.
            
            one_article = {}
            one_article['url'] = url
            one_article['query'] = new_query
            one_article['dr_sources'] = new_top_k
            
            interleave_result.append(one_article)
            url_to_searched_docs[url] = new_top_k
            url_to_past_queries[url].append(new_query)
        
        print(f"DR search for round {i} complete")
        #write to json file with RESULTS from iteration i
        fname = os.path.join(here, f"iter_{i}_ORACLE_SFR_ALL_{args.start_idx}_{args.end_idx}.json")
        with open(fname, 'w') as json_file:
            json.dump(interleave_result, json_file, indent=4)