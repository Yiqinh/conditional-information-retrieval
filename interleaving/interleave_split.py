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
    parser.add_argument("--index_name", type=str, help="Name of the index to load", default="v2-test-dense-index")
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
    url_to_query = {} # url to the initial query / initial story lead
    url_to_truth = {} # url to the ground truth sources
    url_to_searched_docs = {} # url to the first set of 10 sources retrieved using the INITIAL query

    included_docs = set() # pool of all ground truth documents from the test set to be considered

    file_path = os.path.join(os.path.dirname(here), "source_retriever", "v2_search_res", "v2_search_test_prompt1_all.json")

    with open(file_path, 'r') as file:
        articles = json.load(file)
        for i in range(args.start_idx, args.end_idx):
            article = articles[i]
            """
            STRUCTURE OF ARTICLE
            [
                {
                    "url": "www.url.com", 
                    "sources": 
                        [
                            {"Name": "Rebecca Leber", "Original Name": "Rebecca Leber", "Information": "No information provided."}, 
                            {"Name": "Ron", "Original Name": "ron Said \"Black Lives Matter\"?Big", "Information": "No information provided."} 
                        ], 
                    "dr_sources": 
                        [
                            {"id": "time.com/5761206/iran-plane-crash/#Qasem Soleimani", "text": "He was Iran's top military general who was killed in an American drone strike.", "score": "0.76743096"}, 
                            {"id": "www.theguardian.com/us-news/2022/sep/13/ken-starr-dead-clinton-lewinsky#book", "text": "Trump feared assassination by Iran as revenge for Suleimani's death.", "score": "0.62700194"} 
                        ], 
                    "query": "Who was Qassem Soleimani, the powerful Iranian commander killed in a US airstrike, and what was his significance in the Middle East and to the US?"
                    },
            ] 
            """
            url = article['url']
            initial_query = article['query']
            truth = article['sources']
            first_search = article['dr_sources']

            url_to_query[url] = initial_query
            url_to_truth[url] = truth
            url_to_searched_docs[url] = first_search

            #add all source documents from each article to the TOTAL pool of ground truth
            for doc in truth:
                id = url + "#" + doc["Name"]
                included_docs.add(id)

    #LOAD THE DENSE RETRIEVER
    sys.path.append(os.path.join(os.path.dirname(here), "source_retriever"))
    # needs to be imported here to make sure the environment variables are set before
    # the retriv library sets certain defaults
    from dense_retriever import MyDenseRetriever
    #sets the retriv base path
    retriv_cache_dir = args.retriv_cache_dir
    logging.info(f"Setting environment variables: RETRIV_BASE_PATH={retriv_cache_dir}")
    os.environ['RETRIV_BASE_PATH'] = retriv_cache_dir

    dr = MyDenseRetriever.load("v2-test-dense-index") # DENSE RETRIEVER FOR THE TEST SET
    print("loaded the Dense Retriever...")

    #LOAD THE LLM
    helper_dir = os.path.join(os.path.dirname(here), 'helper_functions')
    sys.path.append(helper_dir)
    from vllm_functions import load_model, infer

    LLM_model = load_model(args.model)
    #response = infer(model=my_model, messages=messages, model_id=args.model, batch_size=100)
    print("Loaded the LLM Model...")

    article_order = [url for url, val in url_to_query.items()] #ordering of URLS

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
            
            story_lead = url_to_query[url]
            prompt = ("I am a journalist and you are my helpful assistant. We are working on a story proposed by my editor. " 
                        "We need to think about sources to interview to complete the article. I am providing you with the initial query we asked ourselves when looking for sources to interview. " 
                        "I am also providing you with the sources we interviewed and what information they provided to the story. "
                        "Mimicking the format, I want you to write a new query to help us think about the next informational needs of the story. Please format this query in one sentence. "
                        "Think about this step by step: \n"
                        
                        "- What are these sources that we have already interviewed providing to the story? \n"
                        "- What are we missing from the story? \n"
                        "- What kinds of sources would complete our informational needs if we interviewed them? \n"

                        "You may include your thinking in the output. \n"

                        "We are working on this story: \n" 
                        f"{story_lead} \n"

                        "We have already interviewed these sources: \n"
                        f"{retrieved_str} \n"

                        "Please write the 1 sentence query under the label “QUERY 2” below your thinking."
            )
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
        response = infer(model=LLM_model, messages=messages, model_id=args.model, batch_size=100)
        print(f"Query augmentation {i} has been completed")

        url_to_new_query = {}

        for url, output in zip(article_order, response):
            url_to_new_query[url] = output.split("QUERY 2:")[-1]

        interleave_result = []

        print(f"Starting another round of DR search for augmented query {i}")
        for url in article_order:
            new_query = url_to_new_query[url]
            
            article_seen_ids = [] #Put all of the past source retrievals for this article in a list to avoid duplicate retrieval
            for d in url_to_searched_docs[url]:
                article_seen_ids.append(d['id'])
 
            included_id_list = [id for id in included_docs if id not in article_seen_ids]

            dr_result = dr.search(
                    query=new_query,
                    return_docs=True,
                    include_id_list=included_id_list,
                    cutoff=10)

            for source in dr_result:
                source["score"] = str(source["score"]) #convert to string to write to json file.
            
            one_article = {}
            one_article['url'] = url
            one_article['query'] = new_query
            one_article['dr_sources'] = dr_result

            url_to_searched_docs[url].extend(dr_result)
            interleave_result.append(one_article)
        
        print(f"DR search for round {i} complete")
        #write to json file with RESULTS from iteration i
        fname = os.path.join(here, f"iter_{i}_search_results_{args.start_idx}_{args.end_idx}.json")
        with open(fname, 'w') as json_file:
            json.dump(interleave_result, json_file, indent=4)
        