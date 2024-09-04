import json
import os
import logging
import argparse
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import statistics

here = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
def get_scores(path: str):
    precision_list = []
    recall_list = []
    f1_list = []

    with open(path, 'r') as file:
        articles = json.load(file)
        for article in articles:
            y_pred = set()
            y_true = set()

            for source in article['dr_sources']:
                id = source['id']
                y_pred.add(id)
            for source in article['sources']:
                y_true.add(article['url']+"#"+source['Name'])
            
            if len(y_true) == 0:
                continue
            
            true_pos = set.intersection(y_pred, y_true)
            n = len(true_pos)

            recall = n / len(y_true)
            precision = n / len(y_pred)

            if (recall + precision) == 0:
                f1 = 0

            else:
                f1 = (2 * precision * recall) / (precision + recall)
            
            recall_list.append(recall)
            precision_list.append(precision)
            f1_list.append(f1)
    
    avg_prec = statistics.mean(precision_list)
    avg_rec = statistics.mean(recall_list)
    avg_f1 = statistics.mean(f1_list)

    print("average precision:", avg_prec)
    print("average recall:", avg_rec)
    print("average f1:", avg_f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hf_config',
        type=str,
        default=os.path.join(os.path.dirname(here), 'config.json'),
        help="The path to the json file containing HF_TOKEN"
    )
    parser.add_argument(
        "--index_name",
        type=str,
        help="Name of the index to load",
        default="v2-test-dense-index",
    )
    # defaults and configs
    parser.add_argument(
        "--retriv_cache_dir",
        type=str,
        default=here,
        help="Path to the directory containing indices"
    )
    parser.add_argument(
        "--huggingface_cache_dir",
        type=str,
        default='/project/jonmay_231/spangher/huggingface_cache',
        help="Path to the directory containing HuggingFace cache"
    )
    args = parser.parse_args()

    #set huggingface token
    config_data = json.load(open(args.hf_config))
    os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]

    #set the proper huggingface cache directory
    hf_cache_dir = args.huggingface_cache_dir
    os.environ['HF_HOME'] = hf_cache_dir
    logging.info(f"Setting environment variables: HF_HOME={hf_cache_dir}")

    # needs to be imported here to make sure the environment variables are set before
    # the retriv library sets certain defaults
    from dense_retriever import MyDenseRetriever

    #sets the retriv base path
    retriv_cache_dir = args.retriv_cache_dir
    logging.info(f"Setting environment variables: RETRIV_BASE_PATH={retriv_cache_dir}")
    os.environ['RETRIV_BASE_PATH'] = retriv_cache_dir

    test_dr = MyDenseRetriever.load("v2-test-dense-index")

    id_to_label_index = {}
    included_documents = [] #a list of document ids that need to be included

    #get included documents
    f = os.path.join(os.path.dirname(here), 'source_summaries', 'v2_info_parsed', 'combined_test_v2.json')

    label_index = 0
    with open(f, 'r') as file:
        articles = json.load(file)
        for article in articles:
            for source in article['sources']:
                id = article['url'] + "#" + source['Name']
                included_documents.append(id)
                id_to_label_index[id] = label_index
                label_index += 1


    res = []

    import random

    #get search queries
    counter = 0
    with open(f, 'r') as file:
        articles = json.load(file)
        random.shuffle(articles)
        for article in articles:
            my_query = article['query']

            dr_result = test_dr.search(
                    query=my_query,
                    return_docs=True,
                    include_id_list=included_documents,
                    cutoff=10)

            for source in dr_result:
                source["score"] = str(source["score"]) #convert to string to write to json file.
            
            one_article = {}
            one_article['url'] = article['url']
            one_article['sources'] = article['sources']
            one_article['dr_sources'] = dr_result
            one_article['query'] = my_query

            res.append(one_article)
            counter += 1

            if counter == 50:
                break # small sample test n=1000
  
    fname = os.path.join(here, 'v2_search_res', 'v2_search_test_1000.json')
    with open(fname, 'w') as json_file:
        json.dump(res, json_file)
    
    get_scores(fname)