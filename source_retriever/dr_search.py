import json
import os
import logging
import argparse
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
here = os.path.dirname(os.path.abspath(__file__))
    
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

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
        help="Name of the index to create",
        default="new-index",
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

    dr = MyDenseRetriever.load(args.index_name)

    id_to_label_index = {}
    included_documents = [] #a list of document ids that need to be included

    #get included documents
    f = os.path.join(os.path.dirname(here), 'baseline_queries', 'test_set', 'test_articles.json')
    label_index = 0
    with open(f, 'r') as file:
        articles = json.load(file)
        for url, article in articles.items():
            for id, text in article["sources"].items():
                included_documents.append(id)
                id_to_label_index[id] = label_index
                label_index += 1

    res = {}
    y_true = []
    y_pred = []

    #get search queries
    with open(f, 'r') as file:
        articles = json.load(file)
        for url, article in articles.items():
            my_query = article['query']
            dr_result = dr.search(
                    query=my_query,
                    return_docs=True,
                    include_id_list=included_documents,
                    cutoff=10)
            
            #loop through and update labels, 1 -> document is relevant | document is retrieved, 0 -> document is not relevant | document is not retrieved
            ground_truth = [0 for _ in range(len(included_documents))]
            retriever = [0 for _ in range(len(included_documents))]

            for source in dr_result:
                label = id_to_label_index[source['id']]
                retriever[label] = 1
                source["score"] = str(source["score"]) #convert to string to write to json file.
            
            for id, text in article["sources"].items():
                label = id_to_label_index[id]
                ground_truth[label] = 1
            
            y_true.append(ground_truth)
            y_pred.append(retriever)

            tmp = {}
            tmp["dr"] = dr_result
            tmp["truth"] = article["sources"]
            tmp["query"] = my_query
            res[url] = tmp

    fname = os.path.join(os.path.dirname(here), 'baseline_queries', 'baseline_results', 'retrieved_sources_test_set.json')
    with open(fname, 'w') as json_file:
        json.dump(res, json_file)
    
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    logging.info(f"f1 score is {f1}")
    logging.info(f"precision score is {precision}")
    logging.info(f"recall score is {recall}")
