import json
import os
import logging
import argparse
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


    dr = MyDenseRetriever.load("all_baseline-sources")

    f = os.path.join(os.path.dirname(here), 'baseline_queries', 'test_set', 'test_articles.json')
    with open(here, 'r') as file:
        articles = json.load(file)
        for article in articles:
            my_query = article['query']
            break
    
    included_documents = [] #a list of document ids that need to be included

    res = dr.search(
                query=my_query,
                return_docs=True,
                cutoff=10)
    
    print(type(res))
    print(res)

    """
    [
        {
        "id": "doc_2",
        "text": "Just like witches at black masses",
        "score": 0.9536403
        },
        {
        "id": "doc_1",
        "text": "Generals gathered in their masses",
        "score": 0.6931472
        }
    ]
    """

    """
    Use Sklearn to calculate scores:
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    """

    id_to_label_index = {}

    #loop through and update labels, 1 -> document is relevant | document is retrieved, 0 -> document is not relevant | document is not retrieved
    ground_truth = []
    retrieved = []