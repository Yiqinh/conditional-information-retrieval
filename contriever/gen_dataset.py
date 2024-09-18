import json
import os
import logging
import argparse
import pdb
from tqdm.auto import tqdm
import random
from retriv import SparseRetriever

here = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def main(f):
    print("processing", f)
    
    sr = SparseRetriever(
        index_name="ft-search",
        model="bm25",
        min_df=1,
        tokenizer="whitespace",
        stemmer="english",
        stopwords="english",
        do_lowercasing=True,
        do_ampersand_normalization=True,
        do_special_chars_normalization=True,
        do_acronyms_normalization=True,
        do_punctuation_removal=True,
        )

    source_set = set()
    collection = []

    with open(f, 'r') as file:
        articles = json.load(file)
        for article in articles:
            for source in article['sources']:
                source_set.add(source['Name'])  # title, text, score, title_score, passage_id
                id = article['url'] + "#" + source['Name']
                collection.append({
                    "id": id,
                    "text": source['Information'],
                    "title": source['Name']
                })
    with open('train_collection.jsonl', 'w') as outfile:
        for entry in collection:
            json.dump(entry, outfile)
            outfile.write('\n')
    
    sr = sr.index_file(
        path="train_collection.jsonl",
        show_progress=True,         
        )
                
    res = []

    with open(f, 'r') as file:
        articles = json.load(file)
        for article in tqdm(articles, desc="article source search"):
            my_query = article['query']
            positive_ctxs = []
            for source in article['sources']:
                positive_ctxs.append({
                    "title": source['Name'],
                    "text": source['Information']
                })
            dr_result = sr.search(
                    query=my_query,
                    return_docs=True,
                    cutoff=30)
            hard_negative_ctxs_set = set()
            hard_negative_ctxs = []
            
            positive_ctxs_set = set([s['Name'] for s in article['sources']])
            hard_negative_ctxs_set = set([s['title'] for s in dr_result])
            hard_negative_ctxs_set = hard_negative_ctxs_set - positive_ctxs_set

            for source in dr_result:
                if source['title'] not in hard_negative_ctxs_set:
                    continue
                hard_negative_ctxs.append({
                    "title": source['title'],
                    "text": source['text']
                })
            
            one_article = {}
            one_article['question'] = my_query
            one_article['positive_ctxs'] = positive_ctxs
            
            # pdb.set_trace()
            negative_ctxs_set = source_set - hard_negative_ctxs_set - positive_ctxs_set
            negative_ctxs_collection = [c for c in collection if c['title'] in negative_ctxs_set]

            negative_ctxs = random.sample(negative_ctxs_collection, 50)
            # some adjustment
            for neg in negative_ctxs:
                if 'id' not in neg:
                    continue
                neg.pop('id')

            # pdb.set_trace()
            one_article['negative_ctxs'] = hard_negative_ctxs[:max(10, len(hard_negative_ctxs))] + negative_ctxs

            res.append(one_article)

  
  
    fname = os.path.basename(f)
    with open(f"/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/ft_contriever_{fname}", 'w') as json_file:
        json.dump(res, json_file)
    


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
    # config_data = json.load(open(args.hf_config))
    # os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]

    #set the proper huggingface cache directory
    hf_cache_dir = args.huggingface_cache_dir
    os.environ['HF_HOME'] = hf_cache_dir
    logging.info(f"Setting environment variables: HF_HOME={hf_cache_dir}")

    # needs to be imported here to make sure the environment variables are set before
    # the retriv library sets certain defaults
    # from dense_retriever import MyDenseRetriever


    #sets the retriv base path
    retriv_cache_dir = args.retriv_cache_dir
    logging.info(f"Setting environment variables: RETRIV_BASE_PATH={retriv_cache_dir}")
    os.environ['RETRIV_BASE_PATH'] = retriv_cache_dir
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    # main("/project/jonmay_231/spangher/Projects/conditional-information-retrieval/source_summaries/v2_info_parsed/combined_train_prompt1_v2.json")
    main("/project/jonmay_231/spangher/Projects/conditional-information-retrieval/source_summaries/v2_info_parsed/combined_test_prompt1_v2.json")
    # main("../combined_test_prompt1_v2.json")
