import os
import json
import logging
import argparse
import torch

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
here = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_config', type=str, default=os.path.join(os.path.dirname(here), 'config.json'), help="The path to the json file containing HF_TOKEN")
    parser.add_argument('--embedding_model', type=str, default="Salesforce/SFR-Embedding-2_R", help="The model to use for generating embeddings")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use for inference")
    # defaults and configs
    parser.add_argument("--retriv_cache_dir", type=str, default=here, help="Path to the directory containing indices")
    parser.add_argument("--huggingface_cache_dir", type=str, default='/project/jonmay_231/spangher/huggingface_cache', help="Path to the directory containing HuggingFace cache")
    parser.add_argument('--embedding_dim', type=int, default=None, help="The dimension of the embeddings")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Maximum sequence length for the model")
    parser.add_argument("--batch_size_to_index", type=int, help="Batch size for indexing", default=1)
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
    from retriv import SparseRetriever

    #sets the retriv base path
    retriv_cache_dir = args.retriv_cache_dir
    logging.info(f"Setting environment variables: RETRIV_BASE_PATH={retriv_cache_dir}")
    os.environ['RETRIV_BASE_PATH'] = retriv_cache_dir

   #building the collection of sources
    file_path = os.path.join(here, "v3_combined_ALL.json")
    with open(file_path, 'r') as json_file:
        articles = json.load(json_file)

    all_sources = []
    for article in articles:
        for source in article['truth']:
            one_source = {}
            id = article['url'] + "#" + source['Name']
            text = source['Text_embed']
            one_source['id'] = id
            one_source['text'] = text
            all_sources.append(one_source)

    from retriv import SparseRetriever

    sr = SparseRetriever(
        index_name="v3-ALL-BM25-index",
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
    
    sr.index(
        collection=all_sources,  # File kind is automatically inferred
        show_progress=True,         # Default value       
    )