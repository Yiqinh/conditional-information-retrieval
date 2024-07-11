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
    parser.add_argument(
        '--embedding_model',
        type=str,
        default="Salesforce/SFR-Embedding-2_R",  # "sentence-transformers/all-MiniLM-L6-v2", #
        help="The model to use for generating embeddings"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to use for inference"
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
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=None,  # 4096
        help="The dimension of the embeddings"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,  # 32768,
        help="Maximum sequence length for the model"
    )
    parser.add_argument(
        "--batch_size_to_index",
        type=int,
        help="Batch size for indexing",
        default=1,
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

    # set up index
    dr = MyDenseRetriever(
        index_name=args.index_name,
        model=args.embedding_model,
        normalize=True,
        max_length=args.max_seq_length,
        embedding_dim=args.embedding_dim,
        device=args.device,
        use_ann=True,
    )

    summary_files = ["sources_data_70b__0_10000.json", 
                     "sources_data_70b__100000_110000.json", 
                     "sources_data_70b__10000_20000.json", 
                     "sources_data_70b__110000_120000.json", 
                     "sources_data_70b__120000_130000.json",	
                     "sources_data_70b__200000_200100.json",		
                     "sources_data_70b__200000_205000.json",	
                     "sources_data_70b__205000_210000.json",	
                     "sources_data_70b__210000_220000.json",
                     "sources_data_70b__220000_230000.json",
                     "sources_data_70b__230000_240000.json",
                     "sources_data_70b__240000_250000.json",		
                     "sources_data_70b__310000_320000.json",
                     "sources_data_70b__320000_330000.json",
                     "sources_data_70b__330000_340000.json",
                     "sources_data_70b__80000_90000.json",	
                     "sources_data_70b__90000_100000.json"]
    collection = []

    for file in summary_files:
        file_path = os.path.join(os.path.dirname(here), 'source_summaries', 'json_summaries', file)
        with open(file_path, 'r') as file:
            data = json.load(file)
            for article in data:
                for id, summary in article['sources'].items():
                    new_source_embedding = {"id": id, "text": summary}
                    collection.append(new_source_embedding)

    dr.index(
        collection=collection,  # File kind is automatically inferred
        batch_size=args.batch_size_to_index,  # Default value
        show_progress=True,  # Default value
    )



