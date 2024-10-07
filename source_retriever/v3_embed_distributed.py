import pandas as pd
import torch
from retriv.dense_retriever.encoder import Encoder
import numpy as np 
from retriv.retriv.paths import embeddings_folder_path, encoder_state_path, index_path
import os 


def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'mps' if torch.backends.mps.is_available() else device
    return device


def create_embeddings(
        filtered_df: pd.DataFrame, 
        chunk_num: int,
        index_name: str,
        batch_size: int = 64,
        model_name: str = "bert-base-uncased"
):
    """
    Creates embeddings for the text column in the dataframe between start_idx and stop_idx.
    
    Args:
        filtered_df (pd.DataFrame): The dataframe containing a 'text' column.
        chunk_num (int): The chunk number to use for saving the embeddings.
        index_name (str): The name of the index to use for saving the embeddings.
        batch_size (int): The batch size to use for generating embeddings.
        model_name (str): The name of the model to use for generating embeddings.
    
    Returns:
        List of embeddings for the selected range of texts.
    """
    encoder = Encoder(
        index_name=index_name, 
        model=model_name, 
        device=get_device()
    )
    embeddings = encoder.bencode(filtered_df['text'].tolist(), batch_size=batch_size)
    np.save(os.path.join(embeddings_folder_path(index_name), f"chunk_{chunk_num}.npy"), embeddings)
    return embeddings


# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="data.csv",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="The name of the model to use for generating embeddings"
    )

    parser.add_argument(
        "--index_name",
        type=str,
        default="test_index",
        help="The name of the index to use"
    )
    parser.add_argument(
        "--chunk_num",
        type=int,
        default=0,
        help="The chunk number to use"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="The starting index to use"
    )
    parser.add_argument(
        "--stop_idx",
        type=int,
        default=2,
        help="The stopping index to use"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size to use"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    filtered_df = df.iloc[args.start_idx:args.stop_idx]

    embeddings = create_embeddings(
        filtered_df,
        args.chunk_num,
        args.index_name, 
        args.batch_size, 
        args.model_name
    )

    print(embeddings)
