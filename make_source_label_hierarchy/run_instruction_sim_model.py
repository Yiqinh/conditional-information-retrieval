import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import argparse
import pandas as pd

def last_token_pool(
    last_hidden_states: Tensor,
    attention_mask: Tensor
) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


task = 'Given a description of a narrative role, how might it be similar to other narrative roles? Pay less attention to the specific events or topics being described.'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some queries and passages.')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file containing queries')
    parser.add_argument('--column_name', type=str, required=True, help='Column name in the CSV file containing queries')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
    parser.add_argument('--model_name', type=str, required=True, default="Salesforce/SFR-Embedding-2_R", help='Name of the model to use')    
    parser.add_argument('--nrows', type=int, default=None, help='Number of rows to process from the CSV file')
    
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    if args.nrows is not None:
        df = df.head(args.nrows)
        
    queries = df[args.column_name].tolist()
    queries = list(map(lambda x: get_detailed_instruct(task, x), queries))

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)

    max_length = 4096
    batch_dict = tokenizer(
        queries, 
        max_length=max_length, 
        padding=True, 
        truncation=True,
        return_tensors="pt"
    )
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # save embeddings to file
    torch.save(embeddings, args.output_file)
