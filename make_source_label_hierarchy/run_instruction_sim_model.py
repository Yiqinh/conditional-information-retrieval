import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np

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
    parser.add_argument('--text_column_name', type=str, required=True, help='Column name in the CSV file containing queries')
    parser.add_argument('--is_error_column_name', type=str, default='Is_Error', help='Column name in the CSV file containing error rows to filter out')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
    parser.add_argument('--model_name', type=str, default="Salesforce/SFR-Embedding-2_R", help='Name of the model to use')    
    parser.add_argument('--nrows', type=int, default=None, help='Number of rows to process from the CSV file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    if args.is_error_column_name is not None:
        df = df[df[args.is_error_column_name] == 'No']

    if args.nrows is not None:
        df = df.head(args.nrows)

    queries = df[args.text_column_name].tolist()
    queries = list(map(lambda x: get_detailed_instruct(task, x), queries))
    
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, device_map=args.device)
    max_length = 4096
    embeddings = []    
    for i in tqdm(range(0, len(queries), args.batch_size), desc="Processing batches"):
        batch = queries[i:i + args.batch_size]
        batch_dict = tokenizer(
            batch, 
            max_length=max_length, 
            padding=True, 
            truncation=True,
            return_tensors="pt",
        ).to(args.device)
        
        with torch.no_grad():
            outputs = model(**batch_dict)
        
        batch_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings.append(batch_embeddings)
    
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # convert embeddings to numpy and save to file
    embeddings = embeddings.cpu().numpy()
    np.save(args.output_file, embeddings)


""" example command
    python run_instruction_sim_model.py \
        --csv_file similarity_training_data/source-df-to-label-df.csv \
        --text_column_name "Narrative Function" \
        --nrows 100 \
        --output_file similarity_training_data/source_label_hierarchy_train_embeddings.pt
"""    
