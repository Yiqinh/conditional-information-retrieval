#! /bin/bash
#SBATCH --job-name=run_emb_trained_model
#SBATCH --time=10:00:00
#SBATCH --partition=isi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G


python -c """
from utils_basic import process_source_data
from sentence_transformers import SentenceTransformer
import numpy as np 
#
source_df = process_source_data(data_dir='/project/jonmay_231/spangher/Projects/conditional-information-retrieval/source_summaries/v2_err_raw')
source_df = source_df.loc[lambda df: ~df['Is_Error']].loc[lambda df: df['Narrative Function'].notnull()]
to_label = source_df['Narrative Function'].dropna().tolist()
model = SentenceTransformer('/scratch1/spangher/conditional-information-retrieval/make_source_label_hierarchy/models/mpnet-BASE-all-nli-triplet/trained-model/')
#
embs = model.encode(to_label, batch_size=1, show_progress_bar=True)
np.savez_compressed('/scratch1/spangher/conditional-information-retrieval/make_source_label_hierarchy/similarity_training_data/all_labeled_embeddings_trained_model.npz', embs)
"""
