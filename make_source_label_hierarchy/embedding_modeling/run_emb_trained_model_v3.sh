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
import pandas as pd
import sys
sys.path.insert(0, '../')
#
INPUT_DATA_PATH='/project/jonmay_231/spangher/Projects/conditional-information-retrieval/data/v3_sources/v3_combined_sources.csv.gz'
MODEL_PATH='/project/jonmay_231/spangher/Projects/conditional-information-retrieval/make_source_label_hierarchy/embedding_modeling/models/mpnet-BASE-all-nli-triplet/trained-model/'
OUTPUT_PATH='/project/jonmay_231/spangher/Projects/conditional-information-retrieval/make_source_label_hierarchy/embedding_modeling/v3_embeddings__from_trained_model.npz'
#
#
#
source_df = pd.read_csv(INPUT_DATA_PATH)
source_df = source_df.loc[lambda df: df['Is_Error'] =='No'].loc[lambda df: df['Narrative Function'].notnull()]
to_label = source_df['Narrative Function'].dropna().tolist()
model = SentenceTransformer(MODEL_PATH)
#
embs = model.encode(to_label, batch_size=1, show_progress_bar=True)
np.savez_compressed(OUTPUT_PATH, embs)
"""