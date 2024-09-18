import os
import statistics
import pandas as pd
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv('info_narr_queryv1_df.csv.gz', compression='gzip')

source = []
query = []

all_sims = []

meds = []
lows = []
highs = []

for i in tqdm(range(len(df)), desc="Num Sources"):
    source.append(str(df.iloc[i]['Information']))
    query.append(str(df.iloc[i]['V1 Query']))
    embeddings1 = model.encode(source)
    embeddings2 = model.encode(query)
    # Compute cosine similarities
    similarities = model.similarity(embeddings1, embeddings2)
    sim = similarities.tolist()[0][0]
    all_sims.append(sim)

    if df.iloc[i]['Centrality'] == 'Medium':
        meds.append(sim)
    elif df.iloc[i]['Centrality'] == 'Low':
        lows.append(sim)
    elif df.iloc[i]['Centrality'] == 'High':
        highs.append(sim)

    source.pop()
    query.pop()

df['Info/Query Cosim'] = all_sims

df.to_csv('info_narr_queryv1_df_cosim.csv', index=False)

print(f"the average info/query cosine similarity for High: {statistics.mean(highs)}, Medium: {statistics.mean(meds)}, Low: {statistics.mean(lows)}")