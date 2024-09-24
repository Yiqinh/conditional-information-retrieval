import os
import json
import csv
import statistics
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

here = os.path.dirname(os.path.abspath(__file__))

model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv('info_narr_queryv1_df.csv.gz', compression='gzip')

print(len(df))
text_to_centrality = {}

for i in tqdm(range(len(df)), desc="Num Sources"):
    source_text = df.iloc[i]['Information']
    centrality = df.iloc[i]['Centrality']  
    text_to_centrality[source_text] = centrality

f = os.path.join(os.path.dirname(here), 'source_retriever', 'v2_search_res', 'v2_search_test_prompt1_all.json')

with open(f, 'r') as file:
    articles = json.load(file)

cosims = defaultdict(list)

data = [
    ['url', 'centrality', 'y_true', 'matched_pred', 'cosim']
]

for article in tqdm(articles, desc='Article Cosims Calculated'):
    y_pred = []
    y_true = []

    for source in article['dr_sources']:
        text = source['text']
        y_pred.append(text)

    for source in article['sources']:
        y_true.append(source['Information'])
    
    if len(y_true) == 0:
        continue

    y_true_embedding = model.encode(y_true)
    y_pred_embedding = model.encode(y_pred)

    similarities = model.similarity(y_true_embedding, y_pred_embedding) #calculates the cosine similarity by default

    cost = similarities.numpy()
    row_ind, col_ind = linear_sum_assignment(cost, maximize=True) #hungarian matching

    for r in row_ind: # r is the indices of the y_true sources
        c = col_ind[r] # each row/r is matched to a column, which represents a matched y_pred

        if text_to_centrality.get(y_true[r], None) != None:
            cur_centrality = text_to_centrality[y_true[r]]
            cosims[cur_centrality].append(float(similarities[r][c]))

            data_point = []
            data_point.append(article['url'])
            data_point.append(cur_centrality)
            data_point.append(y_true[r])
            data_point.append(y_pred[c])
            data_point.append(float(similarities[r][c]))

            data.append(data_point)

    break

for c, sims in cosims.items():
    u = statistics.mean(sims)
    print(f"The average cosim for {c} is: {u}")

with open('centrality_ypred_cosim.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write each row to the file
    for row in data:
        writer.writerow(row)






        
