import json
import pdb
import statistics
from tqdm import tqdm

def get_scores(path: str):
    precision_list = []
    recall_list = []
    f1_list = []

    with open(path, 'r') as file:
        articles = json.load(file)
        for article in tqdm(articles):
            y_pred = set()
            y_true = set()

            for source in article['dr_sources']:
                id = source['id']
                y_pred.add(id)
            for source in article['sources']:
                y_true.add(article['url']+"#"+source['Name'])
            
            if len(y_true) == 0:
                continue
            
            true_pos = set.intersection(y_pred, y_true)
            n = len(true_pos)

            recall = n / len(y_true)
            precision = n / len(y_pred)

            if (recall + precision) == 0:
                f1 = 0

            else:
                f1 = (2 * precision * recall) / (precision + recall)
            
            recall_list.append(recall)
            precision_list.append(precision)
            f1_list.append(f1)
    
    avg_prec = statistics.mean(precision_list)
    avg_rec = statistics.mean(recall_list)
    avg_f1 = statistics.mean(f1_list)

    print("average precision:", avg_prec)
    print("average recall:", avg_rec)
    print("average f1:", avg_f1)



get_scores("/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/test_result.json")