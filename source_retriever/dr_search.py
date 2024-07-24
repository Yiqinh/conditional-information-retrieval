from dense_retriever import MyDenseRetriever
import json
import os
import logging
here = os.path.dirname(os.path.abspath(__file__))
    
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

if __name__ == "__main__":

    dr = MyDenseRetriever.load("all_baseline-sources")

    f = os.path.join(os.path.dirname(here), 'baseline_queries', 'test_set', 'test_articles.json')
    with open(here, 'r') as file:
        articles = json.load(file)
        for article in articles:
            my_query = article['query']
            break
             

    res = dr.search(
        query = my_query
    )
    print(type(res))
    print(res)