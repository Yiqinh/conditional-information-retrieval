import os
import json
import re
import logging
here = os.path.dirname(os.path.abspath(__file__))

def strip_quotes(s):
    # Pattern to match any kind of quote at the start and end
    pattern = r'^(?P<quote>[\'"])(.*)(?P=quote)$'
    
    match = re.match(pattern, s)
    if match:
        # If quotes are found, strip them and recurse
        return strip_quotes(match.group(2))
    else:
        # If no quotes are found, return the string as is
        return s
    
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

if __name__ == "__main__":
    #parse to get rid of the LLM response output header labels
    url_to_query = {}

    for i in range(0, 300000, 30000):
        f = os.path.join('llm_output', f"article_sum__{i}_{i+30000}.json")
        with open(f, 'r') as file:
            articles = json.load(file)
            for article in articles:
                temp = article["query"].split('\n\n') #parse to get just the preliminary question
                url_to_query[article["url"]] = strip_quotes(temp[-1])

    #join article url, query, and ground truth sources
    res = {}

    for i in range(0, 272800, 100):
        file = f"sources_data_70b__{i}_{i+100}.json"
        file_path = os.path.join(os.path.dirname(here), 'source_summaries', 'json_summaries', file)
        if (os.path.exists(file_path)):
            with open(file_path, 'r') as file:
                articles = json.load(file)
                for article in articles:
                    curr = {}
                    url_plus_source_name = {article['article_url'] + key: value for key, value in article['sources'].items()}
                    curr["sources"] = url_plus_source_name
                    curr["query"] = url_to_query[article["article_url"]]
                    res[article['article_url']] = curr

    # split into test set and training set
    total = len(res)
    twenty_per = int(total / 5) + (total % 5 > 0)

    logging.info(f"Test Set / Total Split = {twenty_per} / {total}")

    test_set = {}
    training_set = {}

    i = 0
    for url, dict in res.items():
        if i <= twenty_per:
            test_set[url] = dict
        else:
            training_set[url] = dict
        i += 1

    logging.info(f"Num Articles in Test Set: {len(test_set)}")
    logging.info(f"Num Articles in Training Set: {len(training_set)}")
    logging.info(f"Total Articles: {total}")

    with open(os.path.join('test_set', "test_articles.json"), 'w') as file:
        json.dump(test_set, file)

    with open(os.path.join('training_set', "training_articles.json"), 'w') as file:
        json.dump(training_set, file)
        

