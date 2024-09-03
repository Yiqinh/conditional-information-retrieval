import re
import ast
import json
import os

def robust_json(x):
    try:
        return json.loads(x)
    except:
        return None

def robust_ast(x):
    try:
        return ast.literal_eval(x)
    except:
        return None

def split_curly_braces(input_string):
    pattern = r'\{([^{}]*)\}'
    matches = re.findall(pattern, input_string)
    return matches

def robust_parser(f_path: str, seen_urls: set):
    res = []
    file = open(f_path)
    for line in file:

        article = json.loads(line)
        if article['url'] not in seen_urls:
            seen_urls.add(article['url'])
            sources = split_curly_braces(article['response'].split("\n\n")[-1])

            parsed_sources = []
            for source in sources:
                source = "{" + source + "}"
                
                temp = robust_json(source)
                if not temp:
                    temp = robust_ast(source)
                if not temp:
                    #print("skipped source from article", what)
                    continue
                
                source = temp
                index = 0
                one_parsed_source = {}
                for key, value in source.items():
                    if index == 0:
                        one_parsed_source['Name'] = value
                    if index == 1:
                        one_parsed_source['Original name'] = value
                    if index == 2:
                        one_parsed_source['Narrative function'] = value
                    if index == 3:
                        one_parsed_source['Perspective'] = value
                    if index == 4:
                        one_parsed_source['Centrality'] = value
                    if index == 5:
                        one_parsed_source['Justification'] = value
                    index += 1
                
                parsed_sources.append(one_parsed_source)

            if len(parsed_sources) > 0:
                parsed_article = {}
                parsed_article['url'] = article['url']
                parsed_article['sources'] = parsed_sources
                res.append(parsed_article)

    #check for none type
    for article in res:
        for source in article['sources']:
            for key, value in source.items():
                if type(value) != str:
                    article['sources'].remove(source)

    return res

if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))

    all_articles = []
    article_set = set()

    directory = os.path.join(here, "v2_narr_raw")
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            res = robust_parser(file_path, article_set)
            all_articles.extend(res)

    print("total articles: ", len(all_articles))

    test = json.load(os.path.join(here, "v2_info_parsed", "v2_test_set.json"))
    train = json.load(os.path.join(here, "v2_info_parsed", "v2_train_set.json"))

    test_urls = set()
    train_urls = set()

    for article in test:
        test_urls.add(article['url'])

    for article in train:
        train_urls.add(article['url'])

    test_articles = []
    train_articles = []

    counter = 0
    for article in all_articles:
        if article['url'] in test_urls:
            test_articles.append(article)
        elif article['url'] in train_urls:
            train_articles.append(article)
        else:
            counter += 1
    
    print("total articles skipped: ", counter)

    print("the length of train set is", len(train_articles))
    with open(os.path.join(here, 'v2_narr_parsed', 'v2_train_set_narr.json'), 'w') as f:
        json.dump(train_articles, f, indent=4)
    
    print("the length of test set is", len(test_articles))
    with open(os.path.join(here, 'v2_narr_parsed', 'v2_test_set_narr.json'), 'w') as f:
        json.dump(test_articles, f, indent=4)
    
