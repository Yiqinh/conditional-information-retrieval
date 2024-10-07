import re
import ast
import json
import os
import logging

from sklearn.model_selection import train_test_split

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
    
def robust_extract_json_str(lm_string):
    if not lm_string:
        return None
    # Use regular expressions to search for list brackets across multiple lines
    match = re.search(r'\[.*?\]', lm_string, re.DOTALL)
    if match:
        lm_string = match.group(0)
    try:
        return json.loads(lm_string)
    except:
        try:
            return ast.literal_eval(lm_string)
        except:
            pass
    logging.error(f"Could not extract json string from: {lm_string}")
    return None

def split_curly_braces(input_string):
    pattern = r'\{([^{}]*)\}'
    matches = re.findall(pattern, input_string)
    return matches

def robust_parser(f_path: str, seen_urls: set):
    res = []
    file = open(f_path)
    #what = 0
    for line in file:
        #what += 1
        article = json.loads(line)
        if article['url'] not in seen_urls:
            seen_urls.add(article['url'])
            sources = split_curly_braces(article['response'])

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
                        one_parsed_source['Original Name'] = value
                    if index == 2:
                        one_parsed_source['Information'] = value
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
            if (type(source.get('Information', None)) != str) or (type(source.get('Name', None)) != str):
                article['sources'].remove(source)

    return res

here = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    all_articles = []
    article_set = set()

    directory = os.path.join(here, "v2_info_raw")
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            res = robust_parser(file_path, article_set)
            all_articles.extend(res)
                    

    split = train_test_split(all_articles, shuffle=False, test_size=0.2)

    train_articles = split[0]
    test_articles = split[1]

    print("the length of train set is", len(train_articles))
    with open(os.path.join(here, 'v2_info_parsed', 'v2_train_set.json'), 'w') as f:
        json.dump(train_articles, f, indent=4)
    
    print("the length of test set is", len(test_articles))
    with open(os.path.join(here, 'v2_info_parsed', 'v2_test_set.json'), 'w') as f:
        json.dump(test_articles, f, indent=4)