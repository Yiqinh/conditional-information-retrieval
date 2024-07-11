from dense_retriever import MyDenseRetriever
import json


dr = MyDenseRetriever.load("new-index")

queries = [
    {'url': "www.google.com",
     'text': "This is a search engine that is the most popular."},
    {'url': "www.bing.com",
     'text': "This is another search engine that is not so popular."}
]

collections = {}
for summary in queries:
    collections[summary['url']] = dr.search(query = summary['text'])

print(collections)

fname = "scoreOutput.json"

with open(fname, 'w') as json_file:
        json.dump(collections, json_file)