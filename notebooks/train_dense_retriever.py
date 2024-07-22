# import logging
import json

# logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
# logging.getLogger("haystack").setLevel(logging.INFO)

# from haystack.nodes import DensePassageRetriever
# from haystack.utils import fetch_archive_from_http
# from haystack.document_stores import InMemoryDocumentStore

f = open('sample_article_sum.json')

data = json.load(f)

for i in range(10,20):
    print(data[i])