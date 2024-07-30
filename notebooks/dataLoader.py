import random
import json
import os
here = os.path.dirname(os.path.abspath(__file__))

articlesJsonTraining = open(os.path.join(os.path.dirname(here), "baseline_queries", "training_set", "training_articles.json"))
#sourcesJsonTraining = open('name of training sources file with directory.json')
articlesDictTraining = json.load(articlesJsonTraining)
#sourcesDictTraining = json.load(sourcesJsonTraining)
sourcesDictTraining = {}
for url, article in articlesDictTraining.items():
    for id, source_text in article["sources"]:
        sourcesDictTraining[id] = source_text

articlesJsonTesting = open(os.path.join(os.path.dirname(here), "baseline_queries", "test_set", "test_articles.json"))
#sourcesJsonTesting = open('name of Testing sources file with directory.json')
articlesDictTesting = json.load(articlesJsonTesting)
#sourcesDictTesting = json.load(sourcesJsonTesting)
sourcesDictTesting = {}
for url, article in articlesDictTesting.items():
    for id, source_text in article["sources"]:
        sourcesDictTesting[id] = source_text

def dataLoader(articles, sources):
    hayStackData = []
    listSources = list(sources.items)
    for key in articles:
        datapoint = {}
        datapoint["dataset"] = str(key)
        datapoint["question"] = articles[key]["query"]
        datapoint["dataset"] = str(key)
        datapoint["positive_ctxs"] = []
        for source in articles[key]["sources"]:
            currentCtx = {}
            currentCtx["title"] = str(source)
            currentCtx["text"] = articles[key]["sources"][source]
            currentCtx["score"] = 1
            currentCtx["title_score"] = 1
            currentCtx["passage_id"] = str(source)
            datapoint["positive_ctxs"].append(currentCtx)
        datapoint["negative_ctxs"] = []
        datapoint["hard_negative_ctxs"] = []
        for i in range(10):
            source, summary = random.choice(listSources)
            currentCtx = {}
            currentCtx["title"] = str(source)
            currentCtx["text"] = summary
            currentCtx["score"] = 1
            currentCtx["title_score"] = 1
            currentCtx["passage_id"] = str(source)
            datapoint["hard_negative_ctxs"].append(currentCtx)
        hayStackData.append(datapoint)
    return hayStackData

trainingData = dataLoader(articlesDictTraining, sourcesDictTraining)
testingData = dataLoader(articlesDictTesting, sourcesDictTesting)

with open('data/trainingData.json', 'w') as fp:
    json.dump(trainingData, fp)

with open('data/testingData.json', 'w') as fp:
    json.dump(testingData, fp)
