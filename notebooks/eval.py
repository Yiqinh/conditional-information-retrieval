import json
import pdb


def custom_accuracy(y_true, y_pred):
    correct = 0
    total = len(y_true)
    
    for i in range(total):
        if y_true[i] in y_pred[i]:
            correct += 1
    
    return correct / len(y_pred)

test_filename = "/project/jonmay_231/spangher/Projects/conditional-information-retrieval/source_summaries/v2_info_parsed/combined_test_prompt1_v2.json"
retrieval_results_filename = "/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/test_result.json"
with open(retrieval_results_filename, 'w') as json_file:
    retrieval_results = json.load(json_file)
with open(test_filename, 'w') as json_file:
    test_set = json.load(json_file)

id_to_label_index = {}
included_documents = [] #a list of document ids that need to be included
label_index = 0
with open(test_filename, 'r') as file:
    articles = json.load(file)
    for url, article in articles.items():
        for name, text in article["sources"].items():
            included_documents.append(text)
            id_to_label_index[text] = label_index
            label_index += 1


pdb.set_trace()

for query in retrieval_results:

    y_true = [source['Information'] for source in test_set[query]]
    y_pred = [source.split("###") for source in retrieval_results[query]]
    custom_accuracy(y_true, y_pred)

