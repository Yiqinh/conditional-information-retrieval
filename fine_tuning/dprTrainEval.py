import logging
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore
import os
import torch
import json
from tqdm import tqdm
import torch.distributed as dist

# Set environment variables
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
os.environ['NCCL_BUFFSIZE'] = '8388608'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

# Initialize the distributed backend
dist.init_process_group(backend='nccl', init_method='env://')


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

doc_dir = "data"
train_filename = '/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/data/train.json'
dev_filename = '/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/data/test.json'

query_model = "facebook/dpr-question_encoder-single-nq-base"
passage_model = "facebook/dpr-ctx_encoder-single-nq-base"

save_dir = "/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/saved_models"

retriever = DensePassageRetriever(
    document_store=InMemoryDocumentStore(),
    query_embedding_model=query_model,
    passage_embedding_model=passage_model,
    max_seq_len_query=64,
    max_seq_len_passage=256,
)

retriever.train(
    data_dir=doc_dir,
    train_filename=train_filename,
    dev_filename=dev_filename,
    test_filename=dev_filename,
    n_epochs=10,
    batch_size=16,
    grad_acc_steps=8,
    save_dir=save_dir,
    evaluate_every=3000,
    embed_title=True,
    num_positives=1,
    num_hard_negatives=15,
)

reloaded_retriever = DensePassageRetriever.load(load_dir=save_dir, document_store=None)
print("finished loading the retriever")

print(dev_filename)
with open(dev_filename, 'r') as f:
    articles = json.load(f)


results = {}
for article in tqdm(articles):
    question = article['question']
    if question == "":
        print("This question is empty")
        continue
    results[question] = reloaded_retriever.retrieve(question, top_k=10)

print(results)

with open(f"/project/jonmay_231/spangher/Projects/conditional-information-retrieval/fine_tuning/data/test_result.json", 'w') as json_file:
    json.dump(results, json_file)

