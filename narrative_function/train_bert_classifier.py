import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

# Tokenizer and tokenization function
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
def tokenize_function(examples):
    return tokenizer(examples['query'], padding="max_length", truncation=True, max_length=512)

# Prepare multilabel classification labels
def format_labels(example):
    example['labels'] = [example[f] for f in cols_to_model]
    return example

# Metrics function
def compute_metrics(p):
    preds = p.predictions
    preds = (preds > 0.5).astype(int)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='micro')
    precision = precision_score(labels, preds, average='micro')
    recall = recall_score(labels, preds, average='micro')

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

cols_to_model = [
    'Central Figure', 'Contextual Narratives', 'Credible Narratives', 
    'Critical Analysis', 'Enriched Narratives', 'User Interaction'
]

training_data = pd.read_csv('bert_training_data.csv')
# Split data into train and validation sets
train_df, test_df = training_data.loc[lambda df: df['split'] == 'train'], training_data.loc[lambda df: df['split'] == 'test']

from transformers import AutoConfig
train_dataset = Dataset.from_pandas(train_df)
train_tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
train_tokenized_dataset = train_tokenized_dataset.map(format_labels)
train_tokenized_dataset = train_tokenized_dataset.remove_columns(['query'] + cols_to_model)
train_tokenized_dataset.set_format("torch")

test_dataset = Dataset.from_pandas(test_df)
test_tokenized_dataset = test_dataset.map(tokenize_function, batched=True)
test_tokenized_dataset = test_tokenized_dataset.map(format_labels)
test_tokenized_dataset = test_tokenized_dataset.remove_columns(['query'] + cols_to_model)
test_tokenized_dataset.set_format("torch")


config = AutoConfig.from_pretrained('distilbert-base-uncased')
config.problem_type = "multi_label_classification"
config.num_labels = len(cols_to_model)

# Load the model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', 
    config=config
)

# Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=test_tokenized_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train and evaluate
trainer.train()
trainer.evaluate()