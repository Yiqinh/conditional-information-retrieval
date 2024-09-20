import argparse
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator


def main(args):
    # 1. Load a model to finetune with
    model = SentenceTransformer(args.model_name)

    # 3. Load a dataset to finetune on
    dataset = load_dataset("json", data_files=args.data_file)
    dataset = dataset['train'].train_test_split(test_size=args.test_size)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # 4. Define a loss function
    loss = MultipleNegativesRankingLoss(model)

    # 5. Specify training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        run_name=args.run_name,
    )

    # 6. Create an evaluator & evaluate the base model
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["anchor"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="all-nli-dev",
    )
    if args.do_initial_evaluation:
        dev_evaluator(model)

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # 8. Save the trained model
    model.save_pretrained(f"{args.output_dir}/trained-model")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune a SentenceTransformer model.')

    # Model and dataset parameters
    parser.add_argument('--model_name', type=str, default='microsoft/mpnet-base',
                        help='Name of the pre-trained model to fine-tune.')
    parser.add_argument('--data_file', type=str, default='../data/similarity_training_data/source-triplets.jsonl',
                        help='Path to the dataset file.')
    parser.add_argument('--output_dir', type=str, default='models/mpnet-base-all-nli-triplet',
                        help='Output directory for the trained model.')

    # Training parameters
    parser.add_argument('--num_train_epochs', type=int, default=1,
                        help='Number of training epochs.')
    parser.add_argument('--train_batch_size', type=int, default=16,
                        help='Training batch size per device.')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help='Evaluation batch size per device.')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate.')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio.')
    parser.add_argument('--fp16', action='store_true',
                        help='Use fp16 training.')
    parser.add_argument('--bf16', action='store_true',
                        help='Use bf16 training.')

    # Dataset parameters
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--train_subset_size', type=int, default=100000,
                        help='Number of training samples to select.')

    # Evaluation and logging parameters
    parser.add_argument('--eval_strategy', type=str, default='steps',
                        help='Evaluation strategy.')
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='Evaluation steps.')
    parser.add_argument('--save_strategy', type=str, default='steps',
                        help='Model save strategy.')
    parser.add_argument('--save_steps', type=int, default=100,
                        help='Save steps.')
    parser.add_argument('--save_total_limit', type=int, default=2,
                        help='Total number of model checkpoints to keep.')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Logging steps.')
    parser.add_argument('--run_name', type=str, default='mpnet-base-all-nli-triplet',
                        help='Run name for logging.')

    # Other parameters
    parser.add_argument('--do_initial_evaluation', action='store_true',
                        help='Evaluate the base model before training.')

    args = parser.parse_args()    
    main(args)
