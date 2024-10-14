import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset
import os

class BertPoissonRegression(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", hidden_dim=128):
        super(BertPoissonRegression, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token embedding
        x = self.dropout(pooled_output)
        x = torch.relu(self.fc1(x))
        return torch.exp(self.fc2(x))  # Exponential for Poisson rate
    

def process_text_data(texts, y, tokenizer, seq_length=512):
    """Generates dummy text data and Poisson targets."""
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=seq_length, return_tensors="pt")

    # Random weights for generating Poisson targets
    # true_weights = torch.randn(encodings["input_ids"].size(1), 1)
    # rates = torch.exp(encodings["input_ids"].float() @ true_weights)  # Rate parameter λ
    # y = torch.poisson(rates)  # Poisson-distributed targets
    y = torch.tensor(y)
    return encodings, y


def train_model(model, train_loader, optimizer, loss_fn, accelerator, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        with accelerator.main_process_first():
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            input_ids, attention_mask, y_batch = batch
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(input_ids, attention_mask)
            
            # Compute loss
            loss = loss_fn(y_pred, y_batch)
            accelerator.backward(loss)
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            avg_loss = total_loss / len(train_loader.dataset)

            # Update progress bar
            progress_bar.set_postfix(loss=avg_loss)

        accelerator.print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")


def evaluate_model(model, data_loader, metric="mse"):
    """Evaluates the model using MSE or Poisson Deviance."""
    model.eval()
    total_metric = 0.0

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, y_batch = batch
            y_pred = model(input_ids, attention_mask)

            if metric == "mse":
                metric_value = torch.mean((y_pred - y_batch) ** 2).item()
            elif metric == "poisson_deviance":
                metric_value = torch.mean(y_pred - y_batch * torch.log(y_pred + 1e-8)).item()

            total_metric += metric_value * input_ids.size(0)

    avg_metric = total_metric / len(data_loader.dataset)
    print(f"Evaluation - {metric.upper()}: {avg_metric:.4f}")
    return avg_metric


def score_prediction_data(model, data_loader):
    """Scores the prediction data and returns the predictions."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, _ = batch
            y_pred = model(input_ids, attention_mask)
            predictions.extend(y_pred.detach().cpu().numpy())

    return predictions


def save_model(model, optimizer, epoch, accelerator, save_dir="poisson_model_checkpoint"):
    """Saves the model and optimizer state."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Unwrap the model from the `accelerate` wrapper
    unwrapped_model = accelerator.unwrap_model(model)

    # Save the state dicts
    torch.save({
        "model_state_dict": unwrapped_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }, os.path.join(save_dir, "checkpoint.pth"))


def load_model(model, optimizer, accelerator, save_dir="poisson_model_checkpoint"):
    """Loads the model and optimizer state."""
    checkpoint = torch.load(os.path.join(save_dir, "checkpoint.pth"))

    # Unwrap the model to load the state dict correctly
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1  # Resume from the next epoch
    print(f"Resuming training from epoch {start_epoch}")
    return start_epoch


def load_model_without_accelerate(save_dir="poisson_model_checkpoint"):
    """Loads the model state without using accelerate."""
    checkpoint = torch.load(os.path.join(save_dir, "checkpoint.pth"))
    
    # Initialize a new model instance
    model = BertPoissonRegression()
    
    # Load the state dict into the model
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print("Model loaded successfully without using accelerate.")
    return model




if __name__ == "__main__":
    import pandas as pd 
    import json
    accelerator = Accelerator()

    train_df = pd.DataFrame(json.load(open('v3_combined_TRAIN.json')))
    train_df['label'] = train_df['truth'].str.len()
    test_df = pd.DataFrame(json.load(open('v3_combined_TEST.json')))
    test_df['label'] = test_df['truth'].str.len()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_encodings, train_y = process_text_data(train_df['query'].tolist(), train_df['label'].tolist(), tokenizer=tokenizer)
    test_encodings, test_y = process_text_data(test_df['query'].tolist(), test_df['label'].tolist(), tokenizer=tokenizer)

    # Hyperparameters
    batch_size = 8
    epochs = 3
    learning_rate = 2e-5

    # Initialize tokenizer and model
    model = BertPoissonRegression()

    # Create DataLoaders
    train_dataset = TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"], train_y)
    val_dataset = TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"], test_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Prepare model, optimizer, and DataLoaders with accelerate
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    # Ensure the model and data are on the correct device (e.g., CUDA)
    print(f"Using device: {accelerator.device}")

    # Train the model
    print("Training the model...")
    train_model(model, train_loader, optimizer, nn.PoissonNLLLoss(log_input=False), accelerator, epochs)

    # Evaluate the model
    print("\nEvaluating the model...")
    evaluate_model(model, val_loader, metric="mse")
    evaluate_model(model, val_loader, metric="poisson_deviance")

    # Save the model after training
    print("Saving the model...")
    save_model(model, optimizer, epochs - 1, accelerator)

    # score prediction data and dump to file
    # Score the prediction data

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, _ = batch
            y_pred = model(input_ids, attention_mask)
            predictions.extend(y_pred.detach())

    # Dump predictions to disk
    output_file = 'predictions.json'
    all_predictions = accelerator.gather(torch.tensor(predictions, device=accelerator.device))
    all_predictions = all_predictions.cpu().numpy().squeeze().tolist()
    
    # Only save predictions from the main process to avoid overwriting
    if accelerator.is_main_process:
        print(f"Saving predictions to {output_file}")
        predictions = list(map(float, all_predictions))
        with open(output_file, 'w') as f:
            json.dump(predictions, f)
    
    
"""
import torch
import json
import pandas as pd 
from poisson_regression_script import process_text_data, BertPoissonRegression, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
test_df = pd.DataFrame(json.load(open('v3_combined_TEST.json')))
test_df['label'] = test_df['truth'].str.len()
test_encodings, test_y = process_text_data(test_df['query'].tolist(), test_df['label'].tolist(), tokenizer=tokenizer)

model = load_model_without_accelerate()
model.eval()
with torch.no_grad():
    input_ids, attention_mask, y_batch = test_encodings["input_ids"], test_encodings["attention_mask"], test_y
    y_pred = model(input_ids, attention_mask)
    metric_value = torch.mean((y_pred.detach().cpu() - y_batch) ** 2).item()
    print(f"MSE: {metric_value:.4f}")
    metric_value = torch.mean(y_pred.detach().cpu() - y_batch * torch.log(y_pred + 1e-8)).item()
    print(f"Poisson Deviance: {metric_value:.4f}")
"""