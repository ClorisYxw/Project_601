import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import importlib

# Load the configuration file
config_file = 'config.train_my_data'  # Modify this path to your configuration file
config = importlib.import_module(config_file)

# Set the device
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Custom dataset class
class BinaryDataset(Dataset):
    def __init__(self, data_file, block_size):
        with open(data_file, 'rb') as f:
            self.data = np.fromfile(f, dtype=np.uint16)  # Load binary file
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + 1 + self.block_size], dtype=torch.long)
        return x, y

# Load datasets
def load_data():
    print("Loading data...")
    train_dataset = BinaryDataset(os.path.join('data/my_data', 'train.bin'), config.block_size)
    test_dataset = BinaryDataset(os.path.join('data/my_data', 'test.bin'), config.block_size)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    print(f"Loaded train dataset with {len(train_dataset)} samples.")
    print(f"Loaded test dataset with {len(test_dataset)} samples.")
    return train_loader, test_loader

# Define the model
class GPTLanguageModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = torch.nn.Embedding(config.vocab_size, config.n_embd)
        self.transformer = torch.nn.Transformer(
            d_model=config.n_embd,
            nhead=config.n_head,
            num_encoder_layers=config.n_layer,
            num_decoder_layers=config.n_layer,
            dropout=config.dropout
        )
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, x, y=None):
        x_emb = self.embedding(x)
        logits = self.transformer(x_emb, x_emb)
        logits = self.lm_head(logits)
        if y is not None:
            loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), y.view(-1))
            return logits, loss
        return logits

# Training function
def train(model, train_loader, optimizer, scheduler, config):
    model.train()
    running_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % config.log_interval == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
    return running_loss / len(train_loader)

# Evaluation function
def evaluate(model, test_loader, config):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            running_loss += loss.item()
    return running_loss / len(test_loader)

# Main function
def main():
    # Load the data
    train_loader, test_loader = load_data()

    # Initialize the model
    model = GPTLanguageModel(config).to(device)

    # Define the optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    # Training loop
    for epoch in range(config.max_iters):
        print(f"Epoch {epoch + 1}/{config.max_iters}")
        train_loss = train(model, train_loader, optimizer, scheduler, config)
        val_loss = evaluate(model, test_loader, config)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the model
        if epoch % config.eval_interval == 0:
            checkpoint_path = os.path.join(config.out_dir, f"model_epoch{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()

