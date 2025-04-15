# train/trainer.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import time
import numpy as np
from model.transformer import IronMindGPT, Config



# Configs
config = Config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data
def load_dataset(split):
    path = f"data/processed/{split}.bin"
    data = np.memmap(path, dtype=np.uint16, mode='r')
    return torch.from_numpy(np.array(data, dtype=np.int64))

train_data = load_dataset("train")
val_data = load_dataset("val")

# Batch generator
def get_batch(split, block_size, batch_size):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Model
model = IronMindGPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training loop
max_iters = 5000
eval_interval = 500
eval_iters = 200
batch_size = 32
block_size = config.block_size

def estimate_loss():
    model.eval()
    losses = {'train': 0.0, 'val': 0.0}
    with torch.no_grad():
        for split in ['train', 'val']:
            split_loss = 0
            for _ in range(eval_iters):
                X, Y = get_batch(split, block_size, batch_size)
                logits = model(X)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
                split_loss += loss.item()
            losses[split] = split_loss / eval_iters
    model.train()
    return losses

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    X, Y = get_batch('train', block_size, batch_size)
    logits = model(X)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Save model
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/ironmind_gpt.pt")
print("Training complete. Model saved.")
