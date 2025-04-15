# scripts/train_model.py

import sys
import os
import time
import math
import torch
import numpy as np
import importlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.model import GPT
from torch.nn import functional as F

# --- Load Config from CLI argument ---
if len(sys.argv) < 2:
    print("Usage: python scripts/train_model.py config/config_codewriter.py")
    sys.exit(1)

config_path = sys.argv[1]
config_module = config_path.replace("/", ".").replace(".py", "")
config = importlib.import_module(config_module)

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Tokenized Data ---
def load_dataset(split):
    with open(os.path.join("data", config.dataset, f"{split}.bin"), "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint16)
    return torch.tensor(data, dtype=torch.long)

train_data = load_dataset("train")
val_data = load_dataset("val")

# --- Create Batches ---
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+config.block_size] for i in ix])
    return x.to(device), y.to(device)

# --- Initialize Model ---
model = GPT(config)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, config.beta2))

# --- Learning Rate Scheduler ---
def get_lr(it):
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

# --- Evaluate Loss ---
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- Training Loop ---
best_val_loss = float('inf')
for iter in range(config.max_iters):
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % config.log_interval == 0:
        print(f"Step {iter}: loss = {loss.item():.4f}")

    if iter % config.eval_interval == 0:
        losses = estimate_loss()
        print(f"Eval at step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            os.makedirs(config.out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config.out_dir, 'ckpt.pt'))
            print("✅ Saved new best model.")

print("🎉 Training complete.")
