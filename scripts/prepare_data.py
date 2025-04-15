from datasets import load_dataset
from transformers import AutoTokenizer
import os
import pickle
import tiktoken  # If you're using OpenAI tokenization

# Load CodeAlpaca
dataset = load_dataset("sahil2801/CodeAlpaca-20k")['train']

# Create instruction + response merged string
texts = []
for example in dataset:
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}\n\n"
    texts.append(prompt)

full_text = "\n".join(texts)

# Tokenize (simple method with tiktoken)
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode_ordinary(full_text)

# Save to binary format
os.makedirs("data/codewriter", exist_ok=True)
train_ids = ids[:int(0.9 * len(ids))]
val_ids = ids[int(0.9 * len(ids)):]

with open("data/codewriter/train.bin", "wb") as f:
    f.write(bytearray(train_ids))

with open("data/codewriter/val.bin", "wb") as f:
    f.write(bytearray(val_ids))

print("✅ CodeWriterGPT data prepared and saved.")
