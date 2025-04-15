import os
import sys
import torch
import tiktoken
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.model import GPT

# === Load Config ===
class Config:
    block_size = 512
    vocab_size = 50257
    n_layer = 6
    n_head = 6
    n_embd = 384
    dropout = 0.1
    bias = False

config = Config()

# === Load Tokenizer ===
enc = tiktoken.get_encoding("gpt2")

# === Load Trained Model ===
model = GPT(config)
ckpt_path = "checkpoints/codewriter/ckpt.pt"
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
model.eval()

# === Setup Device ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# === Generate Function ===
@torch.no_grad()
def generate(prompt, max_new_tokens=100):
    input_ids = enc.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        input_ids_crop = input_ids[:, -config.block_size:]
        logits, _ = model(input_ids_crop)
        logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=1)

    out = enc.decode(input_ids[0].tolist())
    return out

# === Terminal Chat Loop ===
if __name__ == "__main__":
    print("💬 CodeWriterGPT is ready. Type 'exit' to quit.\n")
    while True:
        user_input = input("🧠 Instruction: ").strip()
        if user_input.lower() in ["exit", "quit"]: break

        prompt = f"### Instruction:\n{user_input}\n### Response:"
        output = generate(prompt)
        print("\n📦 Generated Code:\n")
        print(output.split("### Response:")[-1].strip())
        print("\n" + "="*60 + "\n")
