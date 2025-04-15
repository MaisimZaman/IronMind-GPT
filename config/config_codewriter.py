out_dir = 'checkpoints/codewriter'
eval_interval = 200
eval_iters = 40
log_interval = 10

always_save_checkpoint = True
wandb_log = False
wandb_project = 'codewriter'
wandb_run_name = 'gpt2-codewriter'

dataset = 'codewriter'
gradient_accumulation_steps = 1
batch_size = 8
block_size = 256  # Code needs long context

n_layer = 4
n_head = 4
n_embd = 256

dropout = 0.1
bias = False  # Use LayerNorm without bias for simplicity

learning_rate = 3e-4
max_iters = 5000
lr_decay_iters = 3000
min_lr = 1e-5
beta2 = 0.95
warmup_iters = 100
vocab_size = 50257

decay_lr = True

device = 'cuda'  # Change to 'cpu' if needed
compile = True
