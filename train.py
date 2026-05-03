"""
Fast GAN Training for Language Models (single‑token generation)
Same evaluation as original supervised training.
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# GAN-specific hyperparameters
disc_learning_rate = 1e-4
gen_learning_rate = 1e-4
disc_train_ratio = 1               # discriminator updates per generator update
reward_smoothing = 0.9
max_grad_norm_disc = 1.0
max_grad_norm_gen = 1.0

# -----------------------------------------------------------------------------
# default config values (original)
out_dir = 'out_gan_fast'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
wandb_log = False
wandb_project = 'owt_gan_fast'
wandb_run_name = 'gpt2_gan_fast'
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
learning_rate = 6e-4          # not used for generator, but here for compatibility
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
max_iters = 600000

# -----------------------------------------------------------------------------
# parse config overrides
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# DDP init (MUST be before any use of ddp variable)
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# data loader (same as original)
data_dir = os.path.join('data', dataset)
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# meta info
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# -----------------------------------------------------------------------------
# generator model (original GPT)
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
if init_from == 'scratch':
    print("Initializing a new generator from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    generator = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt_gan_fast.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    generator = GPT(gptconf)
    generator.load_state_dict(checkpoint['generator'])
    # also restore iter_num, reward_baseline if stored
else:
    raise ValueError(f"init_from {init_from} not supported in GAN mode (only 'scratch' or 'resume')")

if block_size < generator.config.block_size:
    generator.crop_block_size(block_size)
    model_args['block_size'] = block_size
generator.to(device)

# -----------------------------------------------------------------------------
# Simple Discriminator (self-contained Transformer classifier)
class SimpleDiscriminator(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=384, n_head=6, n_layer=4, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(n_embd)
        self.classifier = nn.Linear(n_embd, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size
        positions = torch.arange(0, T, device=idx.device).unsqueeze(0).expand(B, T)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(tok_emb + pos_emb)
        x = self.transformer(x)  # (B, T, n_embd)
        x = self.ln_f(x)
        last_hidden = x[:, -1, :]
        logit = self.classifier(last_hidden).squeeze(-1)
        return logit

discriminator = SimpleDiscriminator(model_args['vocab_size'], block_size, n_embd=384, n_head=6, n_layer=4, dropout=0.1)
discriminator.to(device)

# -----------------------------------------------------------------------------
# Optimizer helper
def create_optimizer(model, weight_decay, learning_rate, betas, device_type):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    fused_available = 'fused' in torch.optim.AdamW.__dict__.get('__annotations__', {})
    use_fused = fused_available and device_type == 'cuda'
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
    return optimizer

optimizer_gen = generator.configure_optimizers(weight_decay, gen_learning_rate, (beta1, beta2), device_type)
optimizer_disc = create_optimizer(discriminator, weight_decay, disc_learning_rate, (beta1, beta2), device_type)

scaler_gen = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
scaler_disc = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

if ddp:
    generator = DDP(generator, device_ids=[ddp_local_rank])
    discriminator = DDP(discriminator, device_ids=[ddp_local_rank])

# -----------------------------------------------------------------------------
# Fast GAN helpers (single token generation)
def sample_next_token(generator, prefix_tokens, ctx):
    with ctx:
        logits, _ = generator(prefix_tokens)
    next_logits = logits[:, -1, :]
    probs = F.softmax(next_logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    next_token = dist.sample()
    logprob = dist.log_prob(next_token)
    return next_token, logprob

def reinforce_update_single_token(generator, discriminator, optimizer_gen, reward_baseline,
                                   real_batch_fn, device, ctx, scaler):
    generator.train()
    discriminator.eval()
    real_x, _ = real_batch_fn()
    prefix = real_x[:, :-1]
    sampled_token, logprob = sample_next_token(generator, prefix, ctx)
    completed = torch.cat([prefix, sampled_token.unsqueeze(1)], dim=1)
    with torch.no_grad():
        disc_logit = discriminator(completed)
        reward = torch.sigmoid(disc_logit)
    baseline = reward_baseline[0]
    advantage = reward - baseline
    reward_baseline[0] = baseline * reward_smoothing + reward.mean().item() * (1 - reward_smoothing)
    loss_gen = -(logprob * advantage).mean()
    optimizer_gen.zero_grad(set_to_none=True)
    scaler.scale(loss_gen).backward()
    if max_grad_norm_gen != 0.0:
        scaler.unscale_(optimizer_gen)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm_gen)
    scaler.step(optimizer_gen)
    scaler.update()
    return reward.mean().item(), loss_gen.item()

def train_discriminator_single_token(generator, discriminator, optimizer_disc, real_batch_fn,
                                      device, batch_size, block_size, ctx, scaler):
    discriminator.train()
    generator.eval()
    real_x, _ = real_batch_fn()
    prefix = real_x[:, :-1]
    with torch.no_grad():
        sampled_token, _ = sample_next_token(generator, prefix, ctx)
        fake_x = torch.cat([prefix, sampled_token.unsqueeze(1)], dim=1)
    real_labels = torch.ones(batch_size, device=device)
    fake_labels = torch.zeros(batch_size, device=device)
    real_logits = discriminator(real_x)
    fake_logits = discriminator(fake_x)
    loss_real = F.binary_cross_entropy_with_logits(real_logits, real_labels)
    loss_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
    loss_disc = (loss_real + loss_fake) / 2
    optimizer_disc.zero_grad(set_to_none=True)
    scaler.scale(loss_disc).backward()
    if max_grad_norm_disc != 0.0:
        scaler.unscale_(optimizer_disc)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm_disc)
    scaler.step(optimizer_disc)
    scaler.update()
    return loss_disc.item()

# -----------------------------------------------------------------------------
# Original evaluation function (same as supervised)
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# Training loop (fast single‑token GAN)
iter_num = 0
reward_baseline = [0.5]
raw_gen = generator.module if ddp else generator
raw_disc = discriminator.module if ddp else discriminator

print("Starting fast GAN training (single‑token generation)...")
while True:
    # Discriminator updates
    for _ in range(disc_train_ratio):
        disc_loss = train_discriminator_single_token(
            raw_gen, raw_disc, optimizer_disc, lambda: get_batch('train'),
            device, batch_size, block_size, ctx, scaler_disc
        )
    # Generator REINFORCE update (one token per sequence)
    avg_reward, gen_loss = reinforce_update_single_token(
        raw_gen, raw_disc, optimizer_gen, reward_baseline,
        lambda: get_batch('train'), device, ctx, scaler_gen
    )
    # Log supervised training loss for comparison
    with torch.no_grad():
        real_x, real_y = get_batch('train')
        _, sup_loss = raw_gen(real_x, real_y)
        sup_loss_val = sup_loss.item()
    if iter_num % log_interval == 0 and master_process:
        print(f"iter {iter_num:6d}: sup_train_loss {sup_loss_val:.4f}, "
              f"disc_loss={disc_loss:.4f}, avg_reward={avg_reward:.4f}, gen_loss={gen_loss:.4f}")
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss(raw_gen)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        checkpoint = {
            'generator': raw_gen.state_dict(),
            'discriminator': raw_disc.state_dict(),
            'optimizer_gen': optimizer_gen.state_dict(),
            'optimizer_disc': optimizer_disc.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'reward_baseline': reward_baseline[0],
        }
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt_gan_fast.pt'))
        if always_save_checkpoint:
            print(f"checkpoint saved to {out_dir}/ckpt_gan_fast.pt")
    iter_num += 1
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()