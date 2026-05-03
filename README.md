# GAN nanoGPT
GAN-style training for nanoGPT. Only the last token is generated per sequence to keep speed comparable to the original.

## Usage

```
git clone https://github.com/karpathy/nanoGPT
cd nanoGPT
replace train.py
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

## Results

```
step 2000: train loss 3.6276, val loss 3.6424
```
