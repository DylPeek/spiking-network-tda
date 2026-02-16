#!/usr/bin/env python3
"""
Date: Feb 16, 2026
Description: Train and evaluate an SNN on MNIST with pixel-moving perturbations, saving spike activity for TE+PH analysis.
Inputs:
  - Command-line arguments controlling training, perturbation level, network size, and output paths.
  - MNIST dataset (downloaded via torchvision if not present).
Outputs:
  - Experiment folder containing trained weights, logs, and spike activity arrays for downstream analysis.
"""

import os, csv, random, numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import argparse
plt.rcParams["figure.dpi"] = 120


# argument parser to override hyper‑parameters
def parse_args():
    parser = argparse.ArgumentParser(description="MNIST pixel move noise training with LIF spiking neural network")
    parser.add_argument("--root", type=str, default="mnist_network_64", help="Output root directory")
    parser.add_argument("--t_steps", type=int, default=100, help="Number of discrete time steps per sample")
    parser.add_argument("--n_neurons", type=int, default=64, help="Number of LIF neurons in the recurrent layer")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--print_every", type=int, default=1, help="Save spikes and evaluate every N epochs")
    parser.add_argument("--pr_max", type=float, default=0.50, help="Spike probability for pixel value 1 (255)")
    parser.add_argument("--pr_min", type=float, default=0.02, help="Spike probability for pixel value 0")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--eval_samples", type=int, default=1024, help="Number of test samples to encode (<=10000)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on (cuda or cpu)")
    return parser.parse_args()

args = parse_args()

# assign configuration from arguments
ROOT         = args.root
T_STEPS      = args.t_steps
N_NEURONS    = args.n_neurons
BATCH_SIZE   = args.batch_size
EPOCHS       = args.epochs
PRINT_EVERY  = args.print_every
PR_MAX       = args.pr_max
PR_MIN       = args.pr_min
LR           = args.lr
EVAL_SAMPLES = args.eval_samples
DEVICE       = args.device

# complexity‑level → number‑of‑pixel‑moves; adjust values to change noise difficulty
NOISE_LEVELS = {0: 0, 1: 5, 2: 10, 3: 15, 4: 20,
                5: 25, 6: 30, 7: 35, 8: 40}

os.makedirs(f"{ROOT}/learning", exist_ok=True)
os.makedirs(f"{ROOT}/evaluation", exist_ok=True)

# --- 1. surrogate spike fn ---
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v): ctx.save_for_backward(v); return (v > 0.).float()
    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        return g * (1 / (1 + v.abs())**2)
spike = SpikeFn.apply

  
class RecLIF(nn.Module):
    def __init__(self, n_in, n_rec, beta=0.9):
        super().__init__()
        self.w_in  = nn.Parameter(torch.randn(n_in, n_rec) * 0.08)
        self.w_rec = nn.Parameter(torch.randn(n_rec, n_rec) * 0.04)
        self.beta  = beta
    def forward(self, x):                     # x [B,784,T]
        B,_,T = x.shape
        mem = x.new_zeros(B, self.w_rec.size(0)); outs=[]
        for t in range(T):
            cur = x[:,:,t]@self.w_in + (outs[-1]@self.w_rec if outs else 0.)
            mem = self.beta*mem + cur
            spk = spike(mem-1); mem -= spk; outs.append(spk)
        return torch.stack(outs).permute(1,2,0)        # [B,N,T]

  
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.res = RecLIF(28*28, N_NEURONS)
        self.W   = nn.Parameter(torch.randn(N_NEURONS, 10) * 0.1)
    def forward(self, x):
        res = self.res(x)                      # [B,N,T]
        logits = res.mean(2) @ self.W          # [B,10]
        B,N,T = res.shape
        mem = x.new_zeros(B,10); hist=[]
        for t in range(T):
            mem  = 0.9*mem + res[:,:,t] @ self.W
            spk  = spike(mem-1); mem -= spk; hist.append(spk)
        out_spk = torch.stack(hist).permute(1,2,0)     # [B,10,T]
        return logits, res, out_spk

  
def apply_pixel_moves(flat_img, n_moves):
    """
    Move 'ink' by swapping n_moves foreground pixels (>0) with background
    pixels (==0). Works in-place; flat_img shape [B, 784] in [0,1].
    """
    if n_moves == 0:
        return flat_img
    B, P = flat_img.shape
    for b in range(B):
        fg_idx = torch.nonzero(flat_img[b] > 0, as_tuple=False).flatten()
        bg_idx = torch.nonzero(flat_img[b] == 0, as_tuple=False).flatten()
        if len(fg_idx) == 0 or len(bg_idx) == 0:
            continue
        k = min(n_moves, len(fg_idx), len(bg_idx))
        f_sel = fg_idx[torch.randperm(len(fg_idx))[:k]]
        b_sel = bg_idx[torch.randperm(len(bg_idx))[:k]]
        fg_vals = flat_img[b, f_sel].clone()
        flat_img[b, f_sel] = 0.0
        flat_img[b, b_sel] = fg_vals
    return flat_img

def encode_poisson(img_flat):               # img_flat ∈ [0,1]  [B,784]
    p = PR_MIN + img_flat * (PR_MAX - PR_MIN)
    r = torch.rand(len(img_flat), 28*28, T_STEPS, device=DEVICE)
    return (r < p.unsqueeze(2)).float()

def save_activity(batch_spk, hid, out_spk, digits, nlevels, names, root):
    os.makedirs(root, exist_ok=True)
    for i, nm in enumerate(names):
        pdir = f"{root}/{nm}_{digits[i]}-n{nlevels[i]}"
        act  = f"{pdir}/activity"; os.makedirs(act, exist_ok=True)
        np.save(f"{act}/in.npy",  batch_spk[i].cpu().numpy())
        np.save(f"{act}/out.npy", out_spk[i].cpu().numpy())
        for n in range(N_NEURONS):
            np.save(f"{act}/n{n}.npy", hid[i,n].cpu().numpy())

  
tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Lambda(lambda x: x.view(-1))])
train_ds = datasets.MNIST("./data", train=True,  download=True, transform=tf)
test_ds  = datasets.MNIST("./data", train=False, download=True, transform=tf)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE,
                                           shuffle=True)

VAL_PIX = test_ds.data[:128].float().view(128,-1)/255.
VAL_SPK = encode_poisson(VAL_PIX.to(DEVICE))
VAL_LBL = test_ds.targets[:128].to(DEVICE)

  
net   = MNISTNet().to(DEVICE)
opt   = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=LR,
            total_steps=EPOCHS*len(train_loader),
            pct_start=0.1, final_div_factor=10)
criterion = nn.CrossEntropyLoss()

with open(f"{ROOT}/learning_accuracy_log.csv","w",newline="") as f:
    csv.writer(f).writerow(["epoch","train_acc","val_acc"])

for ep in range(1, EPOCHS+1):
    net.train(); running = 0.0
    for imgs, digits in train_loader:
        imgs, digits = imgs.to(DEVICE), digits.to(DEVICE)

        # pick complexity level per-sample and move pixels
        levels  = torch.tensor(random.choices(list(NOISE_LEVELS), k=len(imgs)),
                               device=DEVICE)
        n_moves = torch.tensor([NOISE_LEVELS[int(l)] for l in levels],
                               device=DEVICE)
        imgs_noisy = imgs.clone()
        for i, mv in enumerate(n_moves):
            apply_pixel_moves(imgs_noisy[i:i+1], int(mv))

        spk = encode_poisson(imgs_noisy)
        logits, _, _ = net(spk)
        loss = criterion(logits, digits)

        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        running += (logits.argmax(1)==digits).float().mean().item()

    train_acc = running / len(train_loader)

    if ep % PRINT_EVERY == 0:
        net.eval()
        with torch.no_grad():
            v_logits, hid, o_spk = net(VAL_SPK)
            v_acc = (v_logits.argmax(1) == VAL_LBL).float().mean().item()
        print(f"E{ep:3d}  train {train_acc*100:5.1f}%  val {v_acc*100:5.1f}%")
        with open(f"{ROOT}/learning_accuracy_log.csv","a",newline="") as f:
            csv.writer(f).writerow([ep,f"{train_acc:.4f}",f"{v_acc:.4f}"])
        save_activity(VAL_SPK.cpu(), hid.cpu(), o_spk.cpu(),
                      VAL_LBL.cpu(), torch.zeros_like(VAL_LBL),
                      [f"val_{i:03d}" for i in range(len(VAL_LBL))],
                      f"{ROOT}/learning/epoch_{ep}")

torch.save(net.state_dict(), f"{ROOT}/model.pth")

  
net.eval()
tot_correct = 0
per_level_correct = {l: [0, 0] for l in NOISE_LEVELS}   # [correct,total]

with open(f"{ROOT}/evaluation_accuracy_log.csv","w",newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["id","label","correct"])

    sample_id = 0
    with torch.no_grad():
        loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE,
                                             shuffle=False)
        for imgs, digits in loader:
            if sample_id >= EVAL_SAMPLES:
                break
            imgs, digits = imgs.to(DEVICE), digits.to(DEVICE)

            # deterministic round-robin assignment
            levels = torch.tensor([list(NOISE_LEVELS)[
                                   (sample_id+i)%len(NOISE_LEVELS)]
                                   for i in range(len(imgs))], device=DEVICE)
            n_moves = torch.tensor([NOISE_LEVELS[int(l)] for l in levels],
                                   device=DEVICE)
            imgs_noisy = imgs.clone()
            for i,mv in enumerate(n_moves):
                apply_pixel_moves(imgs_noisy[i:i+1], int(mv))

            spk = encode_poisson(imgs_noisy)
            logits, hid, o_spk = net(spk)
            preds = logits.argmax(1)

            keep = min(len(imgs), EVAL_SAMPLES - sample_id)
            imgs_noisy, digits = imgs_noisy[:keep], digits[:keep]
            levels            = levels[:keep]
            spk, hid, o_spk   = spk[:keep], hid[:keep], o_spk[:keep]
            preds             = preds[:keep]

            names = [f"test_{sample_id+i:05d}" for i in range(keep)]
            save_activity(spk.cpu(), hid.cpu(), o_spk.cpu(),
                          digits.cpu(), levels.cpu(), names,
                          f"{ROOT}/evaluation")

            for p,t,l in zip(preds.cpu(), digits.cpu(), levels.cpu()):
                corr = int(p==t)
                writer.writerow([sample_id, f"{t.item()}-n{l.item()}", corr])
                per_level_correct[int(l)][0] += corr
                per_level_correct[int(l)][1] += 1
                tot_correct += corr
                sample_id  += 1

overall_acc = 100 * tot_correct / sample_id
with open(f"{ROOT}/evaluation_accuracy.txt","w") as f:
    f.write(f"overall_accuracy: {overall_acc:.2f}%\n")
    for l,(c,tot) in per_level_correct.items():
        acc = 100 * c / max(1,tot)
        f.write(f"noise_level_{l}: {acc:.2f}%  ({tot} samples)\n")

print(f"Evaluation finished – {sample_id} samples, "
      f"overall accuracy {overall_acc:.2f}%")
