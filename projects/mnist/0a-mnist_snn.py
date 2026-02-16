#!/usr/bin/env python3
"""
Date: Feb 16, 2026
Description: Train and evaluate an SNN on MNIST with bit-flip noise, saving spike activity for TE+PH analysis.
Inputs:
  - Command-line arguments (see --help) controlling training, network size, noise settings, and output paths.
  - MNIST dataset (downloaded via torchvision if not present).
Outputs:
  - Experiment folder containing trained weights, logs, and spike activity arrays for learning and evaluation phases.
"""

import os, csv, math, itertools, numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import argparse

plt.rcParams["figure.dpi"] = 120

# argument parser to override hyper‑parameters
def parse_args():
    parser = argparse.ArgumentParser(description="MNIST LIF spiking neural network experiment")
    parser.add_argument("--root", type=str, default="mnist_network_64", help="Output root directory")
    parser.add_argument("--t_steps", type=int, default=100, help="Number of discrete time steps per sample")
    parser.add_argument("--n_neurons", type=int, default=64, help="Number of LIF neurons in the recurrent layer")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--print_every", type=int, default=1, help="Save spikes and evaluate every N epochs")
    parser.add_argument("--pr_max", type=float, default=0.50, help="Spike probability for pixel value 1 (255)")
    parser.add_argument("--pr_min", type=float, default=0.02, help="Spike probability for pixel value 0")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--eval_samples", type=int, default=1024, help="Number of test samples to evaluate (<=10000)")
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

os.makedirs(ROOT, exist_ok=True)
os.makedirs(f"{ROOT}/learning", exist_ok=True)
os.makedirs(f"{ROOT}/evaluation", exist_ok=True)

  
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

    def forward(self, x):                 # x [B, n_in, T]
        B, _, T = x.shape
        mem = x.new_zeros(B, self.w_rec.size(0))
        out = []
        for t in range(T):
            cur = x[:, :, t] @ self.w_in + (out[-1] @ self.w_rec if out else 0.)
            mem = self.beta * mem + cur
            spk = spike(mem - 1.0)
            mem -= spk
            out.append(spk)
        return torch.stack(out).permute(1, 2, 0)     # [B, N, T]

  
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.res = RecLIF(28 * 28, N_NEURONS)
        self.W   = nn.Parameter(torch.randn(N_NEURONS, 10) * 0.1)

    def forward(self, x):                 # x [B, 784, T]
        res_spk = self.res(x)             # [B, N, T]
        logits  = res_spk.mean(2) @ self.W   # [B, 10]

        # spiking read-out (optional diagnostics)
        B, N, T = res_spk.shape
        mem = x.new_zeros(B, 10); outs = []
        for t in range(T):
            mem   = 0.9 * mem + res_spk[:, :, t] @ self.W
            spk_o = spike(mem - 1.0); mem -= spk_o; outs.append(spk_o)
        out_spk = torch.stack(outs).permute(1, 2, 0)   # [B, 10, T]
        return logits, res_spk, out_spk

  
def encode_poisson(img_batch):            # img_batch [B, 784] float 0–1
    p = PR_MIN + img_batch * (PR_MAX - PR_MIN)        # [B, 784]
    p = p.to(DEVICE)
    r = torch.rand(len(img_batch), 28*28, T_STEPS, device=DEVICE)
    return (r < p.unsqueeze(2)).float()               # [B, 784, T]

  
tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Lambda(lambda x: x.view(-1))])
train_ds = datasets.MNIST(root="./data", train=True,  download=True,
                          transform=tf)
test_ds  = datasets.MNIST(root="./data", train=False, download=True,
                          transform=tf)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE,
                                           shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=BATCH_SIZE,
                                           shuffle=False)

  
VAL_SPKES = encode_poisson(test_ds.data[:128].float().view(128, -1) / 255.)
VAL_LABEL = test_ds.targets[:128].to(DEVICE)

  
def save_activity(batch_spk, hid, out_spk, labels, root_dir, names):
    os.makedirs(root_dir, exist_ok=True)
    for i, name in enumerate(names):
        pdir = f"{root_dir}/{name}_{labels[i].item()}"
        act  = f"{pdir}/activity"
        os.makedirs(act, exist_ok=True)
        np.save(f"{act}/in.npy",  batch_spk[i].cpu().numpy())
        np.save(f"{act}/out.npy", out_spk[i].cpu().numpy())
        for n in range(N_NEURONS):
            np.save(f"{act}/n{n}.npy", hid[i, n].cpu().numpy())

  
net   = MNISTNet().to(DEVICE)
opt   = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR,
                                            total_steps=EPOCHS*len(train_loader),
                                            pct_start=0.1, final_div_factor=10)
criterion = nn.CrossEntropyLoss()

csv_path = f"{ROOT}/learning_accuracy_log.csv"
with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "train_acc", "val_acc"])

best_val = 0.0
for epoch in range(1, EPOCHS + 1):
    net.train(); running = 0.0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        spk        = encode_poisson(imgs)
        logits, _, _ = net(spk)
        loss = criterion(logits, lbls)

        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        running += (logits.argmax(1) == lbls).float().mean().item()

    train_acc = running / len(train_loader)

    # validation & spike dump
    if epoch % PRINT_EVERY == 0:
        net.eval()
        with torch.no_grad():
            logits, hid, out_spk = net(VAL_SPKES)
            val_acc = (logits.argmax(1) == VAL_LABEL).float().mean().item()
            print(f"E{epoch:3d}  train {train_acc*100:5.1f}%  val {val_acc*100:5.1f}%")
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{train_acc:.4f}", f"{val_acc:.4f}"])

            save_activity(VAL_SPKES.cpu(), hid.cpu(), out_spk.cpu(),
                          VAL_LABEL.cpu(), f"{ROOT}/learning/epoch_{epoch}",
                          [f"val_{i:03d}" for i in range(len(VAL_LABEL))])

            if val_acc > 0.97 and val_acc > best_val:
                best_val = val_acc
                torch.save(net.state_dict(), f"{ROOT}/best_model.pth")
                print("New best model saved.")

  
torch.save(net.state_dict(), f"{ROOT}/model.pth")

  
net.eval()
correct = 0; total = 0
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, lbls in test_loader:
        if total >= EVAL_SAMPLES:
            break
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        spk = encode_poisson(imgs)
        logits, hid, out_spk = net(spk)
        preds = logits.argmax(1)

        batch_size = len(lbls)
        # clamp if this would exceed limit
        if total + batch_size > EVAL_SAMPLES:
            keep = EVAL_SAMPLES - total
            imgs, lbls = imgs[:keep], lbls[:keep]
            spk, hid, out_spk = spk[:keep], hid[:keep], out_spk[:keep]
            preds = preds[:keep]
            batch_size = keep

        start_idx = total
        names = [f"test_{start_idx+i:05d}" for i in range(batch_size)]
        save_activity(spk.cpu(), hid.cpu(), out_spk.cpu(),
                      lbls.cpu(), f"{ROOT}/evaluation", names)

        correct += (preds == lbls).sum().item()
        total   += batch_size
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(lbls.cpu().tolist())

test_acc = 100.0 * correct / total
with open(f"{ROOT}/evaluation_accuracy.txt", "w") as f:
    f.write(f"overall_accuracy: {test_acc:.2f}%\n")
print(f"Test accuracy on {total} samples: {test_acc:.2f}%")

with open(f"{ROOT}/evaluation_accuracy_log.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "label", "correct"])
    for idx, (pred, true) in enumerate(zip(all_preds, all_labels)):
        w.writerow([idx, true, int(pred == true)])
