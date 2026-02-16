#!/usr/bin/env python3
"""
Date: Feb 16, 2026
Description: Train and evaluate a recurrent spiking neural network on logic-gate tasks (AND, OR, XOR) and save spike activity for downstream TE+PH analysis.
Inputs:
  - Command-line arguments (see --help) controlling network size, task configuration, training schedule, and output paths.
Outputs:
  - Experiment folder containing training logs, evaluation logs, and per-trial spike arrays used by later TDA scripts.
"""

import os, csv, itertools, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
import argparse

# parse command-line arguments to override hyper-parameters
def parse_args():
    """Parse configurable parameters for the logic gate experiment."""
    parser = argparse.ArgumentParser(description="Logic gate LIF spiking neural network experiment")
    parser.add_argument("--n_bits", type=int, default=1, help="Number of input bits for logic gates")
    parser.add_argument("--test_set_size", type=int, default=4096, help="Number of samples in the evaluation set")
    parser.add_argument("--print_every", type=int, default=10, help="Print progress and dump spikes every N epochs")
    parser.add_argument("--patience", type=int, default=3, help="Early‑stopping patience once train/test accuracy reach 100%")
    parser.add_argument("--t_steps", type=int, default=20, help="Number of discrete time steps per sample")
    parser.add_argument("--n_neurons", type=int, default=64, help="Number of LIF neurons in the recurrent layer")
    parser.add_argument("--lr_base", type=float, default=1e-2, help="Base learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-45, help="Weight decay coefficient")
    parser.add_argument("--epochs", type=int, default=15000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Training batch size")
    parser.add_argument("--pr1", type=float, default=0.5, help="Spike probability for bit value 1")
    parser.add_argument("--pr0", type=float, default=0.0, help="Spike probability for bit value 0")
    parser.add_argument("--beta_mem", type=float, default=0.9, help="Decay constant for membrane potential")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device (cuda or cpu)")
    parser.add_argument("--save_root", type=str, default=None,
                        help="Custom output directory (defaults to pattern based on n_bits, t_steps and n_neurons)")
    return parser.parse_args()

args = parse_args()

# assign global configuration from parsed arguments
N_BITS        = args.n_bits
TEST_SET_SIZE = args.test_set_size
PRINT_EVERY   = args.print_every
PATIENCE      = args.patience
T_STEPS       = args.t_steps
N_NEURONS     = args.n_neurons
LR_BASE       = args.lr_base
WEIGHT_DECAY  = args.weight_decay
EPOCHS        = args.epochs
BATCH_SIZE    = args.batch_size
PR1, PR0      = args.pr1, args.pr0
BETA_MEM      = args.beta_mem
DEVICE        = args.device
INPUT_SIZE    = N_BITS + 3

if args.save_root:
    SAVE_ROOT = args.save_root
else:
    SAVE_ROOT = f"and_or_xor_{N_BITS}bit-{T_STEPS}_N-{N_NEURONS}_P_HIGH-5_P_LOW-0_2"
plt.rcParams["figure.dpi"] = 120

GATE_NAMES = ["and", "or", "xor"]
GATE_FUNC  = {
    "and": lambda bits: bits.float().all(dim=-1).float(),
    "or" : lambda bits: bits.float().any(dim=-1).float(),
    "xor": lambda bits: (bits.float().sum(dim=-1) % 2).float(),
}

# create required output folders
for sub in ("learning", "evaluation", "diagrams"):
    os.makedirs(os.path.join(SAVE_ROOT, sub), exist_ok=True)

  
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,v): ctx.save_for_backward(v); return (v>0.).float()
    @staticmethod
    def backward(ctx,g): (v,)=ctx.saved_tensors; return g*(1/(1+v.abs())**2)
spike = SpikeFn.apply

  
class RecLIF(nn.Module):
    def __init__(self,n_in,n_rec,beta=BETA_MEM):
        super().__init__()
        self.w_in  = nn.Parameter(torch.randn(n_in ,n_rec)*.1)
        self.w_rec = nn.Parameter(torch.randn(n_rec,n_rec)*.05)
        self.beta  = beta
    def forward(self,x):
        B,_,T=x.shape; mem=x.new_zeros(B,self.w_rec.size(0)); hist=[]
        for t in range(T):
            cur=x[:,:,t]@self.w_in + (hist[-1]@self.w_rec if hist else 0.)
            mem=self.beta*mem+cur; spk=spike(mem-1); mem-=spk; hist.append(spk)
        return torch.stack(hist).permute(1,2,0)

class LogicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.res=RecLIF(INPUT_SIZE,N_NEURONS)
        self.w  =nn.Parameter(torch.randn(N_NEURONS,1)*.1)
    def forward(self,x):
        res=self.res(x); logits=(res.mean(2)@self.w).squeeze(1)
        B,N,T=res.shape; mem_o=x.new_zeros(B,1); outs=[]
        for t in range(T):
            mem_o=0.9*mem_o+res[:,:,t]@self.w
            spk_o=spike(mem_o-1); mem_o-=spk_o; outs.append(spk_o)
        return logits,res,torch.stack(outs).permute(1,2,0)

  
def encode_poisson(data):
    prob=torch.where(data==1,PR1,PR0)
    rand=torch.rand(*prob.shape,T_STEPS,device=DEVICE)
    return (rand<prob.unsqueeze(2)).float()

def bits_to_label(bits, gate_name):
    """bits can be tuple/list or 1-D tensor."""
    if isinstance(bits, torch.Tensor):
        bits = bits.tolist()
    bit_str = "_".join(map(str, bits))
    return f"{bit_str}_{gate_name}"

def make_batch(batch_size):
    bits  = torch.randint(0,2,(batch_size,N_BITS),device=DEVICE)
    gates = torch.randint(0,3,(batch_size,1),device=DEVICE)
    onehot=torch.zeros(batch_size,3,device=DEVICE)
    onehot[torch.arange(batch_size),gates.squeeze()] = 1
    data=torch.cat([bits.float(), onehot], 1)
    spikes=encode_poisson(data)
    labels=[]
    names=[]
    for b,g in zip(bits, gates.squeeze()):
        gname=GATE_NAMES[int(g)]
        labels.append(GATE_FUNC[gname](b.unsqueeze(0))[0])
        names.append(bits_to_label(b, gname))
    labels=torch.stack(labels)
    return spikes, labels, names

  
truth_bits=list(itertools.product((0,1), repeat=N_BITS))
truth_patterns=[(bits,g) for bits in truth_bits for g in GATE_NAMES]
with torch.random.fork_rng(devices=[DEVICE]):
    torch.manual_seed(42)
    learn_rows=[]; learn_y=[]; learn_names=[]
    for bits,g in truth_patterns:
        one=[int(g=="and"),int(g=="or"),int(g=="xor")]
        learn_rows.append([*bits,*one])
        learn_y.append(int(GATE_FUNC[g](torch.tensor(bits)).item()))
        learn_names.append(bits_to_label(bits,g))
    learn_rows=torch.tensor(learn_rows,device=DEVICE,dtype=torch.float)
    LEARN_SPIKES=encode_poisson(learn_rows)
    LEARN_LABELS=torch.tensor(learn_y,device=DEVICE).float()
    LEARN_FOLDERS=learn_names

  
@torch.no_grad()
def run_and_store(net,spk,lbl,names,out_root,png=False):
    os.makedirs(out_root,exist_ok=True)
    logit,hid,out=net(spk)
    preds=(torch.sigmoid(logit)>.5).float()
    acc=(preds==lbl).float().mean().item()*100
    corr=(preds==lbl).cpu().numpy().astype(int)
    for i,name in enumerate(names):
        act=f"{out_root}/{name}/activity"; os.makedirs(act,exist_ok=True)
        np.save(f"{act}/data.npy", spk[i].cpu().numpy())
        np.save(f"{act}/out0.npy", out[i,0].cpu().numpy())
        for n in range(N_NEURONS):
            np.save(f"{act}/n{n}.npy", hid[i,n].cpu().numpy())
        if png:
            labs=[f"b{j}" for j in range(N_BITS)]+["gAnd","gOr","gXor","out"]
            fig,ax=plt.subplots(len(labs),1,figsize=(6,6),sharex=True,constrained_layout=True)
            for r in range(INPUT_SIZE):
                t=np.arange(T_STEPS); ax[r].eventplot(t[spk[i,r].cpu().numpy().astype(bool)])
                ax[r].set_yticks([]); ax[r].set_ylabel(labs[r],rotation=0,labelpad=15)
            t=np.arange(T_STEPS); ax[-1].eventplot(t[out[i,0].cpu().numpy().astype(bool)])
            ax[-1].set_yticks([]); ax[-1].set_ylabel("out",rotation=0,labelpad=15)
            ax[-1].set_xlabel("t (ms)")
            plt.suptitle(f"{name} → {int(preds[i])}"); plt.savefig(f"{out_root}/{name}/io.png"); plt.close()
    return acc, corr

  
model_path=f"{SAVE_ROOT}/model.pth"
net=LogicNet().to(DEVICE)

if not os.path.isfile(model_path):
    print("Training…")
    opt=torch.optim.AdamW(net.parameters(),lr=LR_BASE,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=3e-3,total_steps=EPOCHS,
                                             pct_start=0.1,final_div_factor=1e4,anneal_strategy="cos")
    crit=nn.BCEWithLogitsLoss()
    learn_csv=f"{SAVE_ROOT}/learning_accuracy_log.csv"
    with open(learn_csv,"w",newline="") as f:
        csv.writer(f).writerow(["epoch","train_acc","test_acc",*LEARN_FOLDERS])

    acc0,_=run_and_store(net,LEARN_SPIKES,LEARN_LABELS,LEARN_FOLDERS,
                         f"{SAVE_ROOT}/learning/epoch_0")
    with open(learn_csv,"a",newline="") as f:
        csv.writer(f).writerow([0,0.0,f"{acc0:.2f}",*(["0"]*len(LEARN_FOLDERS))])

    consec_100 = 0
    for ep in range(1,EPOCHS+1):
        xb,yb,_=make_batch(BATCH_SIZE)
        logit,_,_=net(xb); loss=crit(logit,yb)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
        opt.step(); sched.step()
        train_acc=((torch.sigmoid(logit)>.5)==yb).float().mean().item()*100
        if ep%PRINT_EVERY==0:
            tacc,per=run_and_store(net,LEARN_SPIKES,LEARN_LABELS,LEARN_FOLDERS,
                                   f"{SAVE_ROOT}/learning/epoch_{ep}")
            print(f"E{ep:4d} loss {loss.item():.4f}  train {train_acc:5.1f}%  test {tacc:5.1f}%")
            with open(learn_csv,"a",newline="") as f:
                csv.writer(f).writerow([ep,f"{train_acc:.2f}",f"{tacc:.2f}",*per.tolist()])

            if train_acc == 100.0 and tacc == 100.0:
                consec_100 += 1
            else:
                consec_100 = 0

            if PATIENCE and consec_100 >= PATIENCE:
                print(f"\nEarly stop: 100 % train & test for {PATIENCE} checkpoints.")
                break

    torch.save(net.state_dict(),model_path); print("Weights saved.")
else:
    net.load_state_dict(torch.load(model_path,map_location=DEVICE))
    print("Loaded weights.")

  
print("Final evaluation …")

# fresh random batch
ev_spk, ev_lbl, ev_label_str = make_batch(TEST_SET_SIZE)
sample_ids = [f"test_{i:03d}" for i in range(TEST_SET_SIZE)]

# forward + storage
ev_root = f"{SAVE_ROOT}/evaluation"
acc, correct = run_and_store(net, ev_spk, ev_lbl, sample_ids, ev_root, png=False)

  
with open(f"{SAVE_ROOT}/evaluation_accuracy_log.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["id","label","correct"])
    for i,(lab,flag) in enumerate(zip(ev_label_str, correct)):
        w.writerow([i, lab, int(flag)])

  
per_class={}
for lab,flag in zip(ev_label_str, correct):
    per_class.setdefault(lab, []).append(flag)
with open(f"{SAVE_ROOT}/evaluation_accuracy.txt","w") as f:
    f.write(f"overall_accuracy: {acc:.2f}%\n")
    for lab,vec in per_class.items():
        f.write(f"{lab}: {np.mean(vec)*100:.2f}%\n")

print(f"Evaluation accuracy {acc:.2f}% – done.")
