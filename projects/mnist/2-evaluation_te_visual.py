#!/usr/bin/env python3
"""
Date: Feb 16, 2026
Description: Visualize TE adjacency matrices from MNIST evaluations as graphs and edge-weight histograms.
Inputs:
  - Experiment root containing evaluation/test_*/adj_k-<k>.npy files.
  - Command-line arguments selecting k and display controls (edge fraction, histogram bins, styling).
Outputs:
  - Graph and histogram images saved under <root>/diagrams/evaluation/te_graphs/.
"""

import argparse, os, re, sys, glob, math
import numpy as np
import matplotlib.pyplot as plt

  
def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--root", default="and_or_xor_3bit-20_N-64_P_HIGH-5_P_LOW-0_1")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--edge_frac", type=float, default=0.6)
    p.add_argument("--lw_min", type=float, default=0.1)
    p.add_argument("--lw_max", type=float, default=5.0)
    p.add_argument("--bins",   type=int,   default=100)
    p.add_argument("--alpha_bar", type=float, default=0.8)
    p.add_argument("--color_bar", default="tab:blue")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

  
def layout_positions(n_bits, n_hidden):
    pos={}
    for i in range(n_bits+3): pos[f"in_{i}"]=(0,-i)
    side=math.ceil(math.sqrt(n_hidden)); span=n_bits+3
    for h in range(n_hidden):
        r,c=divmod(h,side); pos[f"hid_{h}"]=(1+c/side,-r/side*span)
    pos["out_0"]=(2,-span/2)
    return pos

def infer_sizes(act_dir):
    files=glob.glob(os.path.join(act_dir,"*.npy"))
    n_hidden=len([f for f in files if re.search(r"[/\\]n\d+\.npy$",f)])
    n_input=np.load(os.path.join(act_dir,"data.npy"),mmap_mode="r").shape[0]
    return n_input-3, n_hidden

def draw_graph(adj,pos,lw_min,lw_max,frac,out_png):
    # keep only positive entries
    pos_mask=adj>0
    if not pos_mask.any():
        print("  [!] no positive edges – skip graph"); return
    vals=adj[pos_mask]
    thresh=np.quantile(vals, frac)
    src,dst=np.where((adj>0)&(adj<=thresh))
    lo=vals.min(); rng=thresh-lo+1e-9
    a_min,a_max=0.1,0.8
    plt.figure(figsize=(7,4)); ax=plt.gca(); ax.axis("off")
    for lbl,(x,y) in pos.items():
        ax.scatter(x,y,s=100,c="white",edgecolor="black",zorder=3)
        ax.text(x,y,lbl,ha="center",va="center",fontsize=7,zorder=4)
    for s,d in zip(src,dst):
        w=(thresh-adj[s,d])/rng
        lw=lw_min+(lw_max-lw_min)*w
        alpha=a_min+(a_max-a_min)*w
        (x0,y0),(x1,y1)=pos[s],pos[d]
        ax.plot([x0,x1],[y0,y1],lw=lw,alpha=alpha,c="tab:blue")
    plt.tight_layout(); plt.savefig(out_png,dpi=150); plt.close()

def draw_hist(adj,bins,color,alpha_bar,lw_min,lw_max,frac,out_png):
    vals=adj[adj>0]
    if vals.size==0:
        print("  [!] no positive edges – skip hist"); return
    lo,hi=vals.min(),vals.max()
    plt.figure(figsize=(5,3))
    plt.hist(vals,bins=bins,range=(lo,hi),color=color,alpha=alpha_bar)
    plt.xlabel("TE weight  (lower = stronger)"); plt.ylabel("count")
    thresh=np.quantile(vals, frac); rng=thresh-lo+1e-9
    w=(thresh-vals)/rng; w=np.clip(w,0,1)
    lw=lw_min+(lw_max-lw_min)*w
    plt.title(f"mean lw={lw.mean():.3f}  |  ΣTE={vals.sum():.3f}")
    plt.tight_layout(); plt.savefig(out_png,dpi=150); plt.close()

# --- main ---
def main():
    a=parse_args()
    eval_dir=os.path.join(a.root,"evaluation")
    if not os.path.isdir(eval_dir): sys.exit("evaluation/ not found")
    out_dir=os.path.join(a.root,"diagrams","evaluation","te_graphs")
    os.makedirs(out_dir,exist_ok=True)

    tests=sorted(d for d in os.listdir(eval_dir) if d.startswith("test_"))
    if not tests: sys.exit("no test folders")
    n_bits,n_hid=infer_sizes(os.path.join(eval_dir,tests[0],"activity"))
    node_order=[f"in_{i}" for i in range(n_bits+3)]+[f"hid_{h}" for h in range(n_hid)]+["out_0"]
    pos_raw=layout_positions(n_bits,n_hid)
    pos={i:pos_raw[lbl] for i,lbl in enumerate(node_order)}

    for td in tests:
        adj_p=os.path.join(eval_dir,td,f"adj_k-{a.k}.npy")
        if not os.path.isfile(adj_p): continue
        g_png=os.path.join(out_dir,f"{td}_graph.png")
        h_png=os.path.join(out_dir,f"{td}_hist.png")
        if not a.overwrite and os.path.isfile(g_png) and os.path.isfile(h_png):
            continue
        adj=np.load(adj_p)
        adj[np.isinf(adj)] = 0
        adj = adj * -1
        draw_graph(adj,pos,a.lw_min,a.lw_max,a.edge_frac,g_png)
        draw_hist(adj,a.bins,a.color_bar,a.alpha_bar,
                  a.lw_min,a.lw_max,a.edge_frac,h_png)
        print("Saved", td)

if __name__=="__main__":
    main()
