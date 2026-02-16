#!/usr/bin/env python3
"""
Date: Feb 16, 2026
Description: Analyze MNIST training-phase spike data by computing TE, persistence diagrams, and Betti AUBC across epochs.
Inputs:
  - Experiment root containing learning/epoch_*/activity/ spike arrays and (optionally) learning_accuracy_log.csv.
  - Command-line arguments selecting TE history length k and homology dimensions.
Outputs:
  - Cached TE/PD/AUBC artifacts under <root>/learning/epoch_*.
  - Summary plots under <root>/diagrams/learning/.
"""

import argparse, os, re, sys, glob, csv
import numpy as np
import matplotlib.pyplot as plt

this_dir = os.path.dirname(os.path.abspath(__file__))
tda_path = os.path.abspath(os.path.join(this_dir, "..", "..", "common", "time-series-tda"))
sys.path.append(tda_path)
from transfer_entropy import compute_transfer_entropy_matrix
from adjacency        import preprocess_adjacency_matrix
from persistence      import compute_persistence_diagram
from io_utils         import save_persistence_diagram


  
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="and_or_xor_5bit-20_N-64_P_HIGH-5_P_LOW-0_1",
                    help="Experiment root (contains learning/)")
    ap.add_argument("--k", type=int, default=10,
                    help="History length for tr ansfer entropy")
    ap.add_argument("--dims", type=int, nargs="+", default=[0,1,2,3,4,5,6,7],
                    help="Homology dimensions to keep")
    ap.add_argument("-f", "--overwrite_existing", action="store_true",
                    help="Force recompute even if files exist")
    return ap.parse_args()

  
def betti_auc(diag, dims):
    auc={d:0.0 for d in dims}
    for b,d,dim in diag:
        if int(dim) in dims and np.isfinite(b) and np.isfinite(d):
            auc[int(dim)]+=d-b
    return auc

def load_spikes(act_dir):
    trains=[]
    for f in sorted(glob.glob(os.path.join(act_dir,"*.npy"))):
        arr=np.load(f).astype(float)
        arr=arr.reshape(arr.shape[0],-1) if arr.ndim>1 else arr[None,:]
        trains.extend(arr)
    return np.vstack(trains)                    # (N,T)

  
def main():
    args=parse_args()
    root=args.root; k=args.k; dims=args.dims; force=args.overwrite_existing

    learn_dir=os.path.join(root,"learning")
    if not os.path.isdir(learn_dir):
        sys.exit("learning/ folder not found")

    diag_dir=os.path.join(root,"diagrams","learning")
    os.makedirs(diag_dir, exist_ok=True)

    # read learning_accuracy_log.csv if available
    acc_log={}
    acc_csv=os.path.join(root,"learning_accuracy_log.csv")
    if os.path.isfile(acc_csv):
        with open(acc_csv,newline="") as f:
            for row in csv.DictReader(f):
                ep=row["epoch"]; acc_log[ep]={k:float(v) for k,v in row.items() if k!="epoch"}
    else:
        print("[!] learning_accuracy_log.csv missing – lower subplot disabled")

    # pattern → dim → [(epoch,val)]
    auc_table={}

    epoch_dirs=sorted([d for d in os.listdir(learn_dir)
                       if os.path.isdir(os.path.join(learn_dir,d))],
                      key=lambda s:int(re.findall(r'\d+',s)[0]))

    for ep in epoch_dirs:
        ep_path=os.path.join(learn_dir,ep)
        epoch_id=re.findall(r'\d+',ep)[0]
        print(f"\n=== Epoch {epoch_id} ===")

        patterns=sorted(d for d in os.listdir(ep_path)
                        if os.path.isdir(os.path.join(ep_path,d)))
        for patt in patterns:
            patt_path=os.path.join(ep_path,patt)
            act_dir =os.path.join(patt_path,"activity")
            if not os.path.isdir(act_dir): continue

            spikes=load_spikes(act_dir)
            print(f"  {patt:12s}  {spikes.shape}")

            adj_file=os.path.join(patt_path,f"adj_k-{k}.npy")
            ph_file =os.path.join(patt_path,f"pd_k-{k}.npy")
            auc_file=os.path.join(patt_path,f"betti_auc_k-{k}.txt")

            # TE adjacency
            if force or not os.path.exists(adj_file):
                adj = compute_transfer_entropy_matrix(spikes, k=k, normalize=False)
                adj = preprocess_adjacency_matrix(adj, invert=True,
                                                  allow_reflexive=False, allow_bijective=False, normalize=False)
                np.save(adj_file,adj)
            else:
                adj=np.load(adj_file)

            # persistence
            if force or not os.path.exists(ph_file):
                diag=compute_persistence_diagram(adj,homology_dimensions=dims)
                save_persistence_diagram(diag,ph_file)
            else:
                diag=np.load(ph_file)

            # Betti-AUC
            if force or not os.path.exists(auc_file):
                auc_vals=betti_auc(diag,dims)
                with open(auc_file,"w") as f:
                    for d,v in auc_vals.items(): f.write(f"betti_auc_{d}: {v}\n")
            else:
                with open(auc_file) as f:
                    auc_vals={int(l.split(":")[0].split("_")[-1]):float(l.split(":")[1]) for l in f}

            auc_table.setdefault(patt,{d:[] for d in dims})
            for d in dims: auc_table[patt][d].append((epoch_id,auc_vals[d]))

# Plotting
    palette=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
             "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    import itertools

    for patt,series_by_dim in auc_table.items():
        epochs=sorted({ep for lst in series_by_dim.values() for ep,_ in lst}, key=int)
        xs=[int(e) for e in epochs]
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(7,6),sharex=True,
                                   gridspec_kw={"height_ratios":[2,1]},constrained_layout=True)

        # top: Betti curves
        cyc=itertools.cycle(palette)
        for d,ser in series_by_dim.items():
            ser_dict=dict(ser)
            ys=[ser_dict.get(e,np.nan) for e in epochs]
            ax1.plot(xs,ys,label=f"Dim {d}",color=next(cyc),linewidth=2)
        ax1.set_ylabel("Σ lifetimes"); ax1.set_title(f"{patt}  Betti-AUC")
        ax1.grid(True,ls="--",alpha=.6); ax1.legend()

        # bottom: accuracy curves
        if acc_log:
            train=[acc_log.get(e,{}).get("train_acc",np.nan) for e in epochs]
            test =[acc_log.get(e,{}).get("test_acc" ,np.nan) for e in epochs]
            patt_acc=[acc_log.get(e,{}).get(patt,np.nan)*100 for e in epochs]
            ax2.plot(xs,train,label="Train %",color="#1f77b4")
            ax2.plot(xs,test ,label="Test %", color="#d62728")
            ax2.plot(xs,patt_acc,label="Pattern %",color="#2ca02c")
            ax2.set_ylim(0,100); ax2.grid(True,ls="--",alpha=.6); ax2.legend()
            ax2.set_ylabel("Accuracy (%)")
        ax2.set_xlabel("Epoch")

        out= os.path.join(diag_dir,f"{patt}_betti_auc.png")
        fig.savefig(out,dpi=150); plt.close(fig)
        print("Diagram saved →",out)

    # combined CSV
    csv_out=os.path.join(diag_dir,"betti_auc_over_epochs.csv")
    with open(csv_out,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["pattern","epoch",*[f"auc{d}" for d in dims]])
        for patt,dseries in auc_table.items():
            for ep in epochs:
                w.writerow([patt,ep]+[dict(dseries[d]).get(ep,np.nan) for d in dims])
    print("CSV table →",csv_out)

if __name__ == "__main__":
    main()
