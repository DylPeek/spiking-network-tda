#!/usr/bin/env python3
"""
Date: Feb 16, 2026
Description: Compute TE adjacency matrices and persistence diagrams from MNIST evaluation spike data, then summarize topology via Betti curves, AUBC, and Wasserstein comparisons.
Inputs:
  - Experiment root containing evaluation/test_*/activity/ spike arrays and evaluation_accuracy_log.csv.
  - Command-line arguments selecting TE history length k, homology dimensions, and grouping modes (noise labels).
Outputs:
  - Cached adjacency matrices and persistence diagrams under evaluation/test_*.
  - Summary CSVs and figures under <root>/diagrams/evaluation/.
"""

import argparse, os, re, sys, glob, csv
import numpy as np, matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# Local imports (time-series-tda helpers)
this_dir = os.path.dirname(os.path.abspath(__file__))
tda_path = os.path.abspath(os.path.join(this_dir, "..", "..", "common", "time-series-tda"))
sys.path.append(tda_path)
from transfer_entropy import compute_transfer_entropy_matrix
from adjacency        import preprocess_adjacency_matrix
from persistence      import compute_persistence_diagram
from io_utils         import save_persistence_diagram


# --- Core utilities ---

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",  default="./mnist_network_64",
                    help="experiment root folder")
    ap.add_argument("--k",     type=int, default=10,
                    help="history length for transfer-entropy")
    ap.add_argument("--dims",  type=int, nargs="+",
                    default=[0,1,2,3,4,5,6,7],
                    help="Betti dimensions to analyse")
    ap.add_argument("--modes", nargs="+", default=["n0","n1","n2","n3","n4","n5","n6","n7","n8"],
                    help="sub-string keywords that group labels into modes")
    ap.add_argument("--overwrite_existing", type=bool, default=False)

    # Betti-curve controls
    ap.add_argument("--betti_grid_points", type=int, default=200,
                    help="number of filtration grid points for Betti curves")
    ap.add_argument("--min_persistence", type=float, default=0.0,
                    help="ignore features with persistence below this threshold in Betti curves")
    return ap.parse_args()


def ensure_dirs(root):
    diag_root = os.path.join(root, "diagrams", "evaluation")
    # AUBC figure tree
    aubc_root  = os.path.join(diag_root, "AUBC")
    bc_root    = os.path.join(diag_root, "BC")
    dirs = {
        "diag_root": diag_root,
        # AUBC
        "aubc_root": aubc_root,
        "aubc_per_dim_dir":        os.path.join(aubc_root, "per_dim_violins"),
        "aubc_per_lab_dir":        os.path.join(aubc_root, "per_label_violins"),
        "aubc_per_mode_dir":       os.path.join(aubc_root, "per_mode_violins"),
        "aubc_per_mode_dim_dir":   os.path.join(aubc_root, "per_mode_dim_violins"),
        "aubc_heatND_dir":         os.path.join(aubc_root, "mode_nd_heatmaps"),
        "aubc_dist_dir":           os.path.join(aubc_root, "mode_nd_distances"),
        # BC
        "bc_root":                 bc_root,
        "bc_betti_dir":            os.path.join(bc_root, "per_mode_dim_betti_curves"),
        "bc_heatND_dir":           os.path.join(bc_root, "mode_nd_heatmaps"),
        "bc_dist_dir":             os.path.join(bc_root, "mode_nd_distances"),
    }
    for p in dirs.values(): os.makedirs(p, exist_ok=True)
    return dirs


def load_id2label(root):
    id2lab = {}
    with open(os.path.join(root, "evaluation_accuracy_log.csv")) as f:
        for r in csv.DictReader(f):
            id2lab[int(r["id"])] = r["label"]
    return id2lab


def load_spikes(act_dir):
    """Load spike npy blocks in directory, flatten per file to time axis and stack -> (N,T)."""
    arrs = []
    for f in sorted(glob.glob(os.path.join(act_dir, "*.npy"))):
        if "in" in os.path.basename(f):
            print("Skipping input array:", os.path.basename(f))
            continue
        a = np.load(f).astype(float)
        arrs.extend(a.reshape(a.shape[0], -1) if a.ndim > 1 else [a])
    return np.vstack(arrs) if arrs else np.zeros((0,))


def compute_or_load_adjacency(act_dir, adj_path, k, force):
    if force or not os.path.exists(adj_path):
        spikes = load_spikes(act_dir)
        adj = preprocess_adjacency_matrix(
            compute_transfer_entropy_matrix(spikes, k=k, normalize=False),
            invert=True, allow_reflexive=False,
            allow_bijective=False, normalize=False)
        np.save(adj_path, adj)
    else:
        adj = np.load(adj_path)
    return adj


def compute_or_load_pd(adj, pd_path, dims, force):
    if force or not os.path.exists(pd_path):
        pd = compute_persistence_diagram(adj, homology_dimensions=dims)
        save_persistence_diagram(pd, pd_path)
    else:
        pd = np.load(pd_path, allow_pickle=True)
    return pd


def betti_auc(pd, dims):
    out = {d: 0.0 for d in dims}
    for b, dth, dim in pd:
        if int(dim) in dims and np.isfinite(b) and np.isfinite(dth):
            out[int(dim)] += (dth - b)
    return out


def classical_mds(D, n_components=2):
    """Return coordinates via classical MDS for distance matrix D."""
    n = D.shape[0]
    D2 = D**2
    J  = np.eye(n) - np.ones((n, n)) / n
    B  = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:n_components]
    L   = np.diag(np.maximum(eigvals[idx], 0))**0.5
    V   = eigvecs[:, idx]
    return V @ L


# --- Betti curves (NEW) ---

def betti_curves_from_pd(pd, dims, t_grid, min_persistence=0.0):
    """
    Compute Betti curves β_d(t) for a single PD over t_grid.
    Infinite deaths count until the end of the grid; features with persistence < threshold are skipped.
    """
    curves = {int(d): np.zeros_like(t_grid, dtype=float) for d in dims}
    if pd is None or len(pd) == 0 or len(t_grid) == 0:
        return curves

    tail  = (t_grid[1] - t_grid[0]) if len(t_grid) > 1 else 0.0
    t_end = t_grid[-1] + tail

    for rec in pd:
        b, de, dim = float(rec[0]), float(rec[1]) if np.isfinite(rec[1]) else np.inf, int(rec[2])
        if dim not in curves or not np.isfinite(b):
            continue
        if not np.isfinite(de):
            de = t_end
        if (de - b) < float(min_persistence):
            continue
        mask = (t_grid >= b) & (t_grid < de)
        if np.any(mask):
            curves[dim][mask] += 1.0
    return curves


def aggregate_class_averaged_betti(pds_by_mode, dims, t_grid, min_persistence=0.0):
    """
    Average Betti curves per mode across its PD list.
    Returns: {mode -> {dim -> avg_curve}}
    """
    out = {}
    for m, pd_list in pds_by_mode.items():
        if not pd_list:
            continue
        sums  = {int(d): np.zeros_like(t_grid, dtype=float) for d in dims}
        count = 0
        for pd in pd_list:
            curves = betti_curves_from_pd(pd, dims, t_grid, min_persistence=min_persistence)
            for d in dims:
                sums[d] += curves[d]
            count += 1
        out[m] = {d: (sums[d] / max(count, 1)) for d in dims}
    return out


def plot_class_averaged_betti_curves(class_avg_curves, t_grid, modes, dims, out_dir):
    """One figure per dimension with all modes overlaid; also write per-dim CSV."""
    for d in dims:
        modes_with_data = [m for m in modes if m in class_avg_curves and d in class_avg_curves[m]]
        if not modes_with_data:
            continue
        # PDF figure
        plt.figure(figsize=(7.0, 4.0))
        for m in modes_with_data:
            plt.plot(t_grid, class_avg_curves[m][d], label=m, linewidth=2.0)
        plt.xlabel("filtration threshold")
        plt.ylabel(f"Betti_{d}(t)")
        plt.title(f"Class-averaged Betti curves (dim {d})")
        plt.grid(alpha=0.3, linestyle="--")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"dim_{d}_betti_curves.pdf"), format="pdf")
        plt.close()
        # CSV export
        out_csv = os.path.join(out_dir, f"dim_{d}_betti_curves.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t"] + modes_with_data)
            for i in range(len(t_grid)):
                w.writerow([t_grid[i]] + [class_avg_curves[m][d][i] for m in modes_with_data])


# --- AUBC plots & CSVs ---

def plot_violin_per_dim(auc_by_label, dims, out_dir):
    """Per-dimension violins with labels as dependents (AUBC)."""
    for d in dims:
        lbls = sorted(auc_by_label)
        if not any(len(auc_by_label[l][d]) for l in lbls):
            continue
        plt.figure(figsize=(8,4))
        plt.violinplot([auc_by_label[l][d] for l in lbls],
                       showmedians=True, showextrema=False)
        plt.xticks(range(1, len(lbls)+1), lbls, rotation=45, ha="right")
        plt.ylabel(f"AUC dim {d}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"dim_{d}_violin.pdf"), format="pdf")
        plt.close()


def plot_violin_per_label(auc_by_label, dims, out_dir):
    """Per-label violins across dims (AUBC)."""
    for lbl in sorted(auc_by_label):
        if not any(len(auc_by_label[lbl][d]) for d in dims):
            continue
        plt.figure(figsize=(8,4))
        plt.violinplot([auc_by_label[lbl][d] for d in dims],
                       showmedians=True, showextrema=False)
        plt.xticks(range(1, len(dims)+1), [str(d) for d in dims])
        plt.ylabel("Betti-AUC"); plt.xlabel("dimension"); plt.title(lbl)
        plt.tight_layout()
        safe = re.sub(r'[\\/*?:"<>| ]','_',lbl)
        plt.savefig(os.path.join(out_dir, f"{safe}_violin.pdf"), format="pdf")
        plt.close()


def plot_violin_per_mode(auc_by_mode, dims, modes, out_dir):
    """Per-mode violins across dims (AUBC)."""
    for m in modes:
        if not any(len(auc_by_mode[m][d]) for d in dims):
            continue
        plt.figure(figsize=(8,4))
        plt.violinplot([auc_by_mode[m][d] for d in dims],
                       showmedians=True, showextrema=False)
        plt.xticks(range(1, len(dims)+1), [str(d) for d in dims])
        plt.ylabel("Betti-AUC"); plt.xlabel("dimension")
        plt.title(f"Mode {m} fingerprint")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{m}_violin.pdf"), format="pdf")
        plt.close()


def plot_violin_per_mode_per_dim(auc_by_mode, dims, modes, out_dir):
    """
    For each dimension d, make one violin figure comparing modes (AUBC).
    Robust autoscaling using pooled quantiles (handles ~0 means with variance).
    """
    q_low, q_high = 0.01, 0.99
    pad_frac = 0.08
    min_range = 1e-6

    for d in dims:
        groups, labels = [], []
        for m in modes:
            vals = np.asarray([v for v in auc_by_mode[m][d] if np.isfinite(v)], dtype=float)
            if vals.size:
                groups.append(vals); labels.append(m)
        if not groups:
            continue

        all_vals = np.concatenate(groups)
        nonneg = np.all(all_vals >= 0)
        if all_vals.size >= 2:
            lo_q = float(np.quantile(all_vals, q_low))
            hi_q = float(np.quantile(all_vals, q_high))
        else:
            lo_q, hi_q = float(all_vals.min()), float(all_vals.max())

        y_min = 0.0 if nonneg else lo_q
        y_max = hi_q
        if not np.isfinite(y_min): y_min = float(np.nanmin(all_vals)) if np.isfinite(np.nanmin(all_vals)) else 0.0
        if not np.isfinite(y_max): y_max = float(np.nanmax(all_vals)) if np.isfinite(np.nanmax(all_vals)) else 1.0

        rng = y_max - y_min
        if rng < min_range:
            stats_lo = min(float(np.mean(g) - np.std(g)) for g in groups)
            stats_hi = max(float(np.mean(g) + np.std(g)) for g in groups)
            y_min = min(y_min, stats_lo, 0.0 if nonneg else stats_lo)
            y_max = max(y_max, stats_hi)
            rng = max(y_max - y_min, min_range)

        pad = pad_frac * rng
        y_min_adj = max(0.0, y_min - 0.25 * pad) if nonneg else (y_min - 0.25 * pad)
        y_max_adj = y_max + pad

        plt.figure(figsize=(6, 3.5))
        parts = plt.violinplot(groups, showmedians=True, showextrema=False, widths=0.7)
        for pc in parts.get('bodies', []):
            pc.set_edgecolor("black"); pc.set_alpha(0.9)
        if 'cmedians' in parts:
            parts['cmedians'].set_color("black"); parts['cmedians'].set_linewidth(1.2)

        plt.xticks(range(1, len(labels) + 1), labels)
        plt.ylabel(f"AUC dim {d}")
        plt.title(f"Betti-AUC by Mode (dim {d})")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.ylim(y_min_adj, y_max_adj)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"dim_{d}_mode_violin.pdf"), format="pdf")
        plt.close()


def write_auc_csvs(diag_root, auc_by_sample, auc_by_label, auc_by_mode, dims, modes):
    with open(os.path.join(diag_root, "betti_auc_by_sample.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","label",*[f"auc_dim_{d}" for d in dims]])
        for row in sorted(auc_by_sample, key=lambda r: r[0]):
            w.writerow(row)

    with open(os.path.join(diag_root, "label_fingerprint.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", *sum([[f"mean_{d}",f"std_{d}"] for d in dims],[])])
        for lbl in sorted(auc_by_label):
            vals_exist = any(auc_by_label[lbl][d] for d in dims)
            means = [np.mean(auc_by_label[lbl][d]) for d in dims] if vals_exist else [np.nan]*len(dims)
            stds  = [np.std (auc_by_label[lbl][d]) for d in dims] if vals_exist else [np.nan]*len(dims)
            w.writerow([lbl, *sum(zip(means,stds),())])

    with open(os.path.join(diag_root, "mode_fingerprint.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", *sum([[f"mean_{d}",f"std_{d}"] for d in dims],[])])
        for m in modes:
            vals_exist = any(auc_by_mode[m][d] for d in dims)
            means = [np.mean(auc_by_mode[m][d]) for d in dims] if vals_exist else [np.nan]*len(dims)
            stds  = [np.std (auc_by_mode[m][d]) for d in dims] if vals_exist else [np.nan]*len(dims)
            w.writerow([m, *sum(zip(means,stds),())])


# --- Wasserstein (AUBC distributions) ---

def compute_wasserstein_nd_aubc(auc_by_mode, dims, modes):
    """
    Per-dimension Wasserstein between AUBC distributions of modes; returns (Wdim, Wavg).
    """
    Wdim = np.zeros((len(dims), len(modes), len(modes)))
    for d_idx, d in enumerate(dims):
        for i, mi in enumerate(modes):
            xi = np.asarray(auc_by_mode[mi][d]) if len(auc_by_mode[mi][d]) else np.array([0.0])
            for j, mj in enumerate(modes):
                if i >= j: continue
                xj = np.asarray(auc_by_mode[mj][d]) if len(auc_by_mode[mj][d]) else np.array([0.0])
                Wdim[d_idx, i, j] = Wdim[d_idx, j, i] = wasserstein_distance(xi, xj)
    Wavg = Wdim.mean(axis=0) if len(dims) > 0 else np.zeros((len(modes), len(modes)))
    return Wdim, Wavg


def write_wasserstein_csvs(diag_root, modes, dims, Wdim, Wavg, suffix):
    out = os.path.join(diag_root, f"mode_nd_wasserstein_{suffix}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        for d_idx, d in enumerate(dims):
            w.writerow([f"dim_{d}"] + [""] * len(modes))
            w.writerow([""] + modes)
            for i, mi in enumerate(modes): w.writerow([mi, *Wdim[d_idx, i]])
            w.writerow([])
        w.writerow(["average"] + [""]); w.writerow([""] + modes)
        for i, mi in enumerate(modes): w.writerow([mi, *Wavg[i]])


def plot_wasserstein_heatmaps_nd(Wdim, Wavg, modes, dims, out_dir):
    # avg across dims
    plt.figure(figsize=(6,5))
    plt.imshow(Wavg, cmap="viridis")
    plt.colorbar(label="avg W₁")
    plt.xticks(range(len(modes)), modes, rotation=45)
    plt.yticks(range(len(modes)), modes)
    plt.title("Wasserstein (avg across dims)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "avg_wasserstein_heatmap.pdf"), format="pdf")
    plt.close()

    # per-dim
    for d_idx, d in enumerate(dims):
        plt.figure(figsize=(6,5))
        plt.imshow(Wdim[d_idx], cmap="viridis")
        plt.colorbar(label=f"W₁ dim {d}")
        plt.xticks(range(len(modes)), modes, rotation=45)
        plt.yticks(range(len(modes)), modes)
        plt.title(f"Wasserstein – Betti dim {d}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"dim_{d}_heatmap.pdf"), format="pdf")
        plt.close()


def plot_mds_maps(Wavg, Wdim, modes, dims, dist_dir, title_prefix):
    def _plot(D, title, out_pdf):
        coords = classical_mds(D)
        plt.figure(figsize=(5,4))
        for i in range(len(modes)):
            for j in range(i+1, len(modes)):
                plt.plot(*zip(coords[i], coords[j]),
                         linestyle="--", alpha=0.9, color="black", linewidth=1.8)
        for i, m in enumerate(modes):
            plt.scatter(*coords[i], s=700, zorder=3,
                        facecolors="#FFFFFF", edgecolors="black", linewidths=1.5)
            plt.text(coords[i][0], coords[i][1], m, ha="center", va="center",
                     fontsize=11, color="black", fontweight='bold', bbox=dict(alpha=0))
        plt.title(title, fontsize=11)
        plt.axis("equal"); plt.axis("off"); plt.tight_layout()
        plt.savefig(out_pdf, format="pdf"); plt.close()

    _plot(Wavg, f"{title_prefix} – avg", os.path.join(dist_dir, "avg_distance_map.pdf"))
    for di, d in enumerate(dims):
        _plot(Wdim[di], f"{title_prefix} – dim {d}",
              os.path.join(dist_dir, f"dim_{d}_distance_map.pdf"))


# --- Wasserstein between class-averaged Betti curves (BC) ---

def compute_wasserstein_nd_betti_curves(class_avg_curves, t_grid, dims, modes):
    """
    Per-dimension Wasserstein between class-averaged Betti curves:
    Uses scipy.stats.wasserstein_distance on support = t_grid with weights = curve values.
    Returns (Wdim, Wavg).
    """
    eps = 0.0  # set >0 if you prefer tiny smoothing to avoid zero-mass edge cases
    Wdim = np.zeros((len(dims), len(modes), len(modes)))
    for d_idx, d in enumerate(dims):
        # prefetch curves and masses
        curves = {}
        masses = {}
        for m in modes:
            c = class_avg_curves.get(m, {}).get(d, None)
            if c is None:
                c = np.zeros_like(t_grid)
            if eps > 0:
                c = c + eps
            curves[m] = c
            masses[m] = float(np.sum(c))
        for i, mi in enumerate(modes):
            for j, mj in enumerate(modes):
                if i >= j: continue
                ci, cj = curves[mi], curves[mj]
                mi_mass, mj_mass = masses[mi], masses[mj]
                if mi_mass == 0.0 and mj_mass == 0.0:
                    dist = 0.0
                elif mi_mass == 0.0 or mj_mass == 0.0:
                    # If only one is empty, define distance as 0 (conservative) or skip.
                    # Here we choose conservative 0.0 to avoid NaNs and keep plots readable.
                    dist = 0.0
                else:
                    dist = wasserstein_distance(t_grid, t_grid,
                                                u_weights=ci, v_weights=cj)
                Wdim[d_idx, i, j] = Wdim[d_idx, j, i] = dist
    Wavg = Wdim.mean(axis=0) if len(dims) > 0 else np.zeros((len(modes), len(modes)))
    return Wdim, Wavg


# --- Main pipeline ---

def main():
    args = parse_args()
    root, k, dims, modes, force = args.root, args.k, args.dims, args.modes, args.overwrite_existing
    betti_grid_points = int(args.betti_grid_points)
    min_persistence   = float(args.min_persistence)

    eval_dir = os.path.join(root, "evaluation")
    if not os.path.isdir(eval_dir):
        sys.exit("evaluation/ folder not found – run evaluation script first")

    paths  = ensure_dirs(root)
    id2lab = load_id2label(root)

    # containers
    auc_by_sample=[]
    auc_by_label={l:{d:[] for d in dims} for l in set(id2lab.values())}
    auc_by_mode ={m:{d:[] for d in dims} for m in modes}
    pds_by_mode = {m: [] for m in modes}
    global_min_birth, global_max_death = np.inf, -np.inf

    # iterate samples
    for td in sorted(d for d in os.listdir(eval_dir) if d.startswith("test_")):
        print("Processing:", td)
        idx   = int(re.findall(r'\d+', td)[0])
        label = id2lab[idx]
        act   = os.path.join(eval_dir, td, "activity")
        if not os.path.isdir(act): continue

        adj_p = os.path.join(eval_dir, td, f"adj_k-{k}.npy")
        pd_p  = os.path.join(eval_dir, td, f"pd_k-{k}.npy")
        auc_p = os.path.join(eval_dir, td, f"betti_auc_k-{k}.txt")

        adj = compute_or_load_adjacency(act, adj_p, k, force)
        pd  = compute_or_load_pd(adj, pd_p, dims, force)

        # global filtration range for Betti curves
        try:
            births = [float(b) for (b, de, dm) in pd if np.isfinite(b)]
            deaths = [float(de) for (b, de, dm) in pd if np.isfinite(de)]
            if births: global_min_birth = min(global_min_birth, min(births))
            if deaths: global_max_death = max(global_max_death, max(deaths))
        except Exception:
            pass

        # AUC (cached)
        if force or not os.path.exists(auc_p):
            auc_vals = betti_auc(pd, dims)
            with open(auc_p, "w") as f:
                [f.write(f"betti_auc_{d}: {v}\n") for d, v in auc_vals.items()]
        else:
            with open(auc_p) as f:
                auc_vals = {int(l.split(":")[0].split("_")[-1]): float(l.split(":")[1]) for l in f}

        # collect
        auc_by_sample.append([idx, label, *[auc_vals[d] for d in dims]])
        for d in dims: auc_by_label[label][d].append(auc_vals[d])
        for m in modes:
            if m in label:
                for d in dims: auc_by_mode[m][d].append(auc_vals[d])
                pds_by_mode[m].append(pd)

    # define Betti grid
    if not np.isfinite(global_min_birth): global_min_birth = 0.0
    if not np.isfinite(global_max_death) or global_max_death <= global_min_birth:
        global_max_death = global_min_birth + 1.0
    t_grid = np.linspace(global_min_birth, global_max_death, num=max(2, betti_grid_points))

    # class-averaged Betti curves (BC)
    class_avg_curves = aggregate_class_averaged_betti(
        pds_by_mode, dims, t_grid, min_persistence=min_persistence
    )
    plot_class_averaged_betti_curves(class_avg_curves, t_grid, modes, dims, paths["bc_betti_dir"])

    # AUBC-based plots & CSVs (figures go under AUBC/)
    plot_violin_per_dim(auc_by_label, dims, paths["aubc_per_dim_dir"])
    plot_violin_per_label(auc_by_label, dims, paths["aubc_per_lab_dir"])
    plot_violin_per_mode(auc_by_mode, dims, modes, paths["aubc_per_mode_dir"])
    plot_violin_per_mode_per_dim(auc_by_mode, dims, modes, paths["aubc_per_mode_dim_dir"])
    write_auc_csvs(paths["diag_root"], auc_by_sample, auc_by_label, auc_by_mode, dims, modes)

    # ND Wasserstein (AUBC distributions) → AUBC/heatmaps + AUBC/distances
    Wdim_AUBC, Wavg_AUBC = compute_wasserstein_nd_aubc(auc_by_mode, dims, modes)
    write_wasserstein_csvs(paths["diag_root"], modes, dims, Wdim_AUBC, Wavg_AUBC, suffix="AUBC")
    plot_wasserstein_heatmaps_nd(Wdim_AUBC, Wavg_AUBC, modes, dims, paths["aubc_heatND_dir"])
    plot_mds_maps(Wavg_AUBC, Wdim_AUBC, modes, dims, paths["aubc_dist_dir"],
                  title_prefix="AUBC Wasserstein map")

    # ND Wasserstein (between Betti curves) → BC/heatmaps + BC/distances
    Wdim_BC, Wavg_BC = compute_wasserstein_nd_betti_curves(class_avg_curves, t_grid, dims, modes)
    write_wasserstein_csvs(paths["diag_root"], modes, dims, Wdim_BC, Wavg_BC, suffix="BC")
    plot_wasserstein_heatmaps_nd(Wdim_BC, Wavg_BC, modes, dims, paths["bc_heatND_dir"])
    plot_mds_maps(Wavg_BC, Wdim_BC, modes, dims, paths["bc_dist_dir"],
                  title_prefix="Betti-curve Wasserstein map")

    print("All figures & CSVs saved →", paths["diag_root"])


if __name__ == "__main__":
    main()
