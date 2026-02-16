#!/usr/bin/env python3
"""
Date: Feb 16, 2026
Description: Visualize TE-derived adjacency matrices from logic-gate evaluations as graphs and edge-weight histograms.
Inputs:
  - Experiment root containing evaluation/test_*/adj_k-<k>.npy files.
  - Command-line arguments selecting k and display controls (edge fraction, linewidth scaling, optional input raster).
Outputs:
  - Per-sample graph and histogram images saved under <root>/diagrams/evaluation/te_graphs/.
"""

import argparse, os, re, sys, glob, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ----------------------------- CLI -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="and_or_xor_2bit-20_N-64_P_HIGH-5_P_LOW-0_2",
                   help="experiment root folder containing evaluation/")
    p.add_argument("--k", type=int, default=10, help="k used in adj_k-<k>.npy filenames")

    # Edge display: keep edges with weight <= quantile(edge_frac) among valid edges.
    # Convention: lower weight = stronger (sublevel filtration).
    p.add_argument("--edge_frac", type=float, default=0.60,
                   help="quantile cutoff q; keep edges with weight <= q (lower=stronger)")
    p.add_argument("--max_edges", type=int, default=1200,
                   help="cap number of displayed edges (strongest kept). 0 disables cap.")

    # Styling
    p.add_argument("--lw_min", type=float, default=0.20)
    p.add_argument("--lw_max", type=float, default=5.25)
    p.add_argument("--edge_alpha_min", type=float, default=0.10)
    p.add_argument("--edge_alpha_max", type=float, default=0.85)
    p.add_argument("--edge_color", default="tab:blue")

    p.add_argument("--node_size_in", type=float, default=190)
    p.add_argument("--node_size_hid", type=float, default=85)

    p.add_argument("--font_size", type=float, default=9.0)
    p.add_argument("--label_size", type=float, default=9.0)
    p.add_argument("--title_size", type=float, default=11.0)

    p.add_argument("--show_hidden_labels", action="store_true",
                   help="label hidden neurons (can clutter); default off")

    # Layout
    p.add_argument("--grid_cols", type=int, default=0,
                   help="hidden grid columns (0 => auto sqrt)")
    p.add_argument("--group_gap", type=float, default=0.65,
                   help="extra vertical gap between bit inputs and gate selectors")
    p.add_argument("--x_pad", type=float, default=2.2,
                   help="horizontal spacing between input column and hidden grid")

    p.add_argument("--input_label_dx", type=float, default=0.42,
                   help="how far left (in data units) to place input text labels from input squares")

    # Raster panel
    p.add_argument("--no_input_raster", action="store_true",
                   help="disable input spike raster panel")
    p.add_argument("--raster_width_ratio", type=float, default=1.0,
                   help="relative width of raster panel vs graph panel")

    p.add_argument("--raster_tick_height", type=float, default=0.28,
                   help="max half-height of each raster tick (auto-scaled down if needed)")
    p.add_argument("--raster_lw", type=float, default=2.0,
                   help="linewidth of raster ticks")
    p.add_argument("--raster_box_lw", type=float, default=1.6,
                   help="linewidth for per-input raster row boxes")
    p.add_argument("--raster_box_alpha", type=float, default=1.0)
    p.add_argument("--input_file_hint", default="data.npy",
                   help="preferred input file name (fallback searches other candidates)")

    # Histogram controls
    p.add_argument("--bins", type=int, default=80)
    p.add_argument("--alpha_bar", type=float, default=0.85)
    p.add_argument("--hist_color", default="0.25")
    p.add_argument("--hist_show_threshold", action="store_true")

    # Weight handling
    p.add_argument("--negate", action="store_true",
                   help="multiply adjacency by -1 before processing")
    p.add_argument("--hist_in_te_units", action="store_true",
                   help="plot histogram in TE units (TE := -weight) if your filtration weight is -TE")

    # Output
    p.add_argument("--fmt", choices=["png", "pdf"], default="pdf")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--overwrite", action="store_true")

    # Layout fine-tuning: reduces the gap between raster and graph subplots
    p.add_argument("--panel_wspace", type=float, default=0.08,
                   help="horizontal whitespace between raster and graph panels (smaller = tighter)")
    return p.parse_args()


# ----------------------------- Utilities -----------------------------

def safe_name(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]+', "_", s.strip())


def load_id2label(root: str):
    path = os.path.join(root, "evaluation_accuracy_log.csv")
    if not os.path.isfile(path):
        return {}
    import csv
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                out[int(r["id"])] = r.get("label", "")
            except Exception:
                continue
    return out


def infer_n_hidden(act_dir: str) -> int:
    files = glob.glob(os.path.join(act_dir, "*.npy"))
    return len([f for f in files if re.search(r"[/\\]n\d+\.npy$", f)])


def parse_logic_label(label: str):
    if not label:
        return None, None
    parts = [p for p in label.strip().split("_") if p != ""]
    if len(parts) < 2:
        return None, None
    gate = parts[-1].strip().upper()
    bit_parts = parts[:-1]
    if all(re.fullmatch(r"[01]", bp) for bp in bit_parts):
        bits = "".join(bit_parts)
        return bits, gate
    return None, gate


def logic_input_labels(n_in: int, gate_names=("AND", "OR", "XOR")):
    if n_in <= 0:
        return []
    if n_in < 3:
        return [f"x{i}" for i in range(n_in)]
    gate_count = 3
    bit_count = n_in - gate_count
    bits = [f"x{i}" for i in range(max(bit_count, 0))]
    gates = [gate_names[i] if i < len(gate_names) else f"g{i}" for i in range(gate_count)]
    return bits + gates


def coerce_to_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    a = np.squeeze(a)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(1, -1)
    if a.ndim == 2:
        return a
    while a.ndim > 2:
        a = np.squeeze(a[0])
        if a.ndim <= 2:
            break
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a


def load_input_spikes(act_dir: str, file_hint: str = "data.npy"):
    if not os.path.isdir(act_dir):
        return None

    files = sorted(glob.glob(os.path.join(act_dir, "*.npy")))
    if not files:
        return None

    hint_path = os.path.join(act_dir, file_hint)
    candidates = []
    if os.path.isfile(hint_path):
        candidates.append(hint_path)

    for f in files:
        base = os.path.basename(f).lower()
        if re.search(r"[/\\]n\d+\.npy$", f):
            continue
        if "in" in base or "input" in base:
            candidates.append(f)

    data_path = os.path.join(act_dir, "data.npy")
    if os.path.isfile(data_path) and data_path not in candidates:
        candidates.append(data_path)

    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    for path in candidates:
        try:
            arr = np.load(path, allow_pickle=False)
            arr = coerce_to_2d(arr)

            # transpose heuristic
            if arr.shape[0] > arr.shape[1] and arr.shape[1] <= 32:
                arr = arr.T

            arr = (arr > 0).astype(int)
            if arr.shape[1] < 4:
                continue
            return arr
        except Exception:
            continue

    return None


# ----------------------------- Layout -----------------------------

def hidden_grid_positions(n_hidden: int, grid_cols: int = 0):
    if grid_cols and grid_cols > 0:
        cols = int(grid_cols)
    else:
        cols = max(1, int(math.ceil(math.sqrt(max(n_hidden, 1)))))
    rows = max(1, int(math.ceil(n_hidden / cols)))

    dx, dy = 1.0, 1.0
    pos = []
    for h in range(n_hidden):
        r, c = divmod(h, cols)
        x = (c - (cols - 1) / 2.0) * dx
        y = ((rows - 1) / 2.0 - r) * dy
        pos.append((x, y))

    y_top = max(y for _, y in pos) if pos else 0.0
    y_bot = min(y for _, y in pos) if pos else 0.0
    x_left = min(x for x, _ in pos) if pos else -1.0
    x_right = max(x for x, _ in pos) if pos else 1.0

    return pos, (x_left, x_right, y_top, y_bot)


def input_positions(n_in: int, y_top: float, y_bot: float, x_in: float, group_gap: float):
    if n_in <= 0:
        return []

    y_center = 0.5 * (y_top + y_bot)
    H = float(y_top - y_bot) if np.isfinite(y_top) and np.isfinite(y_bot) else 6.0
    H = max(H, 2.0)

    if n_in >= 3:
        gate_count = 3
        bit_count = n_in - gate_count

        usable = 0.90 * H
        denom = max(n_in - 1, 1)
        step = (usable - (group_gap if bit_count > 0 else 0.0)) / denom
        step = float(np.clip(step, 0.60, 1.80))

        total = (n_in - 1) * step + (group_gap if bit_count > 0 else 0.0)
        y = y_center + 0.5 * total
        ys = []

        for _ in range(max(bit_count, 0)):
            ys.append(y)
            y -= step

        if bit_count > 0:
            y -= group_gap

        for _ in range(gate_count):
            ys.append(y)
            y -= step

        ys = ys[:n_in]
        return [(x_in, yy) for yy in ys]

    y_vals = np.linspace(y_center + 0.45 * H, y_center - 0.45 * H, num=n_in) if n_in > 1 else np.array([y_center])
    return [(x_in, float(y)) for y in y_vals]


# ----------------------------- Edge extraction -----------------------------

def extract_edges(adj: np.ndarray, edge_frac: float, max_edges: int):
    A = adj.astype(float).copy()
    np.fill_diagonal(A, np.nan)
    valid = np.isfinite(A) & (A != 0.0)
    if not np.any(valid):
        return [], np.nan

    w_all = A[valid]
    thresh = float(np.quantile(w_all, edge_frac))

    keep = valid & (A <= thresh)
    src, dst = np.where(keep)
    w = A[src, dst]

    order = np.argsort(w)  # smallest first (strongest)
    if max_edges and max_edges > 0 and order.size > max_edges:
        order = order[:max_edges]

    edges = [(int(src[i]), int(dst[i]), float(w[i])) for i in order]
    return edges, thresh


def _median_input_spacing(input_positions_xy):
    if not input_positions_xy or len(input_positions_xy) < 2:
        return 1.0
    ys = sorted([y for (_, y) in input_positions_xy], reverse=True)
    diffs = [abs(ys[i] - ys[i + 1]) for i in range(len(ys) - 1)]
    return float(np.median(diffs)) if diffs else 1.0


# ----------------------------- Drawing -----------------------------

def draw_graph_with_raster(adj_plot: np.ndarray,
                           n_in: int, n_hidden: int,
                           input_spikes: np.ndarray,
                           input_labels,
                           pos_in, pos_hid,
                           lw_min, lw_max,
                           alpha_min, alpha_max,
                           edge_color,
                           node_size_in, node_size_hid,
                           font_size, label_size, title_size,
                           show_hidden_labels,
                           edge_frac, max_edges,
                           main_title: str,
                           raster_title: str,
                           graph_title: str,
                           show_raster: bool,
                           raster_width_ratio: float,
                           raster_tick_height: float,
                           raster_lw: float,
                           raster_box_lw: float,
                           raster_box_alpha: float,
                           input_label_dx: float,
                           out_path: str,
                           panel_wspace: float,
                           fmt="png", dpi=220):

    edges, thresh = extract_edges(adj_plot, edge_frac=edge_frac, max_edges=max_edges)
    if not edges:
        print("  [!] no valid edges (nonzero finite) – skip graph")
        return

    w_vals = np.array([w for (_, _, w) in edges], dtype=float)
    w_min = float(np.min(w_vals))
    denom = (thresh - w_min) if np.isfinite(thresh) else (float(np.max(w_vals)) - w_min)
    denom = float(denom) if abs(denom) > 1e-12 else 1.0

    def strength_score(w):
        s = (thresh - w) / denom  # smaller w => larger score
        return float(np.clip(s, 0.0, 1.0))

    positions = pos_in + pos_hid
    labels = input_labels + [f"h{i}" for i in range(n_hidden)]

    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])

    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": title_size,
        "axes.labelsize": font_size,
    })

    if show_raster:
        fig = plt.figure(figsize=(10.4, 5.3))
        gs = fig.add_gridspec(
            1, 2,
            width_ratios=[max(0.25, raster_width_ratio), 3.0],
            wspace=max(0.0, panel_wspace)
        )
        ax = fig.add_subplot(gs[0, 1])
        ax_r = fig.add_subplot(gs[0, 0], sharey=ax)
    else:
        fig = plt.figure(figsize=(9.2, 5.3))
        ax = fig.add_subplot(1, 1, 1)
        ax_r = None

    # ---------------- Graph panel ----------------
    ax.axis("off")

    # IMPORTANT: prevent "inside-axis" whitespace that cannot be cropped
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("W")  # left align within its gridspec cell

    ax.set_title(graph_title, pad=8.0)

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    # Keep pads modest; avoid inflating x-range unnecessarily
    x_pad_left = 1.00 + float(input_label_dx)
    x_pad_right = 0.85
    y_pad = 0.95

    ax.set_xlim(x_min - x_pad_left, x_max + x_pad_right)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Edges behind nodes
    for (s, d, w) in edges:
        (x0, y0) = positions[s]
        (x1, y1) = positions[d]
        sscore = strength_score(w)

        lw = lw_min + (lw_max - lw_min) * sscore
        a = alpha_min + (alpha_max - alpha_min) * sscore

        ax.plot([x0, x1], [y0, y1],
                linewidth=lw, alpha=a, color=edge_color, zorder=1)

    # Nodes
    idx_in = list(range(0, n_in))
    idx_h = list(range(n_in, n_in + n_hidden))

    if idx_in:
        ax.scatter(xs[idx_in], ys[idx_in],
                   s=node_size_in, marker="s",
                   facecolors="white", edgecolors="black",
                   linewidths=0.95, zorder=3)
    if idx_h:
        ax.scatter(xs[idx_h], ys[idx_h],
                   s=node_size_hid, marker="o",
                   facecolors="white", edgecolors="black",
                   linewidths=0.90, zorder=3)

    # Labels
    dy_in = _median_input_spacing(pos_in)
    label_dy = -0.04 * dy_in

    for i in idx_in:
        x, y = positions[i]
        ax.text(
            x - float(input_label_dx),
            y + label_dy,
            labels[i],
            ha="right",
            va="center",
            fontsize=label_size,
            zorder=4,
            clip_on=False
        )

    if show_hidden_labels:
        for j, i in enumerate(idx_h):
            x, y = positions[i]
            ax.text(x, y, f"h{j}", ha="center", va="center",
                    fontsize=label_size * 0.70, zorder=4)

    # ---------------- Raster panel ----------------
    if show_raster and ax_r is not None:
        ax_r.set_aspect("auto")
        ax_r.set_facecolor("white")
        ax_r.set_title(raster_title, pad=8.0)

        for sp in ax_r.spines.values():
            sp.set_visible(False)

        if input_spikes is not None and input_spikes.size > 0:
            inp = coerce_to_2d(input_spikes)
            if inp.shape[0] != n_in and inp.shape[1] == n_in:
                inp = inp.T

            if inp.shape[0] != n_in:
                c = min(inp.shape[0], n_in)
                tmp = np.zeros((n_in, inp.shape[1]), dtype=int)
                tmp[:c] = (inp[:c] > 0).astype(int)
                inp = tmp
            else:
                inp = (inp > 0).astype(int)

            T = int(inp.shape[1])
            ax_r.set_xlim(-0.5, T - 0.5)

            dy_med = max(_median_input_spacing(pos_in), 1e-6)
            tick_h = min(float(raster_tick_height), 0.22 * dy_med)
            box_half = max(tick_h * 1.35, 0.30 * dy_med)

            for ch in range(n_in):
                y = positions[ch][1]

                rect = Rectangle(
                    (-0.5, y - box_half),
                    width=T,
                    height=2.0 * box_half,
                    fill=False,
                    linewidth=raster_box_lw,
                    edgecolor="0.2",
                    alpha=raster_box_alpha,
                    zorder=1
                )
                ax_r.add_patch(rect)

                ts = np.where(inp[ch] > 0)[0]
                if ts.size:
                    ax_r.vlines(
                        ts,
                        y - tick_h,
                        y + tick_h,
                        color="black",
                        linewidth=raster_lw,
                        alpha=0.95,
                        zorder=2
                    )

            ax_r.set_yticks([])
            ax_r.set_xticks([])
            ax_r.set_xlabel("")

            # Solid time axis line (no arrow) + "t"
            y_ax = 0.04
            ax_r.plot([0.08, 0.92], [y_ax, y_ax],
                      transform=ax_r.transAxes, color="0.2", lw=1.0, clip_on=False)
            ax_r.text(0.94, y_ax, "t", transform=ax_r.transAxes,
                      ha="left", va="center", color="0.2", fontsize=label_size)
        else:
            ax_r.axis("off")

    # Layout: reserve top space for suptitle
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])

    # Center the suptitle over the used axes span (raster + graph), not the full figure.
    # This prevents the title looking "off-center" if layout leaves blank margins.
    if show_raster and ax_r is not None:
        b0 = ax_r.get_position()
        b1 = ax.get_position()
        x0 = min(b0.x0, b1.x0)
        x1 = max(b0.x1, b1.x1)
    else:
        b1 = ax.get_position()
        x0, x1 = b1.x0, b1.x1
    fig.suptitle(main_title, x=0.5 * (x0 + x1), y=0.985, fontsize=title_size + 0.5)

    # Save (tight bbox crops any OUTSIDE-axes whitespace)
    if fmt == "png":
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    else:
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def draw_hist(adj_plot: np.ndarray,
              bins: int,
              hist_color: str,
              alpha_bar: float,
              edge_frac: float,
              show_threshold: bool,
              hist_in_te_units: bool,
              title: str,
              out_path: str,
              fmt="png", dpi=220):

    A = adj_plot.astype(float).copy()
    np.fill_diagonal(A, np.nan)
    valid = np.isfinite(A) & (A != 0.0)
    vals = A[valid]
    if vals.size == 0:
        print("  [!] no valid edges (nonzero finite) – skip hist")
        return

    thresh = float(np.quantile(vals, edge_frac))

    if hist_in_te_units:
        x = -vals
        x_thresh = -thresh
        xlabel = "Transfer entropy (higher = stronger)"
    else:
        x = vals
        x_thresh = thresh
        xlabel = "Filtration weight (lower = stronger)"

    plt.rcParams.update({
        "font.size": 9.0,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.0,
    })

    fig = plt.figure(figsize=(6.2, 3.6))
    ax = plt.gca()

    ax.hist(x, bins=bins, color=hist_color, alpha=alpha_bar)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")

    if show_threshold:
        ax.axvline(x_thresh, linestyle="--", linewidth=1.2, color="0.2", alpha=0.9)

    ax.set_title(title)

    plt.tight_layout()
    if fmt == "png":
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    else:
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ----------------------------- Main -----------------------------

def main():
    a = parse_args()
    eval_dir = os.path.join(a.root, "evaluation")
    if not os.path.isdir(eval_dir):
        sys.exit("evaluation/ not found")

    out_dir = os.path.join(a.root, "diagrams", "evaluation", "te_graphs")
    os.makedirs(out_dir, exist_ok=True)

    id2label = load_id2label(a.root)

    tests = sorted(d for d in os.listdir(eval_dir) if d.startswith("test_"))
    if not tests:
        sys.exit("no test folders")

    for td in tests:
        adj_p = os.path.join(eval_dir, td, f"adj_k-{a.k}.npy")
        if not os.path.isfile(adj_p):
            continue

        m = re.findall(r"\d+", td)
        tid = int(m[0]) if m else -1
        lbl = id2label.get(tid, "")
        lbl_safe = safe_name(lbl) if lbl else "unlabeled"

        bits, gate = parse_logic_label(lbl)
        if bits is not None and gate is not None:
            nice = f"x = {bits}, {gate}"
        elif gate is not None and lbl:
            nice = f"{gate}"
        else:
            nice = lbl if lbl else td

        main_title = f"Transfer Entropy Graph ({nice})"
        raster_title = "Input spikes"
        graph_title = "Spiking Network Neurons"

        g_out = os.path.join(out_dir, f"{td}_{lbl_safe}_graph.{a.fmt}")
        h_out = os.path.join(out_dir, f"{td}_{lbl_safe}_hist.{a.fmt}")
        if (not a.overwrite) and os.path.isfile(g_out) and os.path.isfile(h_out):
            continue

        adj = np.load(adj_p).astype(float)
        if a.negate:
            adj = -adj

        act_dir = os.path.join(eval_dir, td, "activity")
        inp = load_input_spikes(act_dir, file_hint=a.input_file_hint)

        n_hidden_files = infer_n_hidden(act_dir) if os.path.isdir(act_dir) else 0
        n_total = int(adj.shape[0])

        if inp is not None:
            inp2 = coerce_to_2d(inp)
            if inp2.shape[0] > inp2.shape[1] and inp2.shape[1] <= 32:
                inp2 = inp2.T
            n_in = int(inp2.shape[0])
            inp = inp2
        else:
            n_in = (len(bits) + 3) if bits is not None else 0

        n_hidden = n_hidden_files if n_hidden_files > 0 else max(n_total - n_in, 0)

        # Drop output neuron if present
        use_slice = False
        if n_in > 0 and n_total == (n_in + n_hidden):
            keep = list(range(n_in + n_hidden))
            use_slice = True
        elif n_in > 0 and n_total == (n_in + n_hidden + 1):
            keep = list(range(n_in + n_hidden))
            use_slice = True
        elif n_total == n_hidden:
            use_slice = False
        elif n_total == (n_hidden + 1):
            adj = adj[:n_hidden, :n_hidden]
            use_slice = False
        else:
            n_hidden = n_total
            use_slice = False

        if use_slice:
            adj_plot = adj[np.ix_(keep, keep)]
            n_hidden_plot = adj_plot.shape[0] - n_in
            if n_hidden_plot >= 0:
                n_hidden = n_hidden_plot
            else:
                n_in = 0
                n_hidden = adj_plot.shape[0]
        else:
            if n_in > 0:
                adj_plot = np.zeros((n_in + n_hidden, n_in + n_hidden), dtype=float)
                H = adj[:n_hidden, :n_hidden]
                adj_plot[n_in:, n_in:] = H
            else:
                adj_plot = adj[:n_hidden, :n_hidden]

        hid_pos, (x_left, x_right, y_top, y_bot) = hidden_grid_positions(n_hidden, grid_cols=a.grid_cols)
        x_in = x_left - a.x_pad
        in_pos = input_positions(n_in, y_top, y_bot, x_in, group_gap=a.group_gap)
        input_labels = logic_input_labels(n_in)

        draw_graph_with_raster(
            adj_plot=adj_plot,
            n_in=n_in, n_hidden=n_hidden,
            input_spikes=inp,
            input_labels=input_labels,
            pos_in=in_pos, pos_hid=hid_pos,
            lw_min=a.lw_min, lw_max=a.lw_max,
            alpha_min=a.edge_alpha_min, alpha_max=a.edge_alpha_max,
            edge_color=a.edge_color,
            node_size_in=a.node_size_in, node_size_hid=a.node_size_hid,
            font_size=a.font_size, label_size=a.label_size, title_size=a.title_size,
            show_hidden_labels=a.show_hidden_labels,
            edge_frac=a.edge_frac, max_edges=a.max_edges,
            main_title=main_title,
            raster_title=raster_title,
            graph_title=graph_title,
            show_raster=(not a.no_input_raster),
            raster_width_ratio=a.raster_width_ratio,
            raster_tick_height=a.raster_tick_height,
            raster_lw=a.raster_lw,
            raster_box_lw=a.raster_box_lw,
            raster_box_alpha=a.raster_box_alpha,
            input_label_dx=a.input_label_dx,
            out_path=g_out,
            panel_wspace=a.panel_wspace,
            fmt=a.fmt, dpi=a.dpi
        )

        hist_title = f"TE edge-weight distribution ({nice})"
        draw_hist(
            adj_plot=adj_plot,
            bins=a.bins,
            hist_color=a.hist_color,
            alpha_bar=a.alpha_bar,
            edge_frac=a.edge_frac,
            show_threshold=a.hist_show_threshold,
            hist_in_te_units=a.hist_in_te_units,
            title=hist_title,
            out_path=h_out,
            fmt=a.fmt, dpi=a.dpi
        )

        print("Saved", td, "→", os.path.basename(g_out), os.path.basename(h_out))


if __name__ == "__main__":
    main()
