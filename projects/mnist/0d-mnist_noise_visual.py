#!/usr/bin/env python3
"""
Date: Feb 16, 2026
Description: Create simple visualizations of MNIST perturbation modes used in the experiments (bit-flip noise and pixel-moving variants).
Inputs:
  - Command-line arguments specifying source images, perturbation level, and output directory.
Outputs:
  - Saved example images illustrating the perturbation process.
"""

import os, torch, random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import argparse

# default noise levels; modify values if needed
NOISE_LEVELS = {
    "flip": {0: 0, 1: 25, 2: 50, 3: 75, 4: 100, 5: 125, 6: 150, 7: 175, 8: 200},
    "move": {0: 0, 1: 5,  2: 10, 3: 15, 4: 20, 5: 25,  6: 30,  7: 35,  8: 40},
}

# argument parser to override configuration
def parse_args():
    parser = argparse.ArgumentParser(description="Visualize MNIST noise effects")
    parser.add_argument("--root_dir", type=str, default="mnist_noise_visuals/bit-flip", help="Output root directory for images")
    parser.add_argument("--noise_type", type=str, choices=["flip", "move"], default="flip", help="Type of noise: flip or move")
    parser.add_argument("--seed", type=int, default=41, help="Random seed for reproducibility")
    return parser.parse_args()

args = parse_args()
ROOT_DIR   = args.root_dir
NOISE_TYPE = args.noise_type
SEED       = args.seed

  
def apply_pixel_flips(flat_img, n_flips):
    """Flip <n_flips> random pixels in a single flat image (in-place)."""
    if n_flips == 0:
        return flat_img
    P = flat_img.shape[0]
    idx = torch.randint(0, P, (n_flips,), device=flat_img.device)
    flat_img[idx] = 1.0 - flat_img[idx]
    return flat_img

def apply_pixel_moves(flat_img, n_moves):
    """Swap 'ink' pixels with background ones (in-place)."""
    if n_moves == 0:
        return flat_img
    fg_idx = torch.nonzero(flat_img > 0, as_tuple=False).flatten()
    bg_idx = torch.nonzero(flat_img == 0, as_tuple=False).flatten()
    if len(fg_idx) == 0 or len(bg_idx) == 0:
        return flat_img
    k = min(n_moves, len(fg_idx), len(bg_idx))
    f_sel = fg_idx[torch.randperm(len(fg_idx))[:k]]
    b_sel = bg_idx[torch.randperm(len(bg_idx))[:k]]
    fg_vals = flat_img[f_sel].clone()
    flat_img[f_sel] = 0.0
    flat_img[b_sel] = fg_vals
    return flat_img

  
def main():
    assert NOISE_TYPE in ["flip", "move"], "Invalid NOISE_TYPE"

    random.seed(SEED)
    torch.manual_seed(SEED)

    levels = NOISE_LEVELS[NOISE_TYPE]
    os.makedirs(ROOT_DIR, exist_ok=True)

    # Load MNIST
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=tf)

    # Group samples by digit
    digit_samples = {i: [] for i in range(10)}
    for i in range(len(test_ds)):
        img, label = test_ds[i]
        digit_samples[label].append(img)

    # Generate one image per digit per noise level
    for level, noise_amt in levels.items():
        level_dir = os.path.join(ROOT_DIR, f"{NOISE_TYPE}_level_{level}")
        os.makedirs(level_dir, exist_ok=True)

        for digit in range(10):
            if not digit_samples[digit]:
                continue

            img_flat = random.choice(digit_samples[digit]).clone()

            if NOISE_TYPE == "flip":
                noisy = apply_pixel_flips(img_flat.clone(), noise_amt)
            else:  # move
                noisy = apply_pixel_moves(img_flat.clone(), noise_amt)

            img_2d = noisy.view(28, 28).cpu().numpy()
            path = os.path.join(level_dir, f"digit_{digit}.png")
            plt.imsave(path, img_2d, cmap='gray')

    print(f"Images saved under: {ROOT_DIR} ({NOISE_TYPE} noise)")

if __name__ == "__main__":
    main()
