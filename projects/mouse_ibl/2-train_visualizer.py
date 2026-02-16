#!/usr/bin/env python3
"""
Date: Feb 16, 2026
Description: Interactive matplotlib viewer for extracted per-trial neuron spike trains.
Inputs:
  - Command-line arguments selecting a trial folder and the number of neurons to display per page.
Outputs:
  - Interactive on-screen plots (no files written).
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox


def parse_args() -> argparse.Namespace:
    """Parse arguments for the neuron spike visualizer."""
    parser = argparse.ArgumentParser(description="Visualize extracted neuron spike trains interactively")
    parser.add_argument("--trial_path", type=str, default="./ibl_data/ZM_2240/0/trial_000",
                        help="Path to the trial folder containing the 'neurons' subfolder")
    parser.add_argument("--neurons_per_page", type=int, default=5,
                        help="Number of neurons to display per page")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trial_path = args.trial_path
    neurons_path = os.path.join(trial_path, 'neurons')
    neurons_per_page = args.neurons_per_page

    # load all neuron spike arrays
    neuron_files = sorted([f for f in os.listdir(neurons_path) if f.endswith('.npy')])
    num_neurons = len(neuron_files)
    if num_neurons == 0:
        raise SystemExit(f"No neuron .npy files found in {neurons_path}")
    neuron_arrays = [np.load(os.path.join(neurons_path, f)) for f in neuron_files]
    trial_length = len(neuron_arrays[0])
    time_vector = np.arange(trial_length)

    # plotting state variables
    total_pages = int(np.ceil(num_neurons / neurons_per_page))
    current_page = 0
    start_idx = 0
    end_idx = trial_length

    # create figure and axes
    fig, axes = plt.subplots(neurons_per_page, 1, figsize=(12, 8), sharex=True)
    plt.subplots_adjust(bottom=0.25)

    def update_plot() -> None:
        """Update the current page of neurons and time window."""
        nonlocal current_page, start_idx, end_idx
        for i in range(neurons_per_page):
            ax = axes[i]
            ax.clear()
            neuron_idx = current_page * neurons_per_page + i
            if neuron_idx < num_neurons:
                spikes = neuron_arrays[neuron_idx][start_idx:end_idx]
                ax.plot(time_vector[start_idx:end_idx], spikes.astype(int), drawstyle='steps-pre')
                ax.set_ylabel(f'Neuron {neuron_idx:03d}')
                ax.set_yticks([0, 1])
            else:
                ax.set_visible(False)
        axes[-1].set_xlabel("Time (step index)")
        fig.canvas.draw_idle()

    # button callback closures
    def next_page(event) -> None:
        nonlocal current_page
        if current_page < total_pages - 1:
            current_page += 1
            update_plot()

    def prev_page(event) -> None:
        nonlocal current_page
        if current_page > 0:
            current_page -= 1
            update_plot()

    def update_time_range(_text: str | None = None) -> None:
        nonlocal start_idx, end_idx
        try:
            new_start = int(float(textbox_start.text))
            new_end = int(float(textbox_end.text))
            if 0 <= new_start < new_end <= trial_length:
                start_idx = new_start
                end_idx = new_end
                update_plot()
        except ValueError:
            pass

    # create UI elements
    axprev = plt.axes([0.1, 0.08, 0.1, 0.05])
    axnext = plt.axes([0.21, 0.08, 0.1, 0.05])
    bprev = Button(axprev, 'Back')
    bnext = Button(axnext, 'Next')
    bprev.on_clicked(prev_page)
    bnext.on_clicked(next_page)

    axbox_start = plt.axes([0.5, 0.08, 0.1, 0.05])
    axbox_end = plt.axes([0.62, 0.08, 0.1, 0.05])
    textbox_start = TextBox(axbox_start, 'Start', initial="0")
    textbox_end = TextBox(axbox_end, 'End', initial=str(trial_length))
    textbox_start.on_submit(update_time_range)
    textbox_end.on_submit(update_time_range)

    # initial plot
    update_plot()
    plt.show()


if __name__ == "__main__":
    main()
