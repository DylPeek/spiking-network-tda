"""
Date: Feb 16, 2026
Description: Interactive matplotlib visualizations for persistence diagrams, adjacency matrices, and signals with spike overlays.
Inputs:
  - Persistence diagram sequences and/or adjacency-matrix sequences.
  - Optional display parameters (titles, sampling rates, labels).
Outputs:
  - Interactive figures displayed on screen (no files written by default).
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

def visualize_persistence_diagrams(diagrams, title="Persistence Diagrams Over Time"):
    """
    Visualize a sequence of persistence diagrams with a slider for navigation.
    Each diagram shows the birth-death pairs for a specific time window, with different dimensions in different colors.

    Parameters:
        diagrams: list of np.ndarray
            A list where each element is an Nx3 array representing a persistence diagram.
            Each row is [birth, death, dimension].
        title: str
            Title for the visualization.
    """
    num_diagrams = len(diagrams)
    current_diagram = 0

    # Define colors for homology dimensions
    dim_colors = {0: "blue", 1: "orange", 2: "green", 3: "red"}  # Extend as needed for higher dimensions

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2)  # Space for the slider

    # Initialize scatter plots for each dimension
    scatter_plots = {}
    for dim, color in dim_colors.items():
        scatter_plots[dim] = ax.scatter([], [], label=f"H{dim}", color=color, alpha=0.7)

    ax.plot([0, 1], [0, 1], "k--", label="Diagonal", alpha=0.5)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(f"{title}\nWindow 1/{num_diagrams}")
    ax.legend()

    # Add a slider for navigating through diagrams
    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])  # Position of the slider
    slider = Slider(slider_ax, "Window", 1, num_diagrams, valinit=1, valstep=1)

    def update(diagram_idx):
        """Update the scatter plots when the slider is moved."""
        nonlocal current_diagram
        current_diagram = int(diagram_idx) - 1  # Adjust for zero-based index
        diagram = diagrams[current_diagram]

        # Update scatter plots for each dimension
        for dim, scatter_plot in scatter_plots.items():
            points = diagram[diagram[:, 2] == dim][:, :2]  # Filter by dimension
            scatter_plot.set_offsets(points)

        ax.set_title(f"{title}\nWindow {current_diagram + 1}/{num_diagrams}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Initialize with the first diagram
    if num_diagrams > 0:
        update(1)

    plt.show()



def visualize_adjacency_matrices(matrices, title="Adjacency Matrices Over Time"):
    """
    Visualize a sequence of adjacency matrices as a grid with a slider.
    Each matrix is displayed as an NxN heatmap, with colors representing values between 0 and 1.

    Parameters:
        matrices: np.ndarray
            A 3D array of shape (W, N, N), where W is the number of windows,
            and each matrix is of size NxN.
        title: str
            Title for the visualization.
    """
    if matrices.ndim != 3:
        raise ValueError("Expected a 3D array of shape (W, N, N) for matrices.")

    num_windows, matrix_size, _ = matrices.shape
    current_window = 0

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2, right=0.8)  # Space for slider and color bar

    # Display the first matrix as a heatmap
    img = ax.imshow(matrices[current_window], cmap="plasma", vmin=0, vmax=1)
    ax.set_title(f"{title}\nWindow 1/{num_windows}")
    ax.set_xlabel("Node Index")
    ax.set_ylabel("Node Index")

    # Add color bar
    cbar_ax = plt.axes([0.85, 0.2, 0.03, 0.6])  # Position of the color bar
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_label("TE Value (0-1)")

    # Add a slider for navigating through time windows
    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])  # Position of the slider
    slider = Slider(slider_ax, "Window", 1, num_windows, valinit=1, valstep=1)

    def update(window_idx):
        """Update the heatmap when the slider is moved."""
        nonlocal current_window
        current_window = int(window_idx) - 1  # Adjust for zero-based index
        img.set_data(matrices[current_window])
        ax.set_title(f"{title}\nWindow {current_window + 1}/{num_windows}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

def visualize_signals_with_spikes(time_series, discrete_signals, sampling_rate, title="Signal Visualization",
                                  x_label="Time (s)", y_label="Amplitude"):
    """
    Visualize raw or smoothed time-series signals with discrete spike signals overlaid as vertical strips.
    Includes a slider to navigate through multiple signals.

    Parameters:
        time_series: np.ndarray
            Array of shape (N, T), where N is the number of signals and T is the number of time steps.
        discrete_signals: np.ndarray
            Array of shape (N, T), where N is the number of signals and T is the number of time steps.
            Each row represents the discrete spike signal for the corresponding time-series signal.
        sampling_rate: float
            Sampling rate of the signals in Hz.
        title: str
            Title of the plot (default: "Signal Visualization").
        x_label: str
            Label for the x-axis (default: "Time (s)").
        y_label: str
            Label for the y-axis (default: "Amplitude").
    """
    if time_series.shape != discrete_signals.shape:
        raise ValueError("time_series and discrete_signals must have the same shape.")

    num_signals, num_steps = time_series.shape
    time_axis = np.arange(num_steps) / sampling_rate

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)  # Make space for the slider
    line, = ax.plot([], [], label="Time Series", color="blue")
    spike_lines = ax.vlines([], 0, 0, label="Spikes", color="red", alpha=0.6)

    # Set default axis properties
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    # Initialize the slider
    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])  # Position of the slider
    signal_slider = Slider(slider_ax, "Signal Index", 0, num_signals - 1, valinit=0, valstep=1)

    # Function to update the plot when the slider changes
    def update(index):
        index = int(index)  # Ensure index is an integer
        current_signal = time_series[index]
        current_spikes = discrete_signals[index]
        line.set_data(time_axis, current_signal)

        # Update spike lines
        spike_indices = np.where(current_spikes > 0)[0]
        spike_times = time_axis[spike_indices]
        spike_lines.set_segments([[[t, ax.get_ylim()[0]], [t, ax.get_ylim()[1]]] for t in spike_times])

        # Update plot limits
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_ylim(np.min(current_signal) - 0.1, np.max(current_signal) + 0.1)
        fig.canvas.draw_idle()

    # Initialize the plot with the first signal
    update(0)

    # Connect the slider to the update function
    signal_slider.on_changed(update)

    plt.show()


def visualize_betti_numbers(betti_numbers):
    time_windows = [row[0] for row in betti_numbers]
    num_dimensions = len(betti_numbers[0]) - 1

    plt.figure(figsize=(10,6))
    for dim in range(num_dimensions):
        betti_dim = [row[dim+1] for row in betti_numbers]
        plt.plot(time_windows, betti_dim, label=f'Betti_{dim}')
    plt.xlabel("Time Window Start")
    plt.ylabel("Betti Number")
    plt.title("Betti Numbers over Time")
    plt.legend()
    plt.grid(True)
    plt.show()
