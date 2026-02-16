"""
Date: Feb 16, 2026
Description: Time-series preprocessing utilities used across experiments (trimming, smoothing, and spike detection).
Inputs:
  - Continuous or discrete time-series arrays and related parameters (time ranges, sampling rates, smoothing settings).
Outputs:
  - Processed arrays such as trimmed signals, smoothed signals, and detected spike timestamps.
"""

import numpy as np
import numpy as np
from scipy.ndimage import gaussian_filter1d
from oasis.functions import deconvolve


def trim_signal_to_time_range(signal, start_time, end_time, sampling_rate):
    """
    Trim a signal to a specific time range based on the start time, end time, and sampling rate.

    Parameters:
        signal: np.ndarray
            1D array representing the signal to be trimmed.
        start_time: float
            Start time in seconds.
        end_time: float
            End time in seconds.
        sampling_rate: float
            Sampling rate in Hz.

    Returns:
        np.ndarray: Trimmed signal.

    Raises:
        ValueError: If the signal is too short to accommodate the requested time range.
    """
    # Calculate the start and end indices
    start_index = int(start_time * sampling_rate)
    end_index = int(end_time * sampling_rate)

    # Check if the signal is long enough
    if end_index > len(signal):
        raise ValueError(
            f"Signal is too short for the requested time range. "
            f"Signal length: {len(signal)}, Requested end index: {end_index}."
        )

    # Return the trimmed signal
    return signal[start_index:end_index]

def trim_signals_to_time_range(signals, start_time, end_time, sampling_rate):
    """
    Trim an array of signals to a specific time range based on the start time, end time, and sampling rate.

    Parameters:
        signals: np.ndarray
            2D array of shape (N, T), where N is the number of signals and T is the time steps.
        start_time: float
            Start time in seconds.
        end_time: float
            End time in seconds.
        sampling_rate: float
            Sampling rate in Hz.

    Returns:
        np.ndarray: Trimmed signals of shape (N, trimmed_length), where
                    trimmed_length = (end_time - start_time) * sampling_rate.

    Raises:
        ValueError: If any signal is too short to accommodate the requested time range.
    """
    # Calculate the start and end indices
    start_index = int(start_time * sampling_rate)
    end_index = int(end_time * sampling_rate)

    # Ensure all signals meet the required length
    for idx, signal in enumerate(signals):
        if len(signal) < end_index:
            raise ValueError(
                f"Signal {idx} is too short for the requested time range. "
                f"Signal length: {len(signal)}, Requested end index: {end_index}."
            )

    # Trim all signals
    trimmed_signals = signals[:, start_index:end_index]
    return trimmed_signals


def smooth_signal(signals, sigma=1):
    """
    Apply Gaussian smoothing to each signal.

    Parameters:
        signals: np.ndarray
            Array of shape (N, T), where N is the number of signals and T is the number of time steps.
        sigma: float
            Standard deviation for Gaussian kernel.

    Returns:
        np.ndarray: Smoothed signals with the same shape as input.
    """
    return np.array([gaussian_filter1d(signal, sigma=sigma) for signal in signals])


from scipy.signal import find_peaks

def peak_detection(signals, height=None, prominence=None, distance=None):
    """
    Detect spikes based on signal peaks.

    Parameters:
        signals: np.ndarray
            Array of shape (N, T), where N is the number of signals and T is the number of time steps.
        height: float or tuple
            Minimum height of the peaks.
        prominence: float
            Required prominence of peaks.
        distance: float
            Minimum distance between consecutive peaks.

    Returns:
        list: A list of arrays containing spike timestamps for each signal.
    """
    spike_timestamps = []
    for signal in signals:
        peaks, _ = find_peaks(signal, height=height, prominence=prominence, distance=distance)
        spike_timestamps.append(peaks)
    return spike_timestamps

def ar2_spike_detection(signals, penalty=10):
    """
    Perform spike inference using the OASIS AR(2) model.

    Parameters:
        signals: np.ndarray
            Array of shape (N, T), where N is the number of smoothed signals and T is the number of time steps.
        penalty: float
            Penalty parameter for the deconvolution algorithm.

    Returns:
        list: A list of arrays containing spike timestamps for each signal.
    """
    spike_timestamps = []
    for signal in signals:
        _, inferred_spikes, _, _, _ = deconvolve(signal, g=(None, None), penalty=penalty)
        spike_times = np.where(inferred_spikes > 0)[0]
        spike_timestamps.append(spike_times)
    return spike_timestamps


def timestamps_to_discrete(timestamps, signal_length):
    """
    Convert spike timestamps into a discrete binary signal.

    Parameters:
        timestamps: list of arrays
            List of spike timestamps for each signal.
        signal_length: int
            Total length of the output discrete signal (number of time steps).
        sampling_rate: float
            Sampling rate in Hz.

    Returns:
        np.ndarray: Array of shape (N, signal_length), where N is the number of signals.
    """
    discrete_signals = []
    for spike_times in timestamps:
        discrete_signal = np.zeros(signal_length, dtype=float)
        for spike_time in spike_times:
            idx = int(spike_time)
            if 0 <= idx < signal_length:
                discrete_signal[idx] = 1.0
        discrete_signals.append(discrete_signal)
    return np.array(discrete_signals)


def discrete_to_timestamps(discrete_signals, sampling_rate):
    """
    Convert a discrete binary signal into spike timestamps.

    Parameters:
        discrete_signals: np.ndarray
            Array of shape (N, T), where N is the number of signals and T is the number of time steps.
        sampling_rate: float
            Sampling rate in Hz.

    Returns:
        list: A list of arrays containing spike timestamps for each signal.
    """
    spike_timestamps = []
    for discrete_signal in discrete_signals:
        spike_indices = np.where(discrete_signal > 0)[0]
        spike_times = spike_indices / sampling_rate
        spike_timestamps.append(spike_times)
    return spike_timestamps


def normalize_signals(signals):
    """
    Scale each signal in signals to the [0, 1] range.

    signals: np.array of shape (N, T)
    where N is the number of signals and T is the number of time points.
    """
    scaled_signals = []
    for sig in signals:
        min_val = np.min(sig)
        max_val = np.max(sig)
        if max_val > min_val:
            ssig = (sig - min_val) / (max_val - min_val)
        else:
            # If the signal is constant, all values are the same.
            # In this case, we can just produce an array of zeros (all points equal min_val).
            ssig = np.zeros_like(sig)
        baseline = np.median(ssig)
        ssig -= baseline
        scaled_signals.append(ssig)
    return np.array(scaled_signals)


def extract_time_windows_centered(signals, total_length, window_length, step_length):
    """
    signals: np.array of shape (N, T)
    total_length: number of samples (e.g. signals.shape[1])
    window_length: length of each window in samples
    step_length: step size in samples

    Returns: list of tuples (midpoint, window_data) where window_data has shape (N, window_length).
    """
    half_w = window_length // 2
    windows = []
    for midpoint in range(0, total_length, step_length):
        # Calculate the nominal start and end
        wstart = midpoint - half_w
        wend = wstart + window_length  # or midpoint + half_w

        # Clamp to valid indices
        valid_start = max(wstart, 0)
        valid_end = min(wend, total_length)

        # Prepare a zero-padded window
        window_data = np.zeros((signals.shape[0], window_length), dtype=signals.dtype)

        # Extract the data we actually have
        subarray = signals[:, valid_start:valid_end]

        # Figure out where subarray fits inside window_data
        # If wstart < 0, we start filling from some positive index in window_data
        start_idx_in_window = valid_start - wstart  # shift for negative wstart
        window_data[:, start_idx_in_window:start_idx_in_window + subarray.shape[1]] = subarray

        # Append the result: use midpoint as the new "identifier"
        windows.append((midpoint, window_data))

    return windows
