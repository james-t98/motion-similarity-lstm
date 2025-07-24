# preprocessing.py
"""
Preprocessing module: loads angle CSVs, creates windowed sequences, normalizes data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_sequences_from_csv(csv_path, window_size=30, stride=5):
    """
    Loads angle data from CSV and creates overlapping sequences for training.

    Args:
        csv_path (str): Path to the CSV file.
        window_size (int): Number of time steps in each sequence.
        stride (int): Step size between windows.

    Returns:
        np.ndarray: Shape (N, T, F), where
                    N = number of sequences,
                    T = window_size,
                    F = number of features (angles).
    """
    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    data = df.values
    sequences = []

    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i:i + window_size]
        sequences.append(window)

    return np.array(sequences)


def normalize_sequences(X):
    """
    Normalizes sequences feature-wise using MinMax scaling.

    Args:
        X (np.ndarray): Shape (N, T, F)

    Returns:
        np.ndarray: Normalized version of X
    """
    if X.size == 0:
        raise ValueError("Input sequence array X is empty.")

    N, T, F = X.shape
    X_reshaped = X.reshape(-1, F)  # Shape: (N * T, F)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    return X_scaled.reshape(N, T, F)
