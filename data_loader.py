import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_angles(
    export_root,
    category,
    exercise,
    window_size=30,
    stride=5,
    test_size=0.2,
    val_size=0.1,
    random_state=42
):
    """
    Loads and processes all angle CSVs for a given category and exercise.
    Applies windowing, normalization, and splits into train/val/test sets.

    Returns: dict with X splits and optional metadata
    """
    folder = os.path.join(export_root, category, exercise)
    all_files = [f for f in os.listdir(folder) if f.endswith("_angles.csv")]

    all_sequences = []
    video_names = []

    for file in all_files:
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path)

        if df.shape[0] < window_size:
            continue  # Skip too short sequences

        angle_data = df.drop(columns=["timestamp"], errors="ignore").values
        
        for i in range(0, len(angle_data) - window_size + 1, stride):
            window = angle_data[i:i+window_size]
            all_sequences.append(window)
            video_names.append(file.replace("_angles.csv", ""))

    if len(all_sequences) == 0:
        raise ValueError("No valid sequences found. Check input files.")

    X = np.array(all_sequences)
    print(f"✅ Loaded {X.shape[0]} sequences from {len(all_files)} files")

    # Normalize across all timepoints and features
    N, T, F = X.shape
    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, F)
    X_scaled = scaler.fit_transform(X_reshaped).reshape(N, T, F)

    # Train/Val/Test split
    X_temp, X_test = train_test_split(X_scaled, test_size=test_size, random_state=random_state)
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val = train_test_split(X_temp, test_size=val_relative_size, random_state=random_state)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "meta": {
            "video_names": video_names,
            "scaler": scaler,
            "angle_names": df.columns.drop("timestamp").tolist() if "timestamp" in df.columns else df.columns.tolist()
        }
    }