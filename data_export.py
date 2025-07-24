# data_export.py
"""
Exports computed joint angles and timestamps to CSV.
"""

import os
import pandas as pd

def export_angle_data(video_name, angles, timestamps, export_path):
    """
    Saves angle data and timestamps into a CSV file.

    Args:
        video_name (str): Name of the video (used in filename).
        angles (np.ndarray): Array of shape (N, F), angles per frame.
        timestamps (np.ndarray): Array of shape (N,), timestamps in seconds.
        export_path (str): Path to export the CSV.
    """
    os.makedirs(export_path, exist_ok=True)

    # Combine timestamps and angles
    df = pd.DataFrame(angles)
    df.insert(0, "timestamp", timestamps)

    file_path = os.path.join(export_path, f"{video_name}_angles.csv")
    df.to_csv(file_path, index=False)

    print(f"[📁] Exported angle data to: {file_path}")
