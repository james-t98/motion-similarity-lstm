"""
Shared utilities: constants, angle calculations, overlay helpers, color definitions, annotations.
"""
import sys
sys.path.append('/content/drive/MyDrive/sdc_msc_data_analytics_project')

import numpy as np
import cv2
import supervision as sv
from scipy.spatial.distance import cosine, euclidean
from fastdtw import fastdtw
from pathlib import Path
import os

LABELS = [
    "nose", "left eye inner", "left eye", "left eye outer",
    "right eye inner", "right eye", "right eye outer", "left ear",
    "right ear", "mouth left", "mouth right", "left shoulder",
    "right shoulder", "left elbow", "right elbow", "left wrist",
    "right wrist", "left pinky", "right pinky", "left index",
    "right index", "left thumb", "right thumb", "left hip",
    "right hip", "left knee", "right knee", "left ankle",
    "right ankle", "left heel", "right heel", "left foot index",
    "right foot index"
]

COLORS = [
    "#FF6347", "#FF6347", "#FF6347", "#FF6347",  # red-orange
    "#FF6347", "#FF1493", "#00FF00", "#FF1493",  # pink, green
    "#00FF00", "#FF1493", "#00FF00", "#FFD700",  # pink, green, gold
    "#00BFFF", "#FFD700", "#00BFFF", "#FFD700",  # blue, gold
    "#00BFFF", "#800080", "#ADFF2F", "#FF4500",  # purple, greenyellow, orange-red
    "#1E90FF", "#DA70D6", "#7FFF00", "#FF69B4",  # blue, orchid, chartreuse, pink
    "#8A2BE2", "#00CED1", "#DC143C", "#FF8C00",  # blue-violet, dark turquoise, crimson, dark orange
    "#32CD32", "#FF00FF", "#4169E1", "#FFB6C1",  # lime green, magenta, royal blue, light pink
    "#20B2AA"                                   # light sea green
]

COLORS = [sv.Color.from_hex(color_hex=c) for c in COLORS]

POSE_LANDMARKS = {
    "NOSE": 0,
    "LEFT_EYE_INNER": 1,
    "LEFT_EYE": 2,
    "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4,
    "RIGHT_EYE": 5,
    "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7,
    "RIGHT_EAR": 8,
    "MOUTH_LEFT": 9,
    "MOUTH_RIGHT": 10,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_PINKY": 17,
    "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19,
    "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21,
    "RIGHT_THUMB": 22,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29,
    "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32,
}

# --- Supervision annotators ---
vertex_annotator = sv.VertexAnnotator(color=sv.Color.GREEN, radius=2)
edge_annotator = sv.EdgeAnnotator(color=sv.Color.GREEN, thickness=2)
vertex_label_annotator = sv.VertexLabelAnnotator(
    color=COLORS, text_color=sv.Color.BLACK, border_radius=2
)

# --- Angle utility ---
def calculate_angle(a, b, c):
    """
    Calculate the angle at point b given points a-b-c.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

# --- Overlay helper ---
def overlay_squat_angles(frame, hip, knee, ankle, shoulder, knee_angle, hip_angle, color=(0, 255, 255)):
    """
    Draw angle text and limb lines for a given side (left/right).
    """
    try:
        cv2.putText(frame, f'Knee: {knee_angle:.1f}°', tuple(knee.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f'Hip: {hip_angle:.1f}°', tuple(hip.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.line(frame, tuple(hip.astype(int)), tuple(knee.astype(int)), color, 2)
        cv2.line(frame, tuple(knee.astype(int)), tuple(ankle.astype(int)), color, 2)
        cv2.line(frame, tuple(shoulder.astype(int)), tuple(hip.astype(int)), color, 2)
    except Exception:
        pass

def compute_similarity_to_reference(X, reference_window, method="cosine"):
    """
    Compute similarity between each window in X and the reference_window using the specified method.
    
    Parameters:
        X: np.ndarray of shape (n_samples, window_size, n_features)
        reference_window: np.ndarray of shape (window_size, n_features)
        method: str, one of ['cosine', 'euclidean', 'dtw']
    
    Returns:
        np.ndarray of similarity scores
    """
    similarities = []

    for window in X:
        if method == "cosine":
            sim = 1 - cosine(window.flatten(), reference_window.flatten())
        elif method == "euclidean":
            dist = euclidean(window.flatten(), reference_window.flatten())
            sim = 1 / (1 + dist)  # Normalize
        elif method == "dtw":
            dist, _ = fastdtw(window, reference_window)
            sim = 1 / (1 + dist)  # Normalize
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        similarities.append(sim)

    return np.array(similarities)

def get_output_paths(video_path, video_root="/content/drive/MyDrive/sdc_msc_data_analytics_project/data/video/input", output_root="/content/drive/MyDrive/sdc_msc_data_analytics_project/data/video/output", export_root="/content/drive/MyDrive/sdc_msc_data_analytics_project/export_data", plots_root="/content/drive/MyDrive/sdc_msc_data_analytics_project/plots"):
    """
    Returns output paths for a given input video:
    - Annotated video
    - Side-by-side comparison
    - CSV export
    - Plot paths (angle & similarity DTW)

    Ensures folder structure is mirrored.
    """
    rel_path = Path(video_path).relative_to(video_root)
    base_stem = rel_path.stem
    subdir = rel_path.parent

    # Ensure folders exist
    output_dir = Path(output_root) / subdir
    export_dir = Path(export_root) / subdir
    plot_dir = Path(plots_root) / subdir

    for d in [output_dir, export_dir, plot_dir]:
        os.makedirs(d, exist_ok=True)

    return {
        "annotated_video": output_dir / f"{base_stem}_annotated.mp4",
        "comparison_video": output_dir / f"{base_stem}_comparison.mp4",
        "csv_export": export_dir / f"{base_stem}_angles.csv",
        "dtw_angle": plot_dir / f"{base_stem}_dtw_angle.png",
        "dtw_similarity": plot_dir / f"{base_stem}_dtw_similarity.png"
    }