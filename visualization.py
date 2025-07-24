# visualization.py
import os
import matplotlib.pyplot as plt
from dtaidistance import dtw

def visualize_dtw_alignment(
    sequence_a,
    sequence_b,
    mode="angle",
    label_a="Reference",
    label_b="Target",
    title=None,
    save_path=None
):
    """
    Visualizes DTW alignment between two 1D sequences and optionally saves the figure.

    Args:
        sequence_a (np.ndarray): First sequence (e.g., reference).
        sequence_b (np.ndarray): Second sequence (e.g., predicted or target).
        mode (str): "angle" or "similarity"
        label_a (str): Label for sequence A in the legend.
        label_b (str): Label for sequence B in the legend.
        title (str): Custom title for the plot.
        save_path (str): Path to save the output figure (PNG).
    """
    distance, paths = dtw.warping_paths(sequence_a, sequence_b)
    best_path = dtw.best_path(paths)

    plt.figure(figsize=(10, 5))
    plt.plot(sequence_a, label=label_a, color="blue", marker='o' if mode=="similarity" else None)
    plt.plot(sequence_b, label=label_b, color="orange", marker='x' if mode=="similarity" else None)

    for (i, j) in best_path:
        plt.plot([i, j], [sequence_a[i], sequence_b[j]], 'gray', alpha=0.3)

    plt.title(title or f"DTW Alignment | Distance = {distance:.3f}")
    plt.xlabel("Timestep" if mode == "angle" else "Sequence Index")
    plt.ylabel("Angle (°)" if mode == "angle" else "Similarity Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[📸] DTW alignment plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()
