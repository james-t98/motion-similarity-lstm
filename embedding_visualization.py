#!/usr/bin/env python3
"""
embedding_visualization.py
---------------------------
Generate 2D t-SNE and PCA plots of LSTM embeddings from trained model

Usage:
python embedding_visualization.py \
  --model_path /path/to/model.h5 \
  --data_root export_data \
  --category physio \
  --exercise squat
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

# ---------- CLI ARGUMENTS ----------
parser = argparse.ArgumentParser(description="Visualize LSTM embeddings with PCA and t-SNE")
parser.add_argument("--model_path", required=True, help="Path to trained LSTM model (.h5)")
parser.add_argument("--data_root", default="export_data", help="Root of export_data")
parser.add_argument("--category", required=True, help="Data category (e.g., physio)")
parser.add_argument("--exercise", required=True, help="Exercise (e.g., squat)")
args = parser.parse_args()

# ---------- Paths ----------
BASE_DIR = "/content/drive/MyDrive/sdc_msc_data_analytics_project"
EXPORT_DIR = os.path.join(BASE_DIR, args.data_root)
PLOTS_DIR = os.path.join(BASE_DIR, "plots", "embedding")
EXERCISE_DIR = os.path.join(EXPORT_DIR, args.category, args.exercise)
os.makedirs(PLOTS_DIR, exist_ok=True)

sys.path.append(BASE_DIR)
from preprocessing import load_sequences_from_csv, normalize_sequences
from utils import compute_similarity_to_reference

# ---------- Load Model ----------
print(f"✅ Loading model: {args.model_path}")
full_model = load_model(args.model_path)

# Define embedding model (first LSTM layer)
input_shape = full_model.input_shape[1:]  # (window_size, n_features)
inp = Input(shape=input_shape)
x = full_model.layers[0](inp)  # LSTM
embedding_model = Model(inputs=inp, outputs=x)

# ---------- Load Data ----------
print("📦 Loading sequences...")
csv_files = sorted([f for f in os.listdir(EXERCISE_DIR) if f.endswith("_angles.csv")])
X_all, y_all, labels = [], [], []

for file in csv_files:
    path = os.path.join(EXERCISE_DIR, file)
    X = normalize_sequences(load_sequences_from_csv(path, window_size=30, stride=5))
    sim_scores = compute_similarity_to_reference(X, X[0], method="cosine")

    X_all.append(X)
    y_all.extend(sim_scores)
    labels.extend([file.replace("_angles.csv", "")] * len(sim_scores))

X_all = np.concatenate(X_all)
y_all = np.array(y_all)
labels = np.array(labels)

print(f"✅ Loaded {len(csv_files)} files, {len(y_all)} windows total")

# ---------- Generate Embeddings ----------
print("🧠 Generating LSTM embeddings...")
embeddings = embedding_model.predict(X_all)

# ---------- Plotting ----------
def plot_embeddings(embeddings, values, method, save_name):
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, learning_rate=200)
    else:
        raise ValueError("Unknown method")

    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=values, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Cosine Similarity to Reference")
    plt.title(f"LSTM Embeddings ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    out_path = os.path.join(PLOTS_DIR, save_name)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"📸 Saved {method.upper()} plot: {out_path}")

plot_embeddings(embeddings, y_all, method="pca", save_name=f"pca_{args.category}_{args.exercise}.png")
plot_embeddings(embeddings, y_all, method="tsne", save_name=f"tsne_{args.category}_{args.exercise}.png")

print("\n✅ Embedding visualizations complete.")
