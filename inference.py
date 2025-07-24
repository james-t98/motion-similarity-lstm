# inference.py

import os
import sys
import csv
import json
import random
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score

# Project config
BASE_DIR = "/content/drive/MyDrive/sdc_msc_data_analytics_project"
EXPORT_DIR = os.path.join(BASE_DIR, "export_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_ROOT = os.path.join(BASE_DIR, "plots", "inference")
LOG_ROOT = os.path.join(BASE_DIR, "logs", "inference")

CATEGORY = "physio"
EXERCISE = "squat"
WINDOW_SIZE = 30
STRIDE = 5
N_TEST_VIDEOS = 3
SIMILARITY_METHOD = "cosine"  # or "mse", "dtw"

sys.path.append(BASE_DIR)
from preprocessing import load_sequences_from_csv, normalize_sequences
from utils import compute_similarity_to_reference
from visualization import visualize_dtw_alignment

def load_video_sequences(csv_path):
    X = load_sequences_from_csv(csv_path, window_size=WINDOW_SIZE, stride=STRIDE)
    return normalize_sequences(X)

def find_latest_model(model_dir, category, exercise):
    files = [f for f in os.listdir(model_dir) if f.startswith(f"lstm_{category}_{exercise}")]
    if not files:
        raise FileNotFoundError("No model found.")
    return os.path.join(model_dir, sorted(files)[-1])

def log_inference_results(run_id, method, results, mse, r2, log_root):
    log_dir = os.path.join(log_root, run_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"inference_{method}.csv")

    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["video_name", "frame_idx", "true_score", "predicted_score"])
        for video_name, values in results.items():
            for i, (true_val, pred_val) in enumerate(zip(values["y_true"], values["y_pred"])):
                writer.writerow([video_name, i, true_val, pred_val])

        writer.writerow([])
        writer.writerow(["summary"])
        writer.writerow(["mse", "r2"])
        writer.writerow([mse, r2])

    print(f"📁 Logged results to: {log_file}")

    # --- JSON log ---
    json_log = {
        "run_id": run_id,
        "method": method,
        "mse": mse,
        "r2_score": r2,
        "test_files": list(results.keys())
    }

    json_path = os.path.join(log_dir, f"inference_{method}.json")
    with open(json_path, "w") as jf:
        json.dump(json_log, jf, indent=2)

    print(f"📄 JSON log saved: {json_path}")

def run_inference_from_model(
    model_path,
    export_root,
    plots_root,
    category,
    exercise,
    strategy="first",
    similarity_method="cosine",
    log_dir=LOG_ROOT,
    run_id=None
):
    print(f"✅ Loading model from: {model_path}")
    model = load_model(model_path)

    run_id = run_id or os.path.splitext(os.path.basename(model_path))[0].replace("lstm_", "")
    exercise_dir = os.path.join(export_root, category, exercise)
    os.makedirs(os.path.join(plots_root, run_id), exist_ok=True)

    csv_files = sorted([f for f in os.listdir(exercise_dir) if f.endswith("_angles.csv")])

    # Select reference file
    if strategy.startswith("custom:"):
        ref_file = strategy.split("custom:")[1]
    elif strategy == "random":
        ref_file = random.choice(csv_files)
    else:
        ref_file = csv_files[0]

    test_files = [f for f in csv_files if f != ref_file]
    random.shuffle(test_files)
    test_files = test_files[:N_TEST_VIDEOS]

    print(f"\n🧠 Reference: {ref_file}")
    print(f"🧪 Test files: {test_files}")

    reference_window = load_video_sequences(os.path.join(exercise_dir, ref_file))[0]

    results = {}
    all_true = []
    all_pred = []

    for test_file in test_files:
        X_test = load_video_sequences(os.path.join(exercise_dir, test_file))
        y_true = compute_similarity_to_reference(X_test, reference_window, method=similarity_method)
        y_pred = model.predict(X_test).flatten()

        base_name = test_file.replace("_angles.csv", "")
        results[base_name] = {"y_true": y_true, "y_pred": y_pred}

        all_true.extend(y_true)
        all_pred.extend(y_pred)

        plot_path = os.path.join(plots_root, run_id, f"sim_vs_pred_{base_name}.png")
        visualize_dtw_alignment(
            y_true,
            y_pred,
            mode="similarity",
            label_a="True Similarity",
            label_b="Predicted Similarity",
            save_path=plot_path
        )
        print(f"✅ Saved similarity plot: {plot_path}")

    mse = mean_squared_error(all_true, all_pred)
    r2 = r2_score(all_true, all_pred)

    log_inference_results(run_id, similarity_method, results, mse, r2, log_dir)

# For direct testing (optional)
if __name__ == "__main__":
    model_path = find_latest_model(MODEL_DIR, CATEGORY, EXERCISE)
    run_inference_from_model(
        model_path=model_path,
        export_root=EXPORT_DIR,
        plots_root=PLOT_ROOT,
        category=CATEGORY,
        exercise=EXERCISE,
        strategy="first",
        similarity_method=SIMILARITY_METHOD
    )

