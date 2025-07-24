# main.py

import sys
import os
import argparse
import numpy as np
from datetime import datetime
sys.path.append('/content/drive/MyDrive/sdc_msc_data_analytics_project')

from utils import compute_similarity_to_reference, get_output_paths
from pose_module import extract_angles_from_video
from data_export import export_angle_data
from preprocessing import load_sequences_from_csv, normalize_sequences
from model_module import train_lstm_regressor, extract_embeddings
from visualization import visualize_dtw_alignment
from video_utils import create_side_by_side_video

def parse_args():
    parser = argparse.ArgumentParser(description="Run motion analysis pipeline")
    parser.add_argument("--video_name", required=True, help="Name of the video file (without extension)")
    parser.add_argument("--exercise_type", required=True, help="Exercise type (e.g., squat, bench_press)")
    parser.add_argument("--category", required=True, help="Dataset category (e.g., Physio, Football)")
    parser.add_argument("--annotate", action="store_true", help="Whether to save annotated video")
    parser.add_argument("--save_side_by_side", action="store_true", help="Whether to save side-by-side comparison video")
    parser.add_argument("--run_id", default=None, help="Custom run ID (default = timestamp)")
    return parser.parse_args()

def run_pipeline_for_video(video_path, exercise_type="squat"):
    paths = get_output_paths(video_path)
    video_name = paths["annotated_video"].stem.replace("_annotated", "")

    # Step 1: Extract angles and save annotated video
    angles, times = extract_angles_from_video(
        video_path,
        str(paths["annotated_video"]),
        exercise_type=exercise_type,
        annotate=True
    )

    if angles.shape[0] == 0:
        print(f"[⚠️] No valid angles found in: {video_path}")
        return

    export_angle_data(video_name, angles, times, str(paths["csv_export"].parent))

    # Step 2: Load CSV → Window + Normalize
    X = load_sequences_from_csv(str(paths["csv_export"]), window_size=30, stride=5)
    X = normalize_sequences(X)

    if X.shape[0] == 0:
        print(f"[⚠️] No valid windows found in: {paths['csv_export']}")
        return

    reference = X[0]
    y_sim = compute_similarity_to_reference(X, reference, method="cosine")

    # Step 3: Train Model
    model, metrics = train_lstm_regressor(X, y_sim)
    print(f"✅ Trained model on {video_name}. Metrics: {metrics}")

    # Step 4: Extract Embeddings & Visualize
    X_embed = extract_embeddings(model, X)

    visualize_dtw_alignment(
        X[0][:, 0],
        X[min(5, len(X)-1)][:, 0],
        mode="angle",
        label_a="Ref",
        label_b="Sample",
        save_path=str(paths["dtw_angle"])
    )

    y_pred = model.predict(X).flatten()
    visualize_dtw_alignment(
        y_sim, y_pred,
        mode="similarity",
        label_a="True",
        label_b="Pred",
        save_path=str(paths["dtw_similarity"])
    )

    # Step 5: Side-by-side
    create_side_by_side_video(
        video_path,
        str(paths["annotated_video"]),
        str(paths["comparison_video"]),
        label_a="Reference",
        label_b="Athlete",
        annotate=True,
        gap_width=30
    )

def main():
    args = parse_args()

    # Setup paths
    base_dir = "/content/drive/MyDrive/sdc_msc_data_analytics_project"
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    input_video = f"{base_dir}/data/video/input/{args.category}/{args.exercise_type}/{args.video_name}.mp4"
    run_dir = f"{base_dir}/runs/{args.category}/{args.exercise_type}/{run_id}"

    output_paths = {
        "annotated_video": f"{run_dir}/annotated/{args.video_name}.mp4",
        "side_by_side": f"{run_dir}/side_by_side/{args.video_name}.mp4",
        "export_data": f"{run_dir}/export_data",
        "plots": f"{run_dir}/plots"
    }

    os.makedirs(output_paths["export_data"], exist_ok=True)
    os.makedirs(output_paths["plots"], exist_ok=True)
    if args.annotate:
        os.makedirs(os.path.dirname(output_paths["annotated_video"]), exist_ok=True)
    if args.save_side_by_side:
        os.makedirs(os.path.dirname(output_paths["side_by_side"]), exist_ok=True)

    # Phase 1: Extract and annotate
    angles, times = extract_angles_from_video(
        input_video,
        output_paths["annotated_video"] if args.annotate else None,
        exercise_type=args.exercise_type,
        annotate=args.annotate
    )
    export_angle_data(args.video_name, angles, times, output_paths["export_data"])
    print("✅ Phase 1 complete")

    # Phase 2: Load + Preprocess
    csv_path = f"{output_paths['export_data']}/{args.video_name}_angles.csv"
    X = load_sequences_from_csv(csv_path, window_size=30, stride=5)
    X = normalize_sequences(X)

    if len(X) == 0:
        raise ValueError("No valid sequences found.")

    reference = X[0]
    y_sim = compute_similarity_to_reference(X, reference, method="cosine")
    print(f"✅ Phase 2 complete: {X.shape} sequences")

    # Phase 3: Train + Embeddings
    model, metrics = train_lstm_regressor(X, y_sim)
    print("✅ Model trained. Metrics:", metrics)
    X_embed = extract_embeddings(model, X)
    print("✅ Phase 3 complete: Embeddings extracted")


    # Phase 4: Visualizations
    visualize_dtw_alignment(
        X[0][:, 0],
        X[5][:, 0],
        mode="angle",
        label_a="Reference",
        label_b="Sample",
        save_path=f"{output_paths['plots']}/dtw_angle_alignment.png"
    )
    print("✅ Phase 4 complete: DTW angle alignment visualized")

    y_pred = model.predict(X).flatten()
    visualize_dtw_alignment(
        y_sim,
        y_pred,
        mode="similarity",
        label_a="True",
        label_b="Predicted",
        save_path=f"{output_paths['plots']}/dtw_similarity_alignment.png"
    )

    # Phase 5: Video Comparison
    if args.save_side_by_side:
        create_side_by_side_video(
            input_video,
            output_paths["annotated_video"],
            output_paths["side_by_side"],
            label_a="Reference",
            label_b="Athlete",
            annotate=True,
            gap_width=30
        )
        print("✅ Phase 5 complete: Side-by-side video created")


if __name__ == "__main__":
    main()
    # run_pipeline_for_video(
    #     video_path="/content/drive/MyDrive/sdc_msc_data_analytics_project/data/video/input/Physio/Squat/v_BodyWeightSquats_g02_c04.mp4",
    #     exercise_type="squat"  # Change per category
    # )
