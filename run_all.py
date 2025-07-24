""#!/usr/bin/env python3
"""
run_all.py: Full pipeline execution script for AI-based movement analysis

Usage Examples:
----------------
1. Run full pipeline on 5 samples (quick mode, default):
   python run_all.py --category physio --exercise squat

2. Run on all available videos:
   python run_all.py --category physio --exercise squat --full_run

3. Run without saving annotated/side-by-side videos:
   python run_all.py --category physio --exercise squat --no_annotate

4. Use a custom reference video:
   python run_all.py --category physio --exercise squat --reference_strategy custom:v_SampleVideo_g01_c01_angles.csv

5. Set a custom model name:
   python run_all.py --category physio --exercise squat --model_name squat_model_v2

6. Use a specific similarity method:
   python run_all.py --category physio --exercise squat --similarity_method cosine
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.append("/content/drive/MyDrive/sdc_msc_data_analytics_project")

from batch_pose_extraction import batch_pose_extraction
from data_loader import load_and_preprocess_angles
from train_model import train_model_from_data
from inference import run_inference_from_model

# -------- CLI ARGUMENT PARSER -------- #
parser = argparse.ArgumentParser(description="Run full motion analysis pipeline")
parser.add_argument("--category", required=True, help="Dataset category (e.g., physio, football)")
parser.add_argument("--exercise", required=True, help="Exercise name (e.g., squat, bench_press)")
parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to run in quick mode")
parser.add_argument("--full_run", action="store_true", help="If set, process all videos in batch")
parser.add_argument("--no_annotate", action="store_true", help="Disable annotation and side-by-side video saving")
parser.add_argument("--reference_strategy", default="first", help="Reference video selection (first, random, custom:<filename>)")
parser.add_argument("--model_name", default=None, help="Optional model filename override")
parser.add_argument("--similarity_method", default="cosine", help="Similarity method: cosine, euclidean, dtw")
args = parser.parse_args()

# -------- GLOBAL CONFIG -------- #
BASE_DIR = "/content/drive/MyDrive/sdc_msc_data_analytics_project"
ANNOTATE = not args.no_annotate
N_SAMPLES = None if args.full_run else args.n_samples

# -------- RUN PIPELINE -------- #
print("\n🟢 Step 1: Pose Extraction")
batch_pose_extraction(
    category=args.category,
    exercise=args.exercise,
    input_root=os.path.join(BASE_DIR, "data", "video", "input"),
    output_root=os.path.join(BASE_DIR, "data", "video", "output"),
    export_root=os.path.join(BASE_DIR, "export_data"),
    save_n_annotated=5 if ANNOTATE else 0,
    sample_limit=N_SAMPLES
)

print("\n🟢 Step 2: Data Preprocessing")
data = load_and_preprocess_angles(
    export_root=os.path.join(BASE_DIR, "export_data"),
    category=args.category,
    exercise=args.exercise,
    window_size=30,
    stride=5
)

print("\n🟢 Step 3: Model Training")
model, run_id = train_model_from_data(
    data,
    category=args.category,
    exercise=args.exercise,
    save_root=os.path.join(BASE_DIR, "models"),
    plots_root=os.path.join(BASE_DIR, "plots", "train"),
    custom_model_name=args.model_name
)

print("\n🟢 Step 4: Inference and Evaluation")
run_inference_from_model(
    model_path=os.path.join(BASE_DIR, "models", f"{args.model_name or f'lstm_{args.category}_{args.exercise}_{run_id}'}.h5"),
    export_root=os.path.join(BASE_DIR, "export_data"),
    plots_root=os.path.join(BASE_DIR, "plots", "inference"),
    category=args.category,
    exercise=args.exercise,
    strategy=args.reference_strategy,
    similarity_method=args.similarity_method,
    log_dir=os.path.join(BASE_DIR, "logs", "inference"),
    run_id=run_id
)

print("\n✅ Pipeline execution complete.")
