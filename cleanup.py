# cleanup.py
import argparse
import os
import shutil

# --------- CONFIG ---------
BASE_DIR = "/content/drive/MyDrive/sdc_msc_data_analytics_project"
EXPORT_DIR = os.path.join(BASE_DIR, "export_data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
VIDEO_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "video", "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
INFERENCE_LOGS_DIR = os.path.join(LOGS_DIR, "inference")

# --------- HELPERS ---------
def delete_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"🧹 Deleted: {path}")
    else:
        print(f"⚠️ Skipped (not found): {path}")

def delete_all_in_dir(path):
    if os.path.exists(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        print(f"🧼 Cleared: {path}")
    else:
        print(f"⚠️ Skipped (not found): {path}")

# --------- MAIN LOGIC ---------
def cleanup_all():
    delete_all_in_dir(EXPORT_DIR)
    delete_all_in_dir(PLOTS_DIR)
    delete_all_in_dir(VIDEO_OUTPUT_DIR)
    delete_all_in_dir(INFERENCE_LOGS_DIR)

    # Carefully delete only non-pose models
    if os.path.exists(MODELS_DIR):
        for file in os.listdir(MODELS_DIR):
            if "pose_landmarker" not in file:
                file_path = os.path.join(MODELS_DIR, file)
                os.remove(file_path)
                print(f"🧼 Deleted model: {file_path}")

def cleanup_exercise(category, exercise):
    export_path = os.path.join(EXPORT_DIR, category, exercise)
    plots_path = os.path.join(PLOTS_DIR, category, exercise)
    model_prefix = f"lstm_{category}_{exercise}"

    delete_folder(export_path)
    delete_folder(plots_path)

    # Delete matching models
    if os.path.exists(MODELS_DIR):
        for file in os.listdir(MODELS_DIR):
            if model_prefix in file and "pose_landmarker" not in file:
                file_path = os.path.join(MODELS_DIR, file)
                os.remove(file_path)
                print(f"🧼 Deleted model: {file_path}")

    # Delete inference logs matching the run prefix
    if os.path.exists(INFERENCE_LOGS_DIR):
        for folder in os.listdir(INFERENCE_LOGS_DIR):
            if folder.startswith(f"{category}_{exercise}") or folder.endswith(f"{exercise}"):
                folder_path = os.path.join(INFERENCE_LOGS_DIR, folder)
                delete_folder(folder_path)

# --------- ENTRY POINT ---------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up project directories")
    parser.add_argument("--mode", choices=["all", "exercise"], required=True,
                        help="Cleanup mode: all or exercise-specific")
    parser.add_argument("--category", type=str, help="Exercise category (e.g., physio, football)")
    parser.add_argument("--exercise", type=str, help="Exercise name (e.g., squat, penalty)")

    args = parser.parse_args()

    if args.mode == "all":
        cleanup_all()
    elif args.mode == "exercise":
        if not args.category or not args.exercise:
            print("❌ Please provide both --category and --exercise for exercise-specific cleanup.")
        else:
            cleanup_exercise(args.category.lower(), args.exercise.lower())
