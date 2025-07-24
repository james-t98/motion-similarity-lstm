# batch_pose_extraction.py

import os
import sys
import random
from datetime import datetime

sys.path.append("/content/drive/MyDrive/sdc_msc_data_analytics_project")

from pose_module import extract_angles_from_video
from data_export import export_angle_data
from video_utils import create_side_by_side_video

def batch_pose_extraction(
    category,
    exercise,
    input_root,
    output_root,
    export_root,
    save_n_annotated=5,
    sample_limit=None,
    generate_side_by_side=True
):
    input_dir = os.path.join(input_root, category, exercise)
    output_annotated_dir = os.path.join(output_root, "annotated", category, exercise)
    output_comparison_dir = os.path.join(output_root, "comparison", category, exercise)
    export_dir = os.path.join(export_root, category, exercise)

    os.makedirs(output_annotated_dir, exist_ok=True)
    os.makedirs(output_comparison_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    video_paths = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".mp4")
    ]
    random.shuffle(video_paths)

    if sample_limit:
        video_paths = video_paths[:sample_limit]

    for i, video_path in enumerate(video_paths):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"📹 Processing video: {video_name}")

        # Annotate only N videos
        annotate = i < save_n_annotated

        # Output paths
        annotated_path = os.path.join(output_annotated_dir, f"{video_name}.mp4")

        # Pose + angles
        angles, timestamps = extract_angles_from_video(
            video_path=video_path,
            output_path=annotated_path,
            exercise_type=exercise.lower(),
            annotate=annotate
        )

        # Export angle CSV
        export_angle_data(video_name, angles, timestamps, export_dir)

        # Optional comparison view
        if annotate and generate_side_by_side:
            comparison_path = os.path.join(output_comparison_dir, f"{video_name}_compare.mp4")
            create_side_by_side_video(
                video_path_a=video_path,
                video_path_b=annotated_path,
                output_path=comparison_path,
                label_a="Original",
                label_b="Annotated",
                annotate=True
            )

    print(f"\n✅ Batch pose extraction complete. Processed {len(video_paths)} video(s).\n")


# Optional CLI support
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True, help="e.g., physio or football")
    parser.add_argument("--exercise", type=str, required=True, help="e.g., squat or penalty")
    parser.add_argument("--input_root", type=str, default="/content/drive/MyDrive/sdc_msc_data_analytics_project/data/video/input")
    parser.add_argument("--output_root", type=str, default="/content/drive/MyDrive/sdc_msc_data_analytics_project/data/video/output")
    parser.add_argument("--export_root", type=str, default="/content/drive/MyDrive/sdc_msc_data_analytics_project/export_data")
    parser.add_argument("--save_n_annotated", type=int, default=5, help="Max annotated videos to save")
    parser.add_argument("--sample_limit", type=int, default=None, help="Max videos to process")
    parser.add_argument("--no_side_by_side", action="store_true", help="Disable side-by-side generation")

    args = parser.parse_args()

    batch_pose_extraction(
        category=args.category,
        exercise=args.exercise,
        input_root=args.input_root,
        output_root=args.output_root,
        export_root=args.export_root,
        save_n_annotated=args.save_n_annotated,
        sample_limit=args.sample_limit,
        generate_side_by_side=not args.no_side_by_side
    )
