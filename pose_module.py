"""
Handles pose estimation using MediaPipe, angle calculation, and annotated video export.
"""
import sys
import cv2
import numpy as np
import mediapipe as mp
import supervision as sv
from tqdm import tqdm

sys.path.append('/content/drive/MyDrive/sdc_msc_data_analytics_project')

from utils import calculate_angle, vertex_annotator, edge_annotator, vertex_label_annotator, POSE_LANDMARKS, COLORS, LABELS
from pose_config import ANGLE_CONFIG
from overlay_presets import apply_overlay

PoseLandmarker = mp.tasks.vision.PoseLandmarker

options_video = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="/content/drive/MyDrive/sdc_msc_data_analytics_project/models/pose_landmarker_heavy.task"),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_poses=2
)

def extract_angles_from_video(video_path, output_path, exercise_type="squat", annotate=True):
    angle_definitions = ANGLE_CONFIG[exercise_type]

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if annotate:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    all_angles = []
    timestamps = []

    with PoseLandmarker.create_from_options(options_video) as landmarker:
        for frame_idx in tqdm(range(frame_count), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int((frame_idx / fps) * 1000)
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_frame, timestamp_ms)
            keypoints = sv.KeyPoints.from_mediapipe(result, (frame_width, frame_height))
            coords = keypoints.xy[0] if len(keypoints.xy) else None

            frame_angles = []
            annotated_frame = frame.copy()

            if coords is not None:
                try:
                    coords_dict = {k: coords[v] for k, v in POSE_LANDMARKS.items() if v < len(coords)}
                    # 1️⃣ Apply overlay
                    if annotate:
                        apply_overlay(annotated_frame, coords_dict, exercise_type, side="both")

                    # 2️⃣ 🔧 Collect angles for all sides and joints
                    for side in angle_definitions["sides"]:
                        for _, points in angle_definitions["angles"].items():
                            try:
                                a = coords_dict[f"{side.upper()}_{points[0]}"]
                                b = coords_dict[f"{side.upper()}_{points[1]}"]
                                c = coords_dict[f"{side.upper()}_{points[2]}"]
                                angle = calculate_angle(a, b, c)
                                frame_angles.append(angle)

                            except KeyError as e:
                                print(f"[⚠️] Missing keypoint: {e} — skipping angle.")

                except Exception as e:
                    print(f"[⚠️] Skipping frame due to error: {e}")
                    continue


            all_angles.append(frame_angles)
            timestamps.append(frame_idx / fps)

            if annotate and writer:
                annotated_frame = vertex_annotator.annotate(scene=annotated_frame, key_points=keypoints)
                annotated_frame = edge_annotator.annotate(scene=annotated_frame, key_points=keypoints)
                annotated_frame = vertex_label_annotator.annotate(scene=annotated_frame, key_points=keypoints, labels=LABELS)
                writer.write(annotated_frame)

    cap.release()
    if annotate and writer:
        writer.release()

    print(f"[✓] Angles extracted. Annotated video saved to: {output_path}" if annotate else "[✓] Angles extracted.")

    valid_indices = [i for i, a in enumerate(all_angles) if None not in a and len(a) > 0]
    angles_valid = [all_angles[i] for i in valid_indices]
    time_valid = [timestamps[i] for i in valid_indices]

    return np.array(angles_valid), np.array(time_valid)
