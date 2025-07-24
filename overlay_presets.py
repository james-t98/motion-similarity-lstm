import cv2
from utils import calculate_angle
import numpy as np

def draw_angle(frame, angle, position, label, color=(0, 255, 255)):
    try:
        text = f"{label}: {angle:.1f}°"
        cv2.putText(frame, text, tuple(position.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    except Exception:
        pass

def overlay_squat(frame, coords, side="both"):
    sides = ["left", "right"] if side == "both" else [side]

    for s in sides:
        s_upper = s.upper()
        try:
            hip = coords[f"{s_upper}_HIP"]
            knee = coords[f"{s_upper}_KNEE"]
            ankle = coords[f"{s_upper}_ANKLE"]
            shoulder = coords[f"{s_upper}_SHOULDER"]

            hip_angle = calculate_angle(shoulder, hip, knee)
            knee_angle = calculate_angle(hip, knee, ankle)

            draw_angle(frame, hip_angle, hip, f"{s_upper} Hip")
            draw_angle(frame, knee_angle, knee, f"{s_upper} Knee")
        except KeyError:
            continue

def overlay_bench_press(frame, coords, side="both"):
    sides = ["left", "right"] if side == "both" else [side]

    for s in sides:
        s_upper = s.upper()
        try:
            shoulder = coords[f"{s_upper}_SHOULDER"]
            elbow = coords[f"{s_upper}_ELBOW"]
            wrist = coords[f"{s_upper}_WRIST"]

            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            draw_angle(frame, elbow_angle, elbow, f"{s_upper} Elbow")
        except KeyError:
            continue

def overlay_running(frame, coords, side="both"):
    sides = ["left", "right"] if side == "both" else [side]

    for s in sides:
        s_upper = s.upper()
        try:
            hip = coords[f"{s_upper}_HIP"]
            knee = coords[f"{s_upper}_KNEE"]
            ankle = coords[f"{s_upper}_ANKLE"]
            foot = coords[f"{s_upper}_FOOT_INDEX"]

            knee_angle = calculate_angle(hip, knee, ankle)
            ankle_angle = calculate_angle(knee, ankle, foot)

            draw_angle(frame, knee_angle, knee, f"{s_upper} Knee")
            draw_angle(frame, ankle_angle, ankle, f"{s_upper} Ankle")
        except KeyError:
            continue

def apply_overlay(frame, coords, exercise_type, side="both"):
    try:
        if exercise_type == "squat":
            overlay_squat(frame, coords, side=side)
        elif exercise_type == "bench_press":
            overlay_bench_press(frame, coords, side=side)
        elif exercise_type == "running":
            overlay_running(frame, coords, side=side)
    except Exception:
        pass
