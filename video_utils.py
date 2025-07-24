# video_utils.py
import os
import cv2

import os
import cv2
import numpy as np

def create_side_by_side_video(
    video_path_a,
    video_path_b,
    output_path,
    label_a="Reference",
    label_b="Target",
    annotate=True,
    resize_height=480,
    gap_width=20  
):
    """
    Creates and saves a side-by-side video comparing two input videos.

    Args:
        video_path_a (str): Path to the first video (left).
        video_path_b (str): Path to the second video (right).
        output_path (str): Path to save the output side-by-side video.
        label_a (str): Label to show above first video.
        label_b (str): Label to show above second video.
        annotate (bool): Whether to overlay labels and timestamp.
        resize_height (int): Height to resize both frames for alignment.
        gap_width (int): Width of the gap between videos.
    """
    cap_a = cv2.VideoCapture(video_path_a)
    cap_b = cv2.VideoCapture(video_path_b)

    fps = int(cap_a.get(cv2.CAP_PROP_FPS))
    frame_count = int(min(cap_a.get(cv2.CAP_PROP_FRAME_COUNT), cap_b.get(cv2.CAP_PROP_FRAME_COUNT)))

    def resize_frame(frame, height):
        h, w = frame.shape[:2]
        scale = height / h
        return cv2.resize(frame, (int(w * scale), height))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_writer = None

    for i in range(int(frame_count)):
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()
        if not ret_a or not ret_b:
            break

        frame_a = resize_frame(frame_a, resize_height)
        frame_b = resize_frame(frame_b, resize_height)

        # Add vertical gap between frames
        gap = np.zeros((resize_height, gap_width, 3), dtype=np.uint8)
        combined = np.hstack((frame_a, gap, frame_b))

        if annotate:
            timestamp_sec = i / fps
            text = f"{label_a}     |     {label_b}     |     t = {timestamp_sec:.2f}s"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = int((combined.shape[1] - text_size[0]) / 2)
            text_y = combined.shape[0] - 10

            cv2.rectangle(combined, 
                          (text_x - 10, text_y - 25), 
                          (text_x + text_size[0] + 10, text_y + 5), 
                          (0, 0, 0), 
                          -1)

            cv2.putText(combined, text, (text_x, text_y), font, font_scale, (0, 255, 255), font_thickness)

        if out_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w = combined.shape[:2]
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        out_writer.write(combined)

    cap_a.release()
    cap_b.release()
    out_writer.release()
    print(f"[🎥] Side-by-side video saved to: {output_path}")
