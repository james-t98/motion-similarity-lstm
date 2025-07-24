"""
dataset.py

Step 1: Download dataset from https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
Step 2: Extract the dataset using terminal:
    unar UCF101.rar
Step 3: Convert .avi files to .mp4 using this script.

Update the 'input_dir' and 'output_dir' below with your desired categories.
"""

import cv2
import os

input_dir = "/Users/jaimetellie/Downloads/UCF-101/SoccerJuggling"
output_dir = "/Users/jaimetellie/Downloads/sdc_msc_data_analytics_project/data/video/input/Football/Juggling"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".avi"):
        in_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, file.replace(".avi", ".mp4"))

        cap = cv2.VideoCapture(in_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
        out.release()
