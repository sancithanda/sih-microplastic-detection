# extract_features.py
"""
Scan a video and extract features per detected contour, saving to CSV for dataset creation.

Usage:
    python extract_features.py --input sample-video/scattering_demo.mp4 --output data/features.csv --label Plastic
You can run multiple times with different labelled videos to build a dataset.
"""

import cv2
import numpy as np
import argparse
import csv
import os
from datetime import datetime

from scattering_detection import extract_features_from_contour  # reuse logic

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def main(args):
    ensure_dir(os.path.dirname(args.output))
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("[ERROR] cannot open", args.input)
        return

    csv_file = open(args.output, "a", newline="")
    writer = csv.writer(csv_file)
    # If empty file, write header
    if os.stat(args.output).st_size == 0:
        header = ["timestamp","frame_no","area","perimeter","circularity","aspect_ratio","extent","solidity","mean_intensity"] + [f"hu{i+1}" for i in range(7)] + ["label"]
        writer.writerow(header)

    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blurred, args.thresh, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < args.min_area or area > args.max_area:
                continue
            features, bbox = extract_features_from_contour(cnt, gray)
            if features is None:
                continue
            timestamp = datetime.utcnow().isoformat()
            writer.writerow([timestamp, frame_no] + list(map(float, features)) + [args.label])
    csv_file.close()
    cap.release()
    print("[INFO] features appended to", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output", default="data/features.csv", help="CSV output")
    parser.add_argument("--label", default="Plastic", help="Label to attach to all detected particles in this video")
    parser.add_argument("--thresh", type=int, default=180)
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--max-area", type=int, default=500)
    args = parser.parse_args()
    main(args)