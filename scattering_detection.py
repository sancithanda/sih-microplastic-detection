# scattering_detection.py
"""
Realtime / demo scattering detection and classification prototype.

Usage:
    python scattering_detection.py --video sample-video/scattering_demo.mp4 --model models/model.pkl --out logs/detections.csv

If --model is not provided or not found, the script uses a fallback rule-based classifier.
Outputs visual overlay and logs detection rows to CSV.
"""

import cv2
import numpy as np
import argparse
import time
import os
import csv
from datetime import datetime

# Optional ML model loading
try:
    from joblib import load as joblib_load
except Exception:
    joblib_load = None

def extract_features_from_contour(cnt, gray):
    """Return a feature vector (list) for a contour and the bounding box."""
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0 or area == 0:
        return None, None

    # Bounding box and ROI
    x, y, w, h = cv2.boundingRect(cnt)
    roi = gray[y:y+h, x:x+w]
    mean_intensity = float(np.mean(roi)) if roi.size > 0 else 0.0

    # Circularity
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

    # Aspect ratio
    aspect_ratio = float(w) / h if h > 0 else 0

    # Extent: area / bounding box area
    bbox_area = w * h if (w*h) > 0 else 1
    extent = float(area) / bbox_area

    # Solidity: area / convex hull area
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull) if hull is not None else 0
    solidity = float(area) / hull_area if hull_area > 0 else 0

    # Hu Moments (log scaled) as additional shape descriptors
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()
    # log transform Hu moments to scale them
    hu_features = []
    for hval in hu:
        if hval == 0:
            hu_features.append(0.0)
        else:
            hu_features.append(-1 * np.sign(hval) * np.log10(abs(hval)))

    features = [
        area,
        perimeter,
        circularity,
        aspect_ratio,
        extent,
        solidity,
        mean_intensity,
    ] + hu_features  # total ~13 features

    return features, (x, y, w, h)


def rule_based_classify(features):
    """A conservative multi-feature rule-based classifier."""
    area = features[0]
    circularity = features[2]
    solidity = features[5]
    mean_intensity = features[6]

    # These thresholds are conservative and can be tuned for your data
    if area < 120 and circularity > 0.45 and solidity > 0.55 and mean_intensity > 120:
        return "Plastic"
    else:
        return "Other"


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main(args):
    # Load ML model if provided
    model = None
    if args.model:
        if joblib_load is None:
            print("[WARN] joblib not installed. Model loading disabled.")
        else:
            if os.path.exists(args.model):
                model = joblib_load(args.model)
                print(f"[INFO] Loaded model from {args.model}")
            else:
                print(f"[WARN] Model file {args.model} not found. Using rule-based classifier.")

    ensure_dir(os.path.dirname(args.out) or ".")

    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
        print("[ERROR] Cannot open video/camera:", args.video)
        return

    # Prepare CSV log
    header = ["timestamp", "frame_no", "x", "y", "w", "h",
              "area", "perimeter", "circularity", "aspect_ratio", "extent", "solidity", "mean_intensity",
              "label"]
    csv_file = open(args.out, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)

    paused = False
    frame_no = 0
    start_time = time.time()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            # optional resizing for speed
            if args.resize and frame.shape[1] > args.resize:
                scale = args.resize / frame.shape[1]
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # apply slight blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Adaptive thresholding or fixed threshold depending on lighting
            if args.adaptive:
                thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
            else:
                # You can tune this threshold or compute using Otsu
                _, thresh = cv2.threshold(blurred, args.thresh, 255, cv2.THRESH_BINARY)

            # Morphological ops to clean
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            count = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # filter by area to ignore noise and huge artifacts
                if area < args.min_area or area > args.max_area:
                    continue

                features, bbox = extract_features_from_contour(cnt, gray)
                if features is None:
                    continue
                x, y, w, h = bbox

                # Classification
                if model is not None:
                    # model expected a 2D array
                    try:
                        pred = model.predict([features])[0]
                        label = str(pred)
                    except Exception as e:
                        # fallback safe rule
                        label = rule_based_classify(features)
                else:
                    label = rule_based_classify(features)

                # Draw box + text
                color = (0, 255, 0) if label.lower().startswith("plastic") else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)
                cv2.putText(frame, label, (x, max(0, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                # Show one feature value
                cv2.putText(frame, f"Circ:{features[2]:.2f}", (x, y+h+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

                # Log
                timestamp = datetime.utcnow().isoformat()
                csv_writer.writerow([timestamp, frame_no, x, y, w, h,
                                     float(features[0]), float(features[1]), float(features[2]),
                                     float(features[3]), float(features[4]), float(features[5]), float(features[6]),
                                     label])
                count += 1

            cv2.putText(frame, f"Count: {count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.imshow("Scattering Detection - Press Esc to quit, p to pause", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord("p"):
            paused = not paused
        elif key == ord("s"):
            # save frame for debugging
            cv2.imwrite(f"debug_frame_{frame_no}.png", frame)
            print("[INFO] saved debug_frame")
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] finished. Log saved to", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scattering detection + classification prototype")
    parser.add_argument("--video", type=str, default=None, help="Path to video file. If not provided, uses webcam (0).")
    parser.add_argument("--model", type=str, default="models/model.pkl", help="Path to trained model (joblib pipeline).")
    parser.add_argument("--out", type=str, default="logs/detections.csv", help="CSV output path for detections.")
    parser.add_argument("--thresh", type=int, default=180, help="Binary threshold value (if not using adaptive).")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive thresholding instead of fixed.")
    parser.add_argument("--min-area", type=int, default=5, help="Minimum contour area to consider.")
    parser.add_argument("--max-area", type=int, default=500, help="Maximum contour area to consider.")
    parser.add_argument("--resize", type=int, default=1000, help="Max width to resize for faster processing (keep aspect).")
    args = parser.parse_args()

    # Ensure folders
    ensure_dir(os.path.dirname(args.out))
    ensure_dir("models")
    ensure_dir("logs")
    main(args)