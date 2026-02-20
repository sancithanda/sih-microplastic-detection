# Smart India Hackathon 2025
## Microplastic Detection System (Computer Vision Prototype)

This repository contains a computer vision based prototype developed for Smart India Hackathon 2025 (College Level Shortlisted).

---

## Problem Statement
Detect and classify microplastic particles from scattering data using optical and computational techniques.

---

## Proposed Solution
The system processes video frames of scattered light particles and performs:

1. Grayscale conversion and noise reduction
2. Thresholding and contour detection
3. Feature extraction:
   - Area
   - Perimeter
   - Circularity
   - Solidity
   - Aspect Ratio
   - Hu Moments
4. Classification:
   - Rule-based classifier (fallback)
   - Optional SVM machine learning model
5. Real-time bounding box visualization
6. CSV logging of detections

---

## Repository Structure

- `scattering_detection.py` → Real-time detection + classification pipeline
- `extract_features.py` → Builds labeled dataset from videos
- `train_model.py` → Trains SVM classifier (StandardScaler + SVC)
- `requirements.txt` → Dependencies
- `docs/SIH_Presentation.pdf` → Official hackathon PPT

---

## How To Run

Install dependencies: