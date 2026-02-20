# train_model.py
"""
Train a classifier pipeline (StandardScaler + SVM) on the CSV features created by extract_features.py.
Saves a pipeline to models/model.pkl
"""

import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib

def main(args):
    df = pd.read_csv(args.data)
    # Drop non-feature columns: timestamp, frame_no if present
    drop_cols = [c for c in ["timestamp","frame_no"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    # Last column is label
    label_col = df.columns[-1]
    X = df.iloc[:, :-1].values
    y = df[label_col].values

    # Simple train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True))
    ])

    print("[INFO] Training classifier...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    cv_scores = cross_val_score(pipeline, X, y, cv=5)

    print(f"[RESULT] train_acc={train_acc:.3f} test_acc={test_acc:.3f} cv_mean={cv_scores.mean():.3f}")

    # Save model
    ensure_dir = lambda p: os.makedirs(p, exist_ok=True)
    ensure_dir(os.path.dirname(args.out))
    joblib.dump(pipeline, args.out)
    print("[INFO] Saved model to", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/features.csv", help="CSV file with features and label")
    parser.add_argument("--out", default="models/model.pkl", help="Output model path")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    main(args)