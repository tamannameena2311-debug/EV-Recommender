import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ev_detector import FEATURE_COLUMNS, SAMPLING_RATE, extract_features_from_capture


def resample_capture(df):
    df = df.copy()
    df = df[["time_sec", "x", "y", "z"]]
    for column in ["time_sec", "x", "y", "z"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna().sort_values("time_sec").drop_duplicates(subset="time_sec").reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("not enough rows")

    df["time_sec"] = df["time_sec"] - df["time_sec"].iloc[0]
    duration = float(df["time_sec"].iloc[-1])
    target_times = np.arange(0, duration, 1 / SAMPLING_RATE)
    if len(target_times) < SAMPLING_RATE * 3:
        raise ValueError("recording shorter than 3 seconds")

    resampled = pd.DataFrame({"time_sec": target_times})
    for axis in ["x", "y", "z"]:
        resampled[axis] = np.interp(target_times, df["time_sec"].values, df[axis].values)
    return resampled


def load_non_ev_csv(path):
    df = pd.read_csv(path)
    df.columns = [str(column).lower().strip() for column in df.columns]

    if "milliseconds" in df.columns:
        time_sec = pd.to_numeric(df["milliseconds"], errors="coerce") / 1000
    elif "time_sec" in df.columns:
        time_sec = pd.to_numeric(df["time_sec"], errors="coerce")
    else:
        time_sec = pd.Series(np.arange(len(df)) / SAMPLING_RATE)

    capture = pd.DataFrame({
        "time_sec": time_sec,
        "x": df["x"],
        "y": df["y"],
        "z": df["z"],
    })
    return resample_capture(capture)


def load_ev_raw_txt(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] < 5:
        raise ValueError("EV raw file needs at least 5 columns")

    capture = pd.DataFrame({
        "time_sec": df.iloc[:, 0],
        "x": df.iloc[:, 2],
        "y": df.iloc[:, 3],
        "z": df.iloc[:, 4],
    })
    return resample_capture(capture)


def extract_file_features(path, label):
    if label == "ev":
        capture = load_ev_raw_txt(path)
    else:
        capture = load_non_ev_csv(path)

    features = extract_features_from_capture(capture)
    features["label"] = label
    features["source_file"] = path.name
    return features


def load_dataset(ev_dir, non_ev_dir):
    files = []
    for path in sorted(Path(ev_dir).glob("*.txt")):
        files.append((path, "ev"))
    for path in sorted(Path(non_ev_dir).glob("*.csv")):
        files.append((path, "non_ev"))

    feature_frames = []
    skipped = []
    for path, label in files:
        try:
            feature_frames.append(extract_file_features(path, label))
            print(f"Loaded {label}: {path.name}")
        except Exception as exc:
            skipped.append((str(path), str(exc)))
            print(f"Skipped {path.name}: {exc}")

    if not feature_frames:
        raise ValueError("No usable accelerometer files found")

    return pd.concat(feature_frames, ignore_index=True), skipped


def evaluate_by_file(test_features, predictions):
    report_rows = []
    temp = test_features.copy()
    temp["predicted_label"] = predictions

    for source_file in temp["source_file"].unique():
        file_rows = temp[temp["source_file"] == source_file]
        true_label = file_rows["label"].iloc[0]
        predicted_label = file_rows["predicted_label"].value_counts().idxmax()
        report_rows.append({
            "source_file": source_file,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "windows": int(len(file_rows)),
        })

    return report_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ev-dir", required=True)
    parser.add_argument("--non-ev-dir", required=True)
    parser.add_argument("--output", default="models/ev_detector.joblib")
    args = parser.parse_args()

    all_features, skipped = load_dataset(args.ev_dir, args.non_ev_dir)
    sources = all_features[["source_file", "label"]].drop_duplicates()
    train_sources, test_sources = train_test_split(
        sources,
        test_size=0.25,
        random_state=42,
        stratify=sources["label"],
    )

    train_features = all_features[all_features["source_file"].isin(train_sources["source_file"])]
    test_features = all_features[all_features["source_file"].isin(test_sources["source_file"])]

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        max_depth=None,
    )

    model.fit(train_features[FEATURE_COLUMNS], train_features["label"])
    predictions = model.predict(test_features[FEATURE_COLUMNS])

    print("\nWindow accuracy:", accuracy_score(test_features["label"], predictions))
    print("\nConfusion matrix:")
    print(confusion_matrix(test_features["label"], predictions))
    print("\nClassification report:")
    print(classification_report(test_features["label"], predictions))
    print("\nFile-level results:")
    for row in evaluate_by_file(test_features, predictions):
        print(row)

    final_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        max_depth=None,
    )
    final_model.fit(all_features[FEATURE_COLUMNS], all_features["label"])

    bundle = {
        "model": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "training_files": int(len(sources)),
        "training_windows": int(len(all_features)),
        "skipped_files": skipped,
        "sklearn_version": sklearn.__version__,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(bundle, args.output)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
