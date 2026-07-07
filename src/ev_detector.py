import io
import os

import joblib
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt


SAMPLING_RATE = 100
WINDOW_SECONDS = 3
MIN_CAPTURE_SECONDS = 9
WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS
LOW_FREQ_CUTOFF = 3.0
FFT_MIN_FREQ = 3.0
FFT_MAX_FREQ = 45.0
STOP_PERCENTILE = 50
MODEL_PATH = os.getenv("EV_DETECTOR_MODEL_PATH", "models/ev_detector.joblib")

FEATURE_COLUMNS = [
    "mean",
    "std",
    "rms",
    "max",
    "min",
    "peak_to_peak",
    "jerk_mean",
    "jerk_std",
    "zero_crossing_rate",
    "fft_peak_freq",
    "fft_peak_mag",
    "fft_energy",
    "spectral_centroid",
    "spectral_entropy",
    "motion_score",
]


def _normalize_columns(df):
    df = df.copy()
    df.columns = [str(col).lower().strip() for col in df.columns]

    aliases = {
        "acceleration_x": "x",
        "acceleration_y": "y",
        "acceleration_z": "z",
        "accel_x": "x",
        "accel_y": "y",
        "accel_z": "z",
        "timestamp": "timestamp_ms",
        "time": "time_sec",
    }
    return df.rename(columns={key: value for key, value in aliases.items() if key in df.columns})


def _prepare_capture(csv_bytes):
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as exc:
        raise ValueError("Could not read accelerometer CSV") from exc

    df = _normalize_columns(df)
    required = ["x", "y", "z"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    if "time_sec" not in df.columns:
        if "timestamp_ms" in df.columns:
            df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
            start_ms = df["timestamp_ms"].dropna().iloc[0]
            df["time_sec"] = (df["timestamp_ms"] - start_ms) / 1000
        else:
            df["time_sec"] = df.index / SAMPLING_RATE

    keep_columns = ["time_sec", "x", "y", "z"]
    df = df[keep_columns].copy()
    for column in keep_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna().sort_values("time_sec").drop_duplicates(subset="time_sec").reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Capture does not contain enough accelerometer samples")

    df["time_sec"] = df["time_sec"] - df["time_sec"].iloc[0]
    duration = float(df["time_sec"].iloc[-1])
    if duration < MIN_CAPTURE_SECONDS:
        raise ValueError(f"Record at least {MIN_CAPTURE_SECONDS} seconds before sending")

    target_times = np.arange(0, duration, 1 / SAMPLING_RATE)
    if len(target_times) < MIN_CAPTURE_SECONDS * SAMPLING_RATE:
        raise ValueError(f"Record at least {MIN_CAPTURE_SECONDS} seconds before sending")

    resampled = pd.DataFrame({"time_sec": target_times})
    for axis in ["x", "y", "z"]:
        resampled[axis] = np.interp(target_times, df["time_sec"].values, df[axis].values)

    return resampled, {
        "raw_samples": int(len(df)),
        "resampled_samples": int(len(resampled)),
        "duration_seconds": round(duration, 2),
    }


def _add_magnitude_and_remove_gravity(df):
    df = df.copy()
    df["magnitude"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    rolling_mean = df["magnitude"].rolling(
        window=SAMPLING_RATE,
        center=True,
        min_periods=1,
    ).mean()
    df["vibration_no_gravity"] = df["magnitude"] - rolling_mean
    return df


def _highpass_filter(signal, cutoff_freq, sampling_rate):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(N=4, Wn=normal_cutoff, btype="highpass")
    return filtfilt(b, a, signal)


def extract_features_from_capture(df):
    df = _add_magnitude_and_remove_gravity(df.sort_values("time_sec").reset_index(drop=True))
    window_infos = []

    for start in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = df.iloc[start:start + WINDOW_SIZE]
        signal = window["vibration_no_gravity"].values
        std_val = np.std(signal)
        peak_to_peak = np.max(signal) - np.min(signal)
        jerk_std = np.std(np.diff(signal))
        motion_score = std_val + peak_to_peak + jerk_std
        window_infos.append({"start": start, "motion_score": motion_score})

    if not window_infos:
        raise ValueError("No complete detection window was found")

    stop_threshold = np.percentile([item["motion_score"] for item in window_infos], STOP_PERCENTILE)
    features = []

    for item in window_infos:
        if item["motion_score"] > stop_threshold:
            continue

        window = df.iloc[item["start"]:item["start"] + WINDOW_SIZE]
        raw_signal = window["vibration_no_gravity"].values

        try:
            filtered_signal = _highpass_filter(
                raw_signal,
                cutoff_freq=LOW_FREQ_CUTOFF,
                sampling_rate=SAMPLING_RATE,
            )
        except Exception:
            continue

        mean_val = np.mean(filtered_signal)
        std_val = np.std(filtered_signal)
        rms_val = np.sqrt(np.mean(filtered_signal ** 2))
        max_val = np.max(filtered_signal)
        min_val = np.min(filtered_signal)
        peak_to_peak = max_val - min_val
        jerk = np.diff(filtered_signal)
        jerk_mean = np.mean(np.abs(jerk))
        jerk_std = np.std(jerk)
        zero_crossings = np.where(np.diff(np.sign(filtered_signal)))[0]
        zero_crossing_rate = len(zero_crossings) / len(filtered_signal)

        fft_values = np.abs(rfft(filtered_signal))
        fft_freqs = rfftfreq(len(filtered_signal), d=1 / SAMPLING_RATE)
        fft_values[0] = 0
        freq_mask = (fft_freqs >= FFT_MIN_FREQ) & (fft_freqs <= FFT_MAX_FREQ)
        selected_freqs = fft_freqs[freq_mask]
        selected_fft = fft_values[freq_mask]

        if len(selected_fft) == 0:
            continue

        peak_index = np.argmax(selected_fft)
        fft_peak_freq = selected_freqs[peak_index]
        fft_peak_mag = selected_fft[peak_index]
        fft_energy = np.sum(selected_fft ** 2)
        spectral_centroid = (
            np.sum(selected_freqs * selected_fft) / np.sum(selected_fft)
            if np.sum(selected_fft) > 0
            else 0
        )
        fft_prob = selected_fft / (np.sum(selected_fft) + 1e-12)
        spectral_entropy = -np.sum(fft_prob * np.log2(fft_prob + 1e-12))

        features.append({
            "mean": mean_val,
            "std": std_val,
            "rms": rms_val,
            "max": max_val,
            "min": min_val,
            "peak_to_peak": peak_to_peak,
            "jerk_mean": jerk_mean,
            "jerk_std": jerk_std,
            "zero_crossing_rate": zero_crossing_rate,
            "fft_peak_freq": fft_peak_freq,
            "fft_peak_mag": fft_peak_mag,
            "fft_energy": fft_energy,
            "spectral_centroid": spectral_centroid,
            "spectral_entropy": spectral_entropy,
            "motion_score": item["motion_score"],
        })

    if not features:
        raise ValueError("No stopped windows found. Try recording while the phone is steady.")

    return pd.DataFrame(features, columns=FEATURE_COLUMNS)


def _load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    model_bundle = joblib.load(MODEL_PATH)
    if isinstance(model_bundle, dict):
        model = model_bundle.get("model")
        feature_columns = model_bundle.get("feature_columns", FEATURE_COLUMNS)
        return model, feature_columns

    return model_bundle, FEATURE_COLUMNS


def _fallback_prediction(features):
    median_rms = float(features["rms"].median())
    median_peak_to_peak = float(features["peak_to_peak"].median())
    median_jerk_std = float(features["jerk_std"].median())
    median_fft_energy = float(features["fft_energy"].median())

    vibration_index = (
        (median_rms * 12)
        + (median_peak_to_peak * 1.8)
        + (median_jerk_std * 7)
        + (np.log10(median_fft_energy + 1) * 0.12)
    )
    ev_probability = float(np.clip(1 / (1 + vibration_index), 0.05, 0.95))
    prediction = "ev" if ev_probability >= 0.5 else "non_ev"
    confidence = float(abs(ev_probability - 0.5) * 2)

    return {
        "prediction": prediction,
        "label": "EV-like" if prediction == "ev" else "Non-EV-like",
        "confidence": round(confidence, 3),
        "ev_probability": round(ev_probability, 3),
        "model_status": "fallback_no_trained_model",
    }


def _model_prediction(features, model, feature_columns):
    x_values = features[feature_columns]
    predictions = model.predict(x_values)
    prediction_counts = pd.Series(predictions).value_counts()
    prediction = str(prediction_counts.idxmax())
    confidence = float(prediction_counts.max() / len(predictions))
    ev_probability = None

    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        probabilities = model.predict_proba(x_values)
        classes = [str(item) for item in model.classes_]
        if "ev" in classes:
            ev_index = classes.index("ev")
            ev_probability = float(np.mean(probabilities[:, ev_index]))

    return {
        "prediction": prediction,
        "label": "EV-like" if prediction == "ev" else "Non-EV-like",
        "confidence": round(confidence, 3),
        "ev_probability": round(ev_probability, 3) if ev_probability is not None else None,
        "model_status": "trained_model",
    }


def predict_accelerometer_csv(csv_bytes):
    capture, capture_meta = _prepare_capture(csv_bytes)
    features = extract_features_from_capture(capture)
    model_bundle = _load_model()

    if model_bundle is None:
        prediction = _fallback_prediction(features)
    else:
        model, feature_columns = model_bundle
        prediction = _model_prediction(features, model, feature_columns)

    summary = {
        "windows_used": int(len(features)),
        "rms": round(float(features["rms"].median()), 6),
        "peak_to_peak": round(float(features["peak_to_peak"].median()), 6),
        "fft_peak_freq": round(float(features["fft_peak_freq"].median()), 3),
        "fft_energy": round(float(features["fft_energy"].median()), 6),
    }

    return {
        **prediction,
        **capture_meta,
        "features": summary,
    }
