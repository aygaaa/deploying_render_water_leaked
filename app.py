from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# ── Load model + scaler + threshold ───────────────────────────
scaler    = joblib.load("scaler.pkl")
threshold = joblib.load("threshold.pkl")

try:
    import tensorflow as tf
    model = tf.keras.models.load_model("water_leakage_model.keras")
    MODEL_LOADED = True
except Exception as e:
    print(f"Model load warning: {e}")
    MODEL_LOADED = False

WINDOW      = 48
NUM_SENSORS = 119

# ── Pages ──────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ── API: health check ──────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({
        "status":       "online",
        "model_loaded": MODEL_LOADED,
        "threshold":    round(threshold, 6),
        "window_size":  WINDOW,
        "num_sensors":  NUM_SENSORS,
    })

# ── API: predict ───────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 503

    data     = request.get_json()
    readings = np.array(data["readings"], dtype=np.float32)

    if readings.shape[0] < WINDOW:
        return jsonify({"error": f"Need at least {WINDOW} rows, got {readings.shape[0]}"}), 400
    if readings.shape[1] != NUM_SENSORS:
        return jsonify({"error": f"Expected {NUM_SENSORS} sensors, got {readings.shape[1]}"}), 400

    scaled  = scaler.transform(readings)
    window  = scaled[-WINDOW:].reshape(1, WINDOW, NUM_SENSORS)
    pred    = model.predict(window, verbose=0)
    mse     = float(np.mean((window - pred) ** 2))
    anomaly = bool(mse > threshold)

    return jsonify({
        "mse":          round(mse, 6),
        "threshold":    round(threshold, 6),
        "anomaly":      anomaly,
        "confidence":   round(min(mse / threshold, 3.0), 3),
        "status":       "LEAK DETECTED" if anomaly else "Normal",
        "severity":     _severity(mse, threshold),
    })

# ── API: simulate (demo without real data) ────────────────────
@app.route("/api/simulate", methods=["POST"])
def simulate():
    data     = request.get_json()
    inject   = data.get("inject_leak", False)
    readings = np.random.randn(WINDOW, NUM_SENSORS).astype(np.float32) * 0.3

    if inject:
        readings[-10:, :5] += np.random.uniform(5, 10, (10, 5))

    if MODEL_LOADED:
        scaled  = scaler.transform(readings)
        window  = scaled[-WINDOW:].reshape(1, WINDOW, NUM_SENSORS)
        pred    = model.predict(window, verbose=0)
        mse     = float(np.mean((window - pred) ** 2))
    else:
        mse = float(np.random.uniform(0.6, 1.2) if inject else np.random.uniform(0.1, 0.3))

    anomaly = bool(mse > threshold)
    return jsonify({
        "mse":       round(mse, 6),
        "threshold": round(threshold, 6),
        "anomaly":   anomaly,
        "status":    "LEAK DETECTED" if anomaly else "Normal",
        "severity":  _severity(mse, threshold),
        "simulated": True,
        "leak_injected": inject,
    })

def _severity(mse, thr):
    ratio = mse / thr
    if ratio < 0.5:   return "low"
    if ratio < 1.0:   return "medium"
    if ratio < 1.5:   return "high"
    return "critical"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
