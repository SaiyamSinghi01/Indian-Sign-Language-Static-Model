from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
import numpy as np
import string, copy, itertools
import traceback
import sys

app = Flask(__name__)
CORS(app)

# Load model
model = keras.models.load_model("isl_static_model.h5")

# Warm up (defensive)
try:
    dummy_input = np.zeros((1, 42), dtype=np.float32)
    _ = model.predict(dummy_input, verbose=0)
    print("✅ Model warmed up and ready!", file=sys.stdout)
except Exception as e:
    print("⚠️ Model warmup failed:", e, file=sys.stderr)

# Alphabet mapping (adjust if your training mapping differs)
alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

def pre_process_landmark_fix(landmarks_data):
    """
    Keep original behavior: center using landmark 0, flatten x,y of each landmark,
    normalize by max absolute value, pad/trim to 42 values (21*2).
    """
    temp = copy.deepcopy(landmarks_data)
    if not temp or len(temp) == 0:
        return [0.0] * 42
    # guard: ensure each element has x/y
    for lm in temp:
        lm['x'] = float(lm.get('x', 0.0))
        lm['y'] = float(lm.get('y', 0.0))

    base_x, base_y = temp[0]['x'], temp[0]['y']
    for lm in temp:
        lm['x'] -= base_x
        lm['y'] -= base_y

    flat = list(itertools.chain.from_iterable([[lm['x'], lm['y']] for lm in temp]))

    # normalize
    max_val = max(map(abs, flat)) if len(flat) > 0 else 1.0
    if max_val == 0:
        max_val = 1.0
    normalized = [float(n) / float(max_val) for n in flat]

    # enforce expected length (42)
    expected_len = 42
    if len(normalized) < expected_len:
        normalized += [0.0] * (expected_len - len(normalized))
    elif len(normalized) > expected_len:
        normalized = normalized[:expected_len]

    return normalized

def predict_array(input_array):
    """
    Run model.predict and return probability vector (1D)
    """
    preds = model.predict(input_array, verbose=0)
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[0] == 1:
        probs = preds[0]
    elif preds.ndim == 1:
        probs = preds
    else:
        # if batch >1, average them
        probs = preds.mean(axis=0)

    # If not close to a proper probability distribution, apply softmax
    s = probs.sum()
    if not np.isclose(s, 1.0, atol=1e-3):
        exps = np.exp(probs - np.max(probs))
        probs = exps / np.sum(exps)
    return probs

def top_k_from_probs(probs, k=3):
    probs = np.asarray(probs).ravel()
    k = min(k, probs.shape[0])
    idxs = probs.argsort()[::-1][:k]
    result = []
    for i in idxs:
        label = alphabet[i] if i < len(alphabet) else str(int(i))
        result.append({'label': label, 'confidence': float(probs[int(i)])})
    return result

@app.route('/health', methods=['GET'])
def health_check():
    try:
        model_loaded = model is not None
    except Exception:
        model_loaded = False
    return jsonify({'status': 'healthy' if model_loaded else 'unhealthy', 'model_loaded': model_loaded})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No JSON body provided', 'success': False}), 400

        # allow either single frame via "landmarks" or multiple via "frames"
        if 'frames' in data and isinstance(data['frames'], list) and len(data['frames']) > 0:
            frames = data['frames']
            processed = []
            for frame in frames:
                if not isinstance(frame, list):
                    continue
                processed.append(pre_process_landmark_fix(frame))
            if len(processed) == 0:
                return jsonify({'error': 'No valid frames provided', 'success': False}), 400
            input_arr = np.asarray(processed, dtype=np.float32)
            probs_avg = predict_array(input_arr)  # averaged probabilities
            best_idx = int(np.argmax(probs_avg))
            best_conf = float(probs_avg[best_idx])
            best_label = alphabet[best_idx] if best_idx < len(alphabet) else str(best_idx)
            top_k = top_k_from_probs(probs_avg, k=3)

            print(f"[PREDICT] frames={len(processed)} -> label={best_label} conf={best_conf:.4f}", file=sys.stdout)
            return jsonify({'prediction': best_label, 'confidence': best_conf, 'top_k': top_k, 'success': True})

        # single frame via landmarks
        landmarks = data.get('landmarks', None)
        if not landmarks:
            return jsonify({'error': 'No landmarks or frames provided', 'success': False}), 400

        vec = pre_process_landmark_fix(landmarks)
        input_array = np.asarray([vec], dtype=np.float32)
        probs = predict_array(input_array)
        best_idx = int(np.argmax(probs))
        best_conf = float(probs[best_idx])
        best_label = alphabet[best_idx] if best_idx < len(alphabet) else str(best_idx)
        top_k = top_k_from_probs(probs, k=3)

        # debug print (server console)
        print(f"[PREDICT] landmarks_len={len(landmarks)} -> label={best_label} conf={best_conf:.4f}", file=sys.stdout)
        return jsonify({'prediction': best_label, 'confidence': best_conf, 'top_k': top_k, 'success': True})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    from waitress import serve
    print("Starting ISL Prediction API Server ...", file=sys.stdout)
    print("API available at: http://localhost:5001", file=sys.stdout)
    serve(app, host='0.0.0.0', port=5001, threads=8)
