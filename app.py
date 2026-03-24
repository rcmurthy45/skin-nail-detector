"""
app.py - SkinAI Flask Backend
No login required - users go directly to main page
"""

import os
import json
import uuid
import hashlib
import random
from datetime import datetime

from flask import (
    Flask, render_template, request,
    redirect, url_for, jsonify
)
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image as PILImage

# ==============================================================================
#  MODEL LOADING
# ==============================================================================

model   = None
TF_MODE = "demo"

# Try TFLite first
try:
    import tflite_runtime.interpreter as tflite
    _path = os.path.join("model", "skin_nail_model.tflite")
    if os.path.exists(_path):
        model   = tflite.Interpreter(model_path=_path)
        model.allocate_tensors()
        TF_MODE = "tflite"
        print("OK: TFLite model loaded")
except Exception:
    pass

# Try Keras next
if model is None:
    try:
        from tensorflow.keras.models import load_model as keras_load
        _path = os.path.join("model", "skin_nail_model.h5")
        if os.path.exists(_path):
            model   = keras_load(_path)
            TF_MODE = "keras"
            print("OK: Keras model loaded")
    except Exception:
        pass

if model is None:
    TF_MODE = "demo"
    print("INFO: No model found - running in demo mode")

# ==============================================================================
#  FLASK APP
# ==============================================================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "skinai_secret_2024_local")

# ==============================================================================
#  CONFIG
# ==============================================================================

UPLOAD_FOLDER      = os.path.join("static", "uploads")
DISEASE_INFO_PATH  = "disease_info.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
IMG_SIZE           = 224
CLASS_NAMES        = ["acne", "eczema", "nail_fungus", "psoriasis"]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("model", exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# ==============================================================================
#  DISEASE INFO
# ==============================================================================

BUILTIN_DISEASE_INFO = {
    "acne": {
        "duration": "Mild acne resolves in 2 to 6 weeks. Severe acne may persist for months.",
        "precautions": [
            "Wash your face twice daily with a gentle cleanser",
            "Avoid touching or picking at pimples",
            "Use oil-free, non-comedogenic moisturizers",
            "Change pillowcases every 2 to 3 days"
        ],
        "tips": [
            "Apply benzoyl peroxide or salicylic acid as spot treatment",
            "Consult a dermatologist if acne is severe or leaving scars"
        ]
    },
    "eczema": {
        "duration": "Eczema is chronic. Flare-ups can last days to weeks.",
        "precautions": [
            "Moisturize skin at least twice daily",
            "Use fragrance-free soaps and detergents",
            "Wear loose breathable cotton clothing",
            "Keep nails short to prevent scratching damage"
        ],
        "tips": [
            "Over-the-counter hydrocortisone cream relieves mild flares",
            "Cold compresses can soothe itching temporarily"
        ]
    },
    "psoriasis": {
        "duration": "Psoriasis is chronic with flare-ups lasting weeks to months.",
        "precautions": [
            "Moisturize daily to reduce scaling and dryness",
            "Avoid triggers like stress, smoking, and alcohol",
            "Protect skin from cuts and sunburn",
            "Bathe in lukewarm water and pat skin dry gently"
        ],
        "tips": [
            "Short controlled sun exposure may reduce plaques",
            "Consult a dermatologist for prescription treatments"
        ]
    },
    "nail_fungus": {
        "duration": "Nail fungus takes 6 to 18 months for full recovery even with treatment.",
        "precautions": [
            "Keep nails short, dry, and clean",
            "Wear moisture-wicking socks and change them daily",
            "Never walk barefoot in public pools or gyms",
            "Do not share nail clippers or footwear"
        ],
        "tips": [
            "Antifungal nail lacquer works for mild infections",
            "See a doctor for oral antifungal medication for severe cases"
        ]
    }
}

try:
    with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
        DISEASE_INFO = json.load(f)
    print(f"OK: disease_info.json loaded ({len(DISEASE_INFO)} diseases)")
except Exception:
    DISEASE_INFO = BUILTIN_DISEASE_INFO
    print("INFO: Using built-in disease database")

# ==============================================================================
#  HELPER FUNCTIONS
# ==============================================================================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = PILImage.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def run_prediction(filepath):
    # Demo mode
    if TF_MODE == "demo" or model is None:
        return random.choice(CLASS_NAMES), round(random.uniform(72.0, 96.0), 1)

    arr = preprocess_image(filepath)

    # TFLite
    if TF_MODE == "tflite":
        inp = model.get_input_details()
        out = model.get_output_details()
        model.set_tensor(inp[0]["index"], arr)
        model.invoke()
        preds      = model.get_tensor(out[0]["index"])[0]
        idx        = int(np.argmax(preds))
        confidence = round(float(preds[idx]) * 100, 1)
        return CLASS_NAMES[idx], confidence

    # Keras
    if TF_MODE == "keras":
        preds      = model.predict(arr)[0]
        idx        = int(np.argmax(preds))
        confidence = round(float(preds[idx]) * 100, 1)
        return CLASS_NAMES[idx], confidence

    return random.choice(CLASS_NAMES), round(random.uniform(72.0, 96.0), 1)

# ==============================================================================
#  ROUTES
# ==============================================================================

@app.route("/")
def index():
    """Go directly to main page — no login needed."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts image file, returns prediction JSON.
    No authentication required.
    """
    if "file" not in request.files:
        return jsonify({"error": "No image file received."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Use JPG, PNG, or WEBP images only."}), 400

    # Save uploaded image
    ext      = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Run prediction
    try:
        predicted_class, confidence = run_prediction(filepath)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    info = DISEASE_INFO.get(predicted_class, {})

    return jsonify({
        "disease":     predicted_class.replace("_", " ").title(),
        "confidence":  confidence,
        "duration":    info.get("duration",    "Consult a dermatologist."),
        "precautions": info.get("precautions", ["Keep the area clean and dry."]),
        "tips":        info.get("tips",        ["See a dermatologist for diagnosis."]),
        "image_url":   f"/static/uploads/{filename}",
        "mode":        TF_MODE
    })


@app.route("/translations/<lang>")
def get_translation(lang):
    """Serve translation JSON files."""
    if lang not in ["en", "hi", "te"]:
        lang = "en"
    path = os.path.join("translations", f"{lang}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    except Exception:
        return jsonify({}), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Page not found"}), 404

@app.errorhandler(500)
def server_error(e):
    import traceback
    print("500 ERROR:\n", traceback.format_exc())
    return "Internal server error. Check Render logs.", 500

# ==============================================================================
#  START SERVER
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  SkinAI - Skin & Nail Disease Detector")
    print(f"  Model mode : {TF_MODE.upper()}")
    print("  URL        : http://127.0.0.1:5000")
    print("  No login required!")
    print("="*50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
