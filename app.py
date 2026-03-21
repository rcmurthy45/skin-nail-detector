"""
╔══════════════════════════════════════════════════════╗
║   AI-Based Skin & Nail Disease Detection Web App     ║
║   B.Tech Final Year Project                          ║
║   Backend: Flask | Model: MobileNetV2 (TensorFlow)   ║
╚══════════════════════════════════════════════════════╝
"""

from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import os
import json
import uuid
import hashlib
from datetime import datetime
from werkzeug.utils import secure_filename

# ── REMOVE these old lines ──────────────────────────
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image as keras_image

# ── ADD these new lines instead ─────────────────────
import numpy as np
from PIL import Image

try:
    import tflite_runtime.interpreter as tflite
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("tflite-runtime not available — running demo mode")

# ── Load TFLite model ────────────────────────────────
interpreter = None
MODEL_PATH  = "model/skin_nail_model.tflite"

if TF_AVAILABLE and os.path.exists(MODEL_PATH):
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print("✅ TFLite model loaded!")
else:
    print("⚠️  Model not found — running in demo mode")

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fallback_dev_key_2024") # Change in production!

# ── Configuration ─────────────────────────────────────────────────────────────
UPLOAD_FOLDER   = "static/uploads"
ALLOWED_EXT     = {"png", "jpg", "jpeg", "gif", "webp"}
MODEL_PATH      = "model/skin_nail_model.h5"
USERS_FILE      = "users.json"
DISEASE_DB_FILE = "disease_info.json"
CLASS_NAMES     = ["acne", "eczema", "nail_fungus", "psoriasis"]  # alphabetical = Keras default

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB limit

# ── Load the trained model (if available) ────────────────────────────────────
model = None
if MODEL_AVAILABLE and os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
else:
    print("⚠️  Model not found at model/skin_nail_model.h5  — train it first with train_model.py")

# ── Load disease info database ───────────────────────────────────────────────
with open(DISEASE_DB_FILE, "r", encoding="utf-8") as f:
    disease_db = json.load(f)

# ── Helper: load / save users ─────────────────────────────────────────────────
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# ── Preprocess image for MobileNetV2 ─────────────────────────────────────────
def preprocess_image(img_path, target_size=(224, 224)):
    img   = keras_image.load_img(img_path, target_size=target_size)
    arr   = keras_image.img_to_array(img)
    arr   = np.expand_dims(arr, axis=0)
    arr   = arr / 255.0          # normalize to [0, 1]
    return arr

# ════════════════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Redirect to login if not authenticated, else to main app."""
    if "username" in session:
        return redirect(url_for("home"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data     = request.get_json()
        username = data.get("username", "").strip().lower()
        password = data.get("password", "")
        users    = load_users()

        if username in users and users[username]["password"] == hash_password(password):
            session["username"] = username
            session["name"]     = users[username]["name"]
            return jsonify({"success": True})
        return jsonify({"success": False, "message": "Invalid username or password."})

    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        data     = request.get_json()
        username = data.get("username", "").strip().lower()
        password = data.get("password", "")
        name     = data.get("name", "").strip()
        users    = load_users()

        if not username or not password or not name:
            return jsonify({"success": False, "message": "All fields are required."})
        if len(password) < 6:
            return jsonify({"success": False, "message": "Password must be at least 6 characters."})
        if username in users:
            return jsonify({"success": False, "message": "Username already taken."})

        users[username] = {
            "name":     name,
            "password": hash_password(password),
            "joined":   datetime.now().isoformat()
        }
        save_users(users)
        session["username"] = username
        session["name"]     = name
        return jsonify({"success": True})

    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ════════════════════════════════════════════════════════════════════════════

@app.route("/home")
def home():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", username=session.get("name", "User"))

# ════════════════════════════════════════════════════════════════════════════
#  PREDICTION API  —  POST /predict
# ════════════════════════════════════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
def predict():
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file received"}), 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    # Save uploaded file
    ext      = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # ── Predict using TFLite ──────────────────────────
    if interpreter is None:
        # Demo mode — random prediction
        import random
        predicted_class = random.choice(CLASS_NAMES)
        confidence      = round(random.uniform(70, 97), 1)
    else:
        # Real prediction
        img = Image.open(filepath).resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()

        predictions     = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_idx   = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence      = round(float(predictions[predicted_idx]) * 100, 1)

    info = disease_db.get(predicted_class, {})

    return jsonify({
        "disease":     predicted_class.replace("_", " ").title(),
        "confidence":  confidence,
        "duration":    info.get("duration", "Consult a dermatologist."),
        "precautions": info.get("precautions", []),
        "tips":        info.get("tips", []),
        "image_url":   f"/static/uploads/{filename}"
    })

# ════════════════════════════════════════════════════════════════════════════
#  TRANSLATION API
# ════════════════════════════════════════════════════════════════════════════

@app.route("/translations/<lang>")
def get_translation(lang):
    """Serve translation JSON for the frontend."""
    allowed_langs = ["en", "hi", "te"]
    if lang not in allowed_langs:
        lang = "en"
    path = f"translations/{lang}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({}), 404

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀  Starting Skin & Nail Disease Detector  —  http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
