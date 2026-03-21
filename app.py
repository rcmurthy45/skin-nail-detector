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

# ── Try importing TensorFlow (optional if model not yet trained) ──────────────
try:
    import numpy as np
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image as keras_image
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("⚠️  TensorFlow not installed. Run: pip install tensorflow")

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
    """
    Accepts a multipart image upload, runs MobileNetV2, returns JSON:
    {
      "disease":     "Acne",
      "confidence":  87.4,
      "duration":    "...",
      "precautions": ["...", "..."],
      "tips":        ["...", "..."]
    }
    """
    if "username" not in session:
        return jsonify({"error": "Unauthorized. Please log in."}), 401

    if "file" not in request.files:
        return jsonify({"error": "No image file received."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use JPG, PNG, GIF, or WEBP."}), 400

    # ── Save uploaded file ────────────────────────────────────────────────
    ext      = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # ── Predict ───────────────────────────────────────────────────────────
    if model is None or not MODEL_AVAILABLE:
        # ── DEMO MODE (no trained model) ──────────────────────────────────
        import random
        predicted_class = random.choice(CLASS_NAMES)
        confidence      = round(random.uniform(70, 97), 1)
    else:
        img_array       = preprocess_image(filepath)
        predictions     = model.predict(img_array)[0]      # shape: (4,)
        predicted_idx   = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence      = round(float(predictions[predicted_idx]) * 100, 1)

    # ── Fetch disease metadata ────────────────────────────────────────────
    info = disease_db.get(predicted_class, {})

    return jsonify({
        "disease":     predicted_class.replace("_", " ").title(),
        "confidence":  confidence,
        "duration":    info.get("duration",    "Consult a dermatologist."),
        "precautions": info.get("precautions", []),
        "tips":        info.get("tips",        []),
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
