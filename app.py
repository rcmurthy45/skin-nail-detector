"""
╔══════════════════════════════════════════════════════════════════╗
║   app.py  —  SkinAI Flask Backend                                ║
║   Works on Render FREE tier (no TensorFlow needed)               ║
║   Supports: tflite-runtime → keras → demo mode (auto fallback)   ║
╚══════════════════════════════════════════════════════════════════╝

HOW TO RUN LOCALLY:
    python app.py
    then open http://127.0.0.1:5000

HOW IT WORKS ON RENDER:
    - If model/skin_nail_model.tflite exists  uses TFLite (best)
    - If model/skin_nail_model.h5 exists      uses Keras
    - If neither exists                        Demo mode (random results)
"""

# == Standard library imports ==================================================
import os
import json
import uuid
import hashlib
import random
from datetime import datetime

# == Flask imports ==============================================================
from flask import (
    Flask, render_template, request,
    redirect, url_for, session, jsonify
)
from werkzeug.utils import secure_filename

# == Image processing (always available) =======================================
import numpy as np
from PIL import Image as PILImage

# ==============================================================================
#  STEP 1 — Try loading ML libraries (graceful fallback if not installed)
# ==============================================================================

model   = None     # will hold the loaded model object
TF_MODE = "demo"   # "tflite" | "keras" | "demo"

# -- Try TFLite first (lightest, works on Render free tier) --------------------
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_PATH = os.path.join("model", "skin_nail_model.tflite")
    if os.path.exists(TFLITE_PATH):
        model = tflite.Interpreter(model_path=TFLITE_PATH)
        model.allocate_tensors()
        TF_MODE = "tflite"
        print("TFLite model loaded successfully!")
    else:
        print("tflite-runtime installed but .tflite model not found")
except Exception as e:
    print(f"tflite-runtime not available: {e}")

# -- Try full Keras/TensorFlow next (works locally) ----------------------------
if model is None:
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing import image as keras_image
        KERAS_PATH = os.path.join("model", "skin_nail_model.h5")
        if os.path.exists(KERAS_PATH):
            model   = load_model(KERAS_PATH)
            TF_MODE = "keras"
            print("Keras (.h5) model loaded successfully!")
        else:
            print("TensorFlow installed but model/skin_nail_model.h5 not found")
    except Exception as e:
        print(f"TensorFlow not available: {e}")

# -- Final status --------------------------------------------------------------
if TF_MODE == "demo":
    print("Running in DEMO mode — predictions are random")
    print("Train model locally and upload .tflite for real predictions")

# ==============================================================================
#  STEP 2 — Flask app setup
# ==============================================================================

app = Flask(__name__)

# Secret key reads from Render environment variable, falls back to local key
app.secret_key = os.environ.get("SECRET_KEY", "skinai_local_dev_key_2024")

# ==============================================================================
#  STEP 3 — Configuration
# ==============================================================================

UPLOAD_FOLDER      = os.path.join("static", "uploads")
DISEASE_INFO_PATH  = "disease_info.json"
USERS_DB_PATH      = "users.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
IMG_SIZE           = 224  # MobileNetV2 input size

# Disease class names — must match training order (alphabetical = Keras default)
CLASS_NAMES = ["acne", "eczema", "nail_fungus", "psoriasis"]

# Create required folders if they do not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("model", exist_ok=True)

# Max upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# ==============================================================================
#  STEP 4 — Load disease info database
# ==============================================================================

try:
    with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
        DISEASE_INFO = json.load(f)
    print(f"Disease database loaded ({len(DISEASE_INFO)} diseases)")
except FileNotFoundError:
    print("disease_info.json not found — using empty database")
    DISEASE_INFO = {}

# ==============================================================================
#  STEP 5 — Helper functions
# ==============================================================================

def allowed_file(filename):
    """Check if uploaded file has an allowed image extension."""
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

def hash_password(password):
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load all users from users.json file."""
    if os.path.exists(USERS_DB_PATH):
        with open(USERS_DB_PATH, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save all users to users.json file."""
    with open(USERS_DB_PATH, "w") as f:
        json.dump(users, f, indent=2)

def preprocess_image_pil(img_path):
    """
    Load and preprocess image using Pillow only (no TensorFlow needed).
    Returns numpy array of shape (1, 224, 224, 3) normalized to [0, 1].
    """
    img = PILImage.open(img_path).convert("RGB")   # ensure 3 channels (no alpha)
    img = img.resize((IMG_SIZE, IMG_SIZE))           # resize to 224x224
    arr = np.array(img, dtype=np.float32)            # convert to numpy float array
    arr = arr / 255.0                                 # normalize pixels to [0, 1]
    arr = np.expand_dims(arr, axis=0)                 # add batch dim: (1, 224, 224, 3)
    return arr

# ==============================================================================
#  STEP 6 — Prediction function (handles all 3 modes automatically)
# ==============================================================================

def run_prediction(filepath):
    """
    Run AI prediction on the image at filepath.
    Returns: (predicted_class_name, confidence_percentage)

    Automatically picks the best available method:
      tflite -> keras -> demo (random)
    """

    # -- DEMO MODE: no model available -----------------------------------------
    if TF_MODE == "demo" or model is None:
        predicted_class = random.choice(CLASS_NAMES)
        confidence      = round(random.uniform(72.0, 96.0), 1)
        return predicted_class, confidence

    # -- TFLITE MODE: lightweight, perfect for Render --------------------------
    if TF_MODE == "tflite":
        arr = preprocess_image_pil(filepath)

        # Get model input and output tensor info
        input_details  = model.get_input_details()
        output_details = model.get_output_details()

        # Feed the image into the model
        model.set_tensor(input_details[0]["index"], arr)
        model.invoke()  # run the inference

        # Read output predictions — shape: (1, 4) for 4 classes
        predictions     = model.get_tensor(output_details[0]["index"])[0]
        predicted_idx   = int(np.argmax(predictions))        # index of highest score
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence      = round(float(predictions[predicted_idx]) * 100, 1)
        return predicted_class, confidence

    # -- KERAS MODE: full TensorFlow, works on local machine -------------------
    if TF_MODE == "keras":
        arr             = preprocess_image_pil(filepath)
        predictions     = model.predict(arr)[0]              # shape: (4,)
        predicted_idx   = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence      = round(float(predictions[predicted_idx]) * 100, 1)
        return predicted_class, confidence

    # Fallback safety net
    return random.choice(CLASS_NAMES), round(random.uniform(72.0, 96.0), 1)

# ==============================================================================
#  ROUTES — Authentication
# ==============================================================================

@app.route("/")
def index():
    """Root URL — redirect to home if logged in, else to login page."""
    if "username" in session:
        return redirect(url_for("home"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    """
    GET  /login  — shows the login page
    POST /login  — receives JSON {username, password}, returns JSON response
    """
    if request.method == "POST":
        data     = request.get_json()
        username = data.get("username", "").strip().lower()
        password = data.get("password", "")

        users = load_users()

        if username in users and users[username]["password"] == hash_password(password):
            session["username"] = username
            session["name"]     = users[username].get("name", username)
            return jsonify({"success": True})

        return jsonify({"success": False, "message": "Invalid username or password."})

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """
    GET  /signup — shows the signup page
    POST /signup — receives JSON {name, username, password}, creates account
    """
    if request.method == "POST":
        data     = request.get_json()
        username = data.get("username", "").strip().lower()
        password = data.get("password", "")
        name     = data.get("name", "").strip()

        # Validate inputs
        if not username or not password or not name:
            return jsonify({"success": False, "message": "All fields are required."})

        if len(password) < 6:
            return jsonify({"success": False, "message": "Password must be at least 6 characters."})

        users = load_users()

        if username in users:
            return jsonify({"success": False, "message": "Username already taken."})

        # Create new user record
        users[username] = {
            "name":     name,
            "password": hash_password(password),
            "joined":   datetime.now().isoformat()
        }
        save_users(users)

        # Auto login after signup
        session["username"] = username
        session["name"]     = name
        return jsonify({"success": True})

    return render_template("signup.html")


@app.route("/logout")
def logout():
    """Clear the session and redirect to login page."""
    session.clear()
    return redirect(url_for("login"))


# ==============================================================================
#  ROUTES — Main App
# ==============================================================================

@app.route("/home")
def home():
    """Main prediction page — user must be logged in."""
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", username=session.get("name", "User"))


# ==============================================================================
#  ROUTES — AI Prediction API
# ==============================================================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict

    Accepts : multipart/form-data  with key 'file' containing an image
    Returns : JSON response with these fields:
        {
            "disease":     "Acne",
            "confidence":  87.4,
            "duration":    "2-6 weeks...",
            "precautions": ["wash face twice daily", ...],
            "tips":        ["use benzoyl peroxide", ...],
            "image_url":   "/static/uploads/abc123.jpg",
            "mode":        "tflite"  (or "keras" or "demo")
        }
    """

    # User must be logged in
    if "username" not in session:
        return jsonify({"error": "Unauthorized. Please log in first."}), 401

    # Must have a file in the request
    if "file" not in request.files:
        return jsonify({"error": "No image file received."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use JPG, PNG, or WEBP."}), 400

    # -- Save the uploaded image -----------------------------------------------
    ext      = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"          # unique random filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # -- Run AI prediction -----------------------------------------------------
    try:
        predicted_class, confidence = run_prediction(filepath)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # -- Look up disease information -------------------------------------------
    info = DISEASE_INFO.get(predicted_class, {})

    # -- Return the result as JSON ---------------------------------------------
    return jsonify({
        "disease":     predicted_class.replace("_", " ").title(),
        "confidence":  confidence,
        "duration":    info.get("duration",    "Consult a dermatologist for accurate timeline."),
        "precautions": info.get("precautions", ["Consult a qualified dermatologist."]),
        "tips":        info.get("tips",        ["Maintain proper skin hygiene."]),
        "image_url":   f"/static/uploads/{filename}",
        "mode":        TF_MODE
    })


# ==============================================================================
#  ROUTES — Translations API
# ==============================================================================

@app.route("/translations/<lang>")
def get_translation(lang):
    """
    Serve translation JSON files for multi-language support.

    GET /translations/en  returns English strings
    GET /translations/hi  returns Hindi strings
    GET /translations/te  returns Telugu strings
    """
    allowed_langs = ["en", "hi", "te"]
    if lang not in allowed_langs:
        lang = "en"   # default to English

    path = os.path.join("translations", f"{lang}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"error": f"Translation file {lang}.json not found"}), 404


# ==============================================================================
#  START THE SERVER
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  SkinAI: Skin and Nail Disease Detector")
    print(f"  Model mode : {TF_MODE.upper()}")
    print("  URL        : http://127.0.0.1:5000")
    print("=" * 55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
