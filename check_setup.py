"""
check_setup.py — Run this to verify your project setup before training.
Usage:  python check_setup.py
"""

import os, sys, json

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

print("\n" + "="*55)
print("  SkinAI — Setup Checker")
print("="*55)

errors = 0

# 1. Python version
v = sys.version_info
ok = v.major == 3 and v.minor >= 9
print(f"\n{'Python version':<35}", f"{PASS} {v.major}.{v.minor}.{v.micro}" if ok else f"{FAIL} {v.major}.{v.minor} — need 3.9+")
if not ok: errors += 1

# 2. Flask
try:
    import flask
    print(f"{'Flask':<35}", f"{PASS} {flask.__version__}")
except ImportError:
    print(f"{'Flask':<35}", f"{FAIL} Not installed — run: pip install flask")
    errors += 1

# 3. TensorFlow
try:
    import tensorflow as tf
    print(f"{'TensorFlow':<35}", f"{PASS} {tf.__version__}")
except ImportError:
    print(f"{'TensorFlow':<35}", f"{WARN} Not installed — run: pip install tensorflow")

# 4. Pillow
try:
    from PIL import Image
    import PIL
    print(f"{'Pillow (PIL)':<35}", f"{PASS} {PIL.__version__}")
except ImportError:
    print(f"{'Pillow':<35}", f"{FAIL} Not installed — run: pip install Pillow")
    errors += 1

# 5. NumPy
try:
    import numpy as np
    print(f"{'NumPy':<35}", f"{PASS} {np.__version__}")
except ImportError:
    print(f"{'NumPy':<35}", f"{FAIL} Not installed")
    errors += 1

# 6. Required files
print("\n── Project Files ──────────────────────────────────")
for f in ["app.py", "train_model.py", "disease_info.json", "requirements.txt"]:
    exists = os.path.exists(f)
    print(f"  {f:<40}", PASS if exists else FAIL)
    if not exists: errors += 1

for f in ["translations/en.json", "translations/hi.json", "translations/te.json"]:
    exists = os.path.exists(f)
    print(f"  {f:<40}", PASS if exists else FAIL)
    if not exists: errors += 1

for f in ["templates/login.html", "templates/signup.html", "templates/index.html"]:
    exists = os.path.exists(f)
    print(f"  {f:<40}", PASS if exists else FAIL)
    if not exists: errors += 1

# 7. Trained model
print("\n── Model ───────────────────────────────────────────")
model_path = "model/skin_nail_model.h5"
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024*1024)
    print(f"  {model_path:<40} {PASS} ({size_mb:.1f} MB)")
else:
    print(f"  {model_path:<40} {WARN} Not found — run train_model.py")

# 8. Dataset
print("\n── Dataset ─────────────────────────────────────────")
classes = ["acne", "eczema", "nail_fungus", "psoriasis"]
total = 0
for cls in classes:
    path = f"dataset/{cls}"
    if os.path.exists(path):
        count = len([x for x in os.listdir(path) if x.lower().endswith((".jpg",".jpeg",".png",".webp"))])
        total += count
        status = PASS if count >= 50 else (WARN + " (need more images)")
        print(f"  dataset/{cls:<30} {status}  {count} images")
    else:
        print(f"  dataset/{cls:<30} {FAIL}  folder missing")

if total == 0:
    print(f"\n  {WARN} No dataset found — see README.md for download instructions")
else:
    print(f"\n  Total images: {total}")

# Summary
print("\n" + "="*55)
if errors == 0:
    print("  ✅ All checks passed! Run:  python app.py")
else:
    print(f"  ❌ {errors} issue(s) found. Fix them before running.")
print("="*55 + "\n")
