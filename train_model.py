# =============================================================
# train_model.py
# AI Skin & Nail Disease Detection — Model Training Script
# Uses MobileNetV2 Transfer Learning + Data Augmentation
# =============================================================

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── CONFIGURATION ────────────────────────────────────────────
IMG_SIZE = 224          # MobileNetV2 expects 224x224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001
DATASET_DIR = "dataset"  # Root folder with subfolders per class
MODEL_SAVE_PATH = "model/skin_nail_model.h5"
CLASS_NAMES = ["acne", "eczema", "nail_fungus", "psoriasis"]  # Must match folder names

# ─── STEP 1: DATA AUGMENTATION ────────────────────────────────
# Training generator with augmentation to prevent overfitting
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,           # Normalize pixel values to [0, 1]
    rotation_range=20,           # Randomly rotate images up to 20 degrees
    horizontal_flip=True,        # Flip images horizontally
    zoom_range=0.2,              # Random zoom in/out by 20%
    brightness_range=[0.8, 1.2], # Randomly adjust brightness
    width_shift_range=0.1,       # Shift image left/right by 10%
    height_shift_range=0.1,      # Shift image up/down by 10%
    shear_range=0.1,             # Shear transformation
    fill_mode='nearest',         # Fill gaps created by shifts
    validation_split=0.2         # 80% train, 20% validation
)

# Validation generator — NO augmentation, just rescaling
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

print("[INFO] Loading training data with augmentation...")
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

print("[INFO] Loading validation data...")
val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

# Save class indices to JSON (needed during prediction)
class_indices = train_generator.class_indices
class_indices_inv = {v: k for k, v in class_indices.items()}
with open("model/class_indices.json", "w") as f:
    json.dump(class_indices_inv, f)
print(f"[INFO] Class mapping saved: {class_indices_inv}")

# ─── STEP 2: BUILD MODEL (MobileNetV2 Transfer Learning) ──────
print("[INFO] Building MobileNetV2 model...")

# Load MobileNetV2 pretrained on ImageNet, remove top classification layers
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,              # We'll add our own classifier
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base model layers initially (use pretrained features)
base_model.trainable = False

# Add custom classification layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)         # Reduce spatial dimensions
x = BatchNormalization()(x)             # Normalize activations
x = Dense(256, activation='relu')(x)   # Fully connected layer
x = Dropout(0.5)(x)                    # Dropout to prevent overfitting
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(len(CLASS_NAMES), activation='softmax')(x)  # Output layer

model = Model(inputs=base_model.input, outputs=predictions)

# ─── STEP 3: COMPILE MODEL ────────────────────────────────────
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─── STEP 4: CALLBACKS ────────────────────────────────────────
os.makedirs("model", exist_ok=True)

callbacks = [
    # Save the best model based on validation accuracy
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    # Stop training early if validation loss doesn't improve
    EarlyStopping(monitor='val_loss', patience=7,
                  restore_best_weights=True, verbose=1),
    # Reduce learning rate when learning plateaus
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, min_lr=1e-7, verbose=1)
]

# ─── STEP 5: TRAIN (PHASE 1 — Frozen Base) ────────────────────
print("\n[PHASE 1] Training with frozen base model...")
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

# ─── STEP 6: FINE-TUNING (PHASE 2 — Unfreeze Top Layers) ──────
print("\n[PHASE 2] Fine-tuning top layers of MobileNetV2...")
# Unfreeze the top 30 layers of the base model for fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Recompile with a much lower learning rate
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    initial_epoch=15,
    callbacks=callbacks,
    verbose=1
)

# ─── STEP 7: SAVE FINAL MODEL ─────────────────────────────────
model.save(MODEL_SAVE_PATH)
print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")

# ─── STEP 8: PLOT ACCURACY & LOSS (Optional) ──────────────────
try:
    import matplotlib.pyplot as plt

    # Combine histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    epochs_range = range(len(acc))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, acc, label='Training Accuracy', color='royalblue')
    ax1.plot(epochs_range, val_acc, label='Validation Accuracy', color='tomato')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, loss, label='Training Loss', color='royalblue')
    ax2.plot(epochs_range, val_loss, label='Validation Loss', color='tomato')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model/training_history.png', dpi=150)
    print("📊 Training graph saved to: model/training_history.png")
    plt.show()
except ImportError:
    print("[INFO] matplotlib not installed — skipping graph generation.")
