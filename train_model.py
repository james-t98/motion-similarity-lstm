# train_model.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError

sys.path.append("/content/drive/MyDrive/sdc_msc_data_analytics_project")

from utils import compute_similarity_to_reference
from data_loader import load_and_preprocess_angles


def train_model_from_data(
    data,
    category,
    exercise,
    save_root,
    plots_root,
    custom_model_name=None,
    epochs=50,
    batch_size=16,
    patience=5
):
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]

    # Use first window as reference
    y_train = compute_similarity_to_reference(X_train, X_train[0], method="cosine")
    y_val = compute_similarity_to_reference(X_val, X_train[0], method="cosine")
    y_test = compute_similarity_to_reference(X_test, X_train[0], method="cosine")

    print(f"📦 Data loaded: X_train={X_train.shape}, y_train={y_train.shape}")

    # Build model
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())

    # Train
    callbacks = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Run ID for saving files
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = now

    # Save model
    model_name = custom_model_name or f"lstm_{category}_{exercise}_{run_id}"
    model_path = os.path.join(save_root, f"{model_name}.h5")
    os.makedirs(save_root, exist_ok=True)
    model.save(model_path)
    print(f"✅ Model saved to: {model_path}")

    # Save training plot
    os.makedirs(plots_root, exist_ok=True)
    plot_path = os.path.join(plots_root, f"{model_name}.png")
    plot_training_history(history, save_path=plot_path)

    # Evaluate
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 Evaluation Results:")
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return model, run_id


def plot_training_history(history, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ Training plot saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ---------- CLI Support (Optional) ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, help="e.g. physio")
    parser.add_argument("--exercise", required=True, help="e.g. squat")
    parser.add_argument("--custom_model_name", default=None)
    parser.add_argument("--export_root", default="/content/drive/MyDrive/sdc_msc_data_analytics_project/export_data")
    parser.add_argument("--save_root", default="/content/drive/MyDrive/sdc_msc_data_analytics_project/models")
    parser.add_argument("--plots_root", default="/content/drive/MyDrive/sdc_msc_data_analytics_project/plots/train")

    args = parser.parse_args()

    data = load_and_preprocess_angles(
        export_root=args.export_root,
        category=args.category,
        exercise=args.exercise,
        window_size=30,
        stride=5
    )

    train_model_from_data(
        data,
        category=args.category,
        exercise=args.exercise,
        save_root=args.save_root,
        plots_root=args.plots_root,
        custom_model_name=args.custom_model_name
    )
