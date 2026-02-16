import os
from datetime import datetime

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


def build_model(input_dim: int) -> keras.Model:
    """
    Build a simple feed‑forward neural network for binary classification.
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    return model


def main():
    print("=" * 70)
    print("Class 5 Assignment - Tabular Classification Training")
    print("=" * 70)

    # Reproducibility
    np.random.seed(42)
    keras.utils.set_random_seed(42)

    data_path = "data/class5.csv"
    model_output_dir = "examples"
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")

    target_col = "ProdTaken"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    print(f"\nTarget distribution:\n{y.value_counts()}")

    # One‑hot encode categorical features
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"\nCategorical columns: {categorical_cols}")

    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print(f"Number of features after encoding: {X_encoded.shape[1]}")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and train model
    model = build_model(X_train_scaled.shape[1])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )
    ]

    print("\nTraining model...")
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy, test_auc = model.evaluate(
        X_test_scaled, y_test, verbose=0
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    y_pred_proba = model.predict(X_test_scaled)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_output_dir, f"class5_assignment_model_{timestamp}.h5")
    final_model_path = os.path.join(model_output_dir, "class5_assignment_model.h5")

    model.save(model_path)
    model.save(final_model_path)

    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Best model (timestamped) saved to: {model_path}")
    print(f"Latest model saved to: {final_model_path}")

    return model, history


if __name__ == "__main__":
    main()

