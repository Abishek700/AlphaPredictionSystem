import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle

from prepare_data import prepare_data


def train_random_forest_regressor():
    """
    Train Random Forest model to predict continuous alpha values.
    Evaluation uses MAE and tolerance-based accuracy.
    """

    # Load data
    X, y = prepare_data()
    y = y.astype(float)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=30,
        random_state=42,
        n_jobs=-1
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)
    preds = np.clip(preds, 0.001, 0.15)

    # Metrics
    mae = mean_absolute_error(y_test, preds)

    tolerance = 0.015
    tol_accuracy = np.mean(np.abs(preds - y_test) <= tolerance)

    print("\n===== RANDOM FOREST REGRESSOR RESULTS =====")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Accuracy within ±{tolerance}: {tol_accuracy:.3f}")

    # RETURN model & scaler
    return model, scaler


# =====================================================
# TRAIN + SAVE MODEL
# =====================================================
if __name__ == "__main__":

    model, scaler = train_random_forest_regressor()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "final_alpha_recommendation_system.pkl")

    system = {
        "stage1_model": model,
        "scaler": scaler
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(system, f)

    print(f"\n✔ Model successfully saved to:\n{MODEL_PATH}")

