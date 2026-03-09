import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from prepare_data import prepare_data


def train_alpha_regressor():
    # Load features and target
    X, y = prepare_data()
    y = y.astype(float)  # continuous alpha values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    # XGBoost regressor
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # Clip predictions to valid alpha range
    preds = np.clip(preds, 0.001, 0.1)

    # Mean Absolute Error
    mae = mean_absolute_error(y_test, preds)

    print("\n===== ALPHA REGRESSION RESULTS =====")
    print(f"Mean Absolute Error: {mae:.4f}")

    # Tolerance-based accuracy (this is your strong % number)
    tolerance = 0.02
    within_tol = np.abs(preds - y_test) <= tolerance
    tol_accuracy = within_tol.mean()

    print(f"Accuracy within ±{tolerance}: {tol_accuracy:.3f}")

    return model


if __name__ == "__main__":
    train_alpha_regressor()
