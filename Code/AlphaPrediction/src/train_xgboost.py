from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from prepare_data import prepare_data
import numpy as np

def train_xgboost():
    X, y = prepare_data()

    # Convert labels to integers for XGBoost
    classes = sorted(y.unique())
    class_to_int = {c: i for i, c in enumerate(classes)}
    int_to_class = {i: c for c, i in class_to_int.items()}
    y_int = y.map(class_to_int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_int,
        test_size=0.3,
        random_state=42,
        stratify=y_int
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=len(classes),
        eval_metric="mlogloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds_int = model.predict(X_test)
    preds = [int_to_class[i] for i in preds_int]
    y_test_labels = [int_to_class[i] for i in y_test]

    print("\n===== XGBOOST RESULTS =====")
    print("Accuracy:", accuracy_score(y_test_labels, preds))
    print(classification_report(y_test_labels, preds))

    return model

if __name__ == "__main__":
    train_xgboost()
