from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from prepare_data import prepare_data

def train_random_forest():
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\n===== RANDOM FOREST RESULTS =====")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    return model

if __name__ == "__main__":
    train_random_forest()
