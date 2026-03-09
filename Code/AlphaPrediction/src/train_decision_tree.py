from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from prepare_data import prepare_data

def train_decision_tree():
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=4,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\n===== DECISION TREE RESULTS =====")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    return model

if __name__ == "__main__":
    train_decision_tree()
