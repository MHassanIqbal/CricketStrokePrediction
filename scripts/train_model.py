import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report


def train_model():
    X, y = joblib.load("data/features_labels.pkl")

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred, target_names=model.classes_))

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=10)
    print(f"10-fold CV Accuracy: {np.mean(cv_scores)} ± {np.std(cv_scores)}")

    # Save the model
    joblib.dump(model, "models/random_forest_model.pkl")

if __name__ == "__main__":
    train_model()
