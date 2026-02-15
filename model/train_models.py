import os
import json
import joblib
import pandas as pd

import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# local util (your file: model/metrics_utils.py)
from metrics_utils import compute_classification_metrics


def load_kaggle_creditcardfraud():
    """
    Downloads Kaggle dataset: mlg-ulb/creditcardfraud
    Loads creditcard.csv and returns (X, y).
    """
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = os.path.join(path, "creditcard.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find creditcard.csv at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Kaggle dataset schema: features + target 'Class'
    target_col = "Class"
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' not found. Columns: {df.columns.tolist()}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def main():
    # -------------------------------
    # 1) LOAD DATA
    # -------------------------------
    X, y = load_kaggle_creditcardfraud()

    # Binary classification (0/1)
    if y.nunique() != 2:
        raise ValueError(f"Expected binary target with 2 classes, found {y.nunique()} classes.")

    # -------------------------------
    # 2) TRAIN/TEST SPLIT
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Helpful for imbalance-aware models
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    # -------------------------------
    # 3) DEFINE MODELS (6 required)
    # -------------------------------
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),

        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),

        "Naive Bayes": GaussianNB(),

        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        ),

        "XGBoost (Ensemble)": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight
        )
    }

    # -------------------------------
    # 4) OUTPUT FOLDERS
    # -------------------------------
    os.makedirs("model/saved_models", exist_ok=True)
    os.makedirs("model/artifacts", exist_ok=True)

    metrics_rows = []
    confusion_store = {}

    # Save feature columns so Streamlit can validate uploaded CSV
    feature_cols = list(X.columns)
    with open("model/artifacts/feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save a sample of test set for quick app testing (optional but useful)
    X_test.sample(n=min(2000, len(X_test)), random_state=42).to_csv(
        "model/artifacts/sample_test_data.csv", index=False
    )

    # -------------------------------
    # 5) TRAIN + EVALUATE + SAVE
    # -------------------------------
    for name, model in models.items():
        print(f"\nTraining: {name}")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # AUC requires probabilities
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        else:
            raise ValueError(f"{name} does not support predict_proba; cannot compute AUC as required.")

        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        metrics["ML Model Name"] = name
        metrics_rows.append(metrics)

        # store confusion matrix for app display
        from sklearn.metrics import confusion_matrix
        confusion_store[name] = confusion_matrix(y_test, y_pred)

        # save model with stable filenames
        safe_name = (
            name.lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("-", "_")
        )
        joblib.dump(model, f"model/saved_models/{safe_name}.pkl")

    # -------------------------------
    # 6) SAVE METRICS TABLE + CONF MATRICES
    # -------------------------------
    metrics_df = pd.DataFrame(metrics_rows)[
        ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]

    metrics_df.to_csv("model/artifacts/metrics_table.csv", index=False)
    joblib.dump(confusion_store, "model/artifacts/confusion_matrices.pkl")

    print("\n✅ Training complete.")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
