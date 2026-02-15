import os
import json
import subprocess
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="ML Classifier Demo", layout="wide")
st.title("ML Assignment 2: Classification Model Explorer")

METRICS_PATH = "model/artifacts/metrics_table.csv"
CM_PATH = "model/artifacts/confusion_matrices.pkl"
FEATURES_PATH = "model/artifacts/feature_columns.json"
BALANCED_CSV_PATH = "model/artifacts/balanced_test_data.csv"

# Keep small for Streamlit free tier uploads
N_PER_CLASS = 200  # 200 fraud + 200 non-fraud = 400 rows


def ensure_artifacts_exist():
    """If metrics/models are missing, run training script to generate them."""
    if os.path.exists(METRICS_PATH) and os.path.exists(CM_PATH):
        return True

    st.warning("Artifacts not found (metrics/models). Running training to generate them...")

    try:
        result = subprocess.run(
            ["python", "model/train_models.py"],
            check=True,
            capture_output=True,
            text=True
        )
        st.success("Training finished. Artifacts generated.")
        st.code(result.stdout)
        return os.path.exists(METRICS_PATH) and os.path.exists(CM_PATH)

    except subprocess.CalledProcessError as e:
        st.error("Training failed. See error log below:")
        st.code((e.stdout or "") + "\n" + (e.stderr or ""))
        return False


def ensure_balanced_csv_exists():
    """
    Create a balanced fraud vs non-fraud CSV from Kaggle dataset if missing.
    Output contains ONLY features (no Class column).
    """
    if os.path.exists(BALANCED_CSV_PATH):
        return True

    try:
        import kagglehub  # needs to be in requirements.txt

        st.info("Generating balanced sample CSV (fraud vs non-fraud) for download...")

        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        csv_path = os.path.join(path, "creditcard.csv")
        df = pd.read_csv(csv_path)

        fraud = df[df["Class"] == 1]
        nonfraud = df[df["Class"] == 0]

        if len(fraud) < N_PER_CLASS:
            st.error(f"Not enough fraud samples in dataset ({len(fraud)}) to create {N_PER_CLASS} per class.")
            return False

        balanced = pd.concat(
            [
                fraud.sample(N_PER_CLASS, random_state=42),
                nonfraud.sample(N_PER_CLASS, random_state=42),
            ],
            axis=0,
        ).sample(frac=1, random_state=42).reset_index(drop=True)

        balanced_features = balanced.drop(columns=["Class"])

        os.makedirs(os.path.dirname(BALANCED_CSV_PATH), exist_ok=True)
        balanced_features.to_csv(BALANCED_CSV_PATH, index=False)

        st.success(f"Balanced CSV generated: {BALANCED_CSV_PATH}")
        return True

    except Exception as e:
        st.error("Could not generate balanced CSV. (Usually Kaggle auth or missing dependency.)")
        st.code(str(e))
        return False


# 1) Ensure artifacts exist (metrics/models)
ok = ensure_artifacts_exist()
if not ok:
    st.stop()

# 2) Load artifacts
metrics_df = pd.read_csv(METRICS_PATH)
confusion_store = joblib.load(CM_PATH)

# 3) Download button section
st.subheader("Download Sample Test Data (Balanced Fraud vs Non-Fraud)")

balanced_ok = ensure_balanced_csv_exists()
if balanced_ok and os.path.exists(BALANCED_CSV_PATH):
    balanced_df = pd.read_csv(BALANCED_CSV_PATH)

    st.caption(f"Generated with {N_PER_CLASS} fraud + {N_PER_CLASS} non-fraud rows (features only).")
    st.dataframe(balanced_df.head())

    csv_bytes = balanced_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download balanced_test_data.csv",
        data=csv_bytes,
        file_name="balanced_test_data.csv",
        mime="text/csv",
    )
else:
    st.info("Balanced sample not available yet. Check Kaggle setup and try again.")

st.divider()


# Model filename mapping (must match train_models.py output)
model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
    "XGBoost (Ensemble)": "xgboost_ensemble.pkl"
}

# Sidebar controls
st.sidebar.header("Controls")
selected_model = st.sidebar.selectbox("Select Model", list(model_files.keys()))
uploaded_file = st.sidebar.file_uploader("Upload CSV (test data only)", type=["csv"])


# Show metrics
st.subheader("Evaluation Metrics (computed on held-out test split)")
row = metrics_df[metrics_df["ML Model Name"] == selected_model].iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{row['Accuracy']:.4f}")
c1.metric("AUC", f"{row['AUC']:.4f}")
c2.metric("Precision", f"{row['Precision']:.4f}")
c2.metric("Recall", f"{row['Recall']:.4f}")
c3.metric("F1 Score", f"{row['F1']:.4f}")
c3.metric("MCC", f"{row['MCC']:.4f}")

st.divider()

# Confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_store[selected_model]
st.write(cm)

st.divider()

# Predictions section
st.subheader("Run Predictions on Uploaded CSV")

expected_cols = None
if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, "r") as f:
        expected_cols = json.load(f)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Validate columns + reorder
    if expected_cols is not None:
        missing = [c for c in expected_cols if c not in data.columns]
        if missing:
            st.error(f"Uploaded CSV missing required columns (showing up to 10): {missing[:10]}")
            st.stop()
        data = data[expected_cols]

    model_path = os.path.join("model/saved_models", model_files[selected_model])
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    model = joblib.load(model_path)
    preds = model.predict(data)

    st.write("Uploaded Data Preview")
    st.dataframe(data.head())

    st.write("Predictions (first 50)")
    st.write(preds[:50])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(data)
        st.write("Prediction Probabilities (first 10 rows)")
        st.dataframe(pd.DataFrame(probs).head(10))
else:
    st.info("Upload a CSV with the same feature columns used in training.")
