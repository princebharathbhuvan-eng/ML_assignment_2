# ML Assignment 2: Classification Model Explorer

## a) Problem Statement
This project implements and compares **six machine learning classification models** on a single dataset and deploys an **interactive Streamlit web app** to demonstrate model selection, evaluation metrics, confusion matrix, and predictions on uploaded test CSV data.

The Streamlit app provides:
- CSV upload (test data only)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix
- Prediction outputs (+ probabilities where supported)
- Downloadable **balanced sample test dataset** (fraud vs non-fraud)

---

## b) Dataset Description (Kaggle)  [1 mark]
**Dataset Name:** Credit Card Fraud Detection  
**Source:** Kaggle (mlg-ulb/creditcardfraud)  
**Link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

### Dataset Summary
- Total records: **284,807**
- Fraud cases: **492**
- Target column: **Class** (0 = Non-fraud, 1 = Fraud)
- Features: **30 numeric features**
  - `V1` to `V28` (PCA-anonymized)
  - `Time`
  - `Amount`

### Notes
- The dataset is **highly imbalanced** (fraud is rare), so metrics such as **MCC, F1, Precision, Recall, and AUC** are more informative than Accuracy alone.

---

## c) Models Used + Evaluation Metrics  [6 marks]
Six classification models were trained on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

**Evaluation metrics computed for each model:**
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Model Comparison Table (All 6 models + all metrics)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------:|----:|----------:|-------:|---------:|----:|
| Logistic Regression | 0.9755 | 0.9721 | 0.0610 | 0.9184 | 0.1144 | 0.2332 |
| Decision Tree | 0.9989 | 0.8619 | 0.6762 | 0.7245 | 0.6995 | 0.6994 |
| kNN | 0.9995 | 0.9437 | 0.9186 | 0.8061 | 0.8587 | 0.8603 |
| Naive Bayes | 0.9923 | 0.9677 | 0.1377 | 0.6633 | 0.2281 | 0.3000 |
| Random Forest (Ensemble) | 0.9995 | 0.9515 | 0.9494 | 0.7653 | 0.8475 | 0.8522 |
| XGBoost (Ensemble) | 0.9995 | 0.9794 | 0.8723 | 0.8367 | 0.8542 | 0.8541 |

---

## Observations on Model Performance  [3 marks]

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Very high **Recall (0.9184)** → detects most fraud cases, but **very low Precision (0.0610)** → many false alarms. |
| Decision Tree | Reasonable balance, but **lower AUC (0.8619)** suggests weaker ranking ability; may overfit compared to ensembles. |
| kNN | Strong overall performance with **highest MCC (0.8603)** and **highest F1 (0.8587)**; good balance of errors but can be slower. |
| Naive Bayes | Fast baseline but weaker fraud detection quality; low precision and moderate recall → more false positives. |
| Random Forest (Ensemble) | **Highest Precision (0.9494)** → fewer false fraud alerts, with solid MCC/F1; robust and stable. |
| XGBoost (Ensemble) | **Highest AUC (0.9794)** with strong MCC/F1 and good Recall; best overall “ranking + balance” performer. |

### Key takeaway for imbalanced fraud data
Accuracy is very high for all models due to class imbalance, so **MCC/F1/Recall/AUC** provide a better comparison of fraud detection capability.

---

## Streamlit Application (Required Features)
✅ Dataset upload option (CSV – test data only)  
✅ Model selection dropdown  
✅ Display of evaluation metrics  
✅ Confusion matrix display  
✅ Prediction output (+ probabilities)

---

## How to Run Locally

### 1) Install dependencies
```bash
pip install -r requirements.txt
python model/train_models.py
### Start streamlit app 

streamlit run app.py
