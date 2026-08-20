# 🛡️ FraudShield AI: Enterprise Fraud Detection

![FraudShield Banner](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.13-blue) ![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

FraudShield is a production-grade machine learning pipeline designed to detect credit card fraud in highly imbalanced environments. 

## 🧠 The Architecture

In the standard credit card fraud dataset, only **0.17% of transactions are fraudulent**. Standard AI models fail in this environment by simply predicting "Not Fraud" 100% of the time to achieve 99.8% accuracy. 

To solve this, FraudShield implements a sophisticated, multi-layered architecture:

1. **Synthetic Data Generation (SMOTE):** We utilized the Synthetic Minority Over-sampling Technique (SMOTE) to mathematically generate highly realistic synthetic fraud patterns, balancing the training environment.
2. **Cost-Sensitive Learning (XGBoost):** The core engine is an Extreme Gradient Boosting classifier. It is configured to aggressively penalize itself for missing fraudulent transactions, prioritizing Recall over standard Accuracy.
3. **Dynamic Threshold Tuning:** Instead of using a default 50% probability cutoff, the pipeline calculates the F1-Score across the entire Precision-Recall curve to mathematically derive the absolute optimal threshold for blocking hackers.
4. **Explainable AI (SHAP):** To ensure regulatory compliance and transparency, we integrated SHAP (SHapley Additive exPlanations) to break open the XGBoost black box and provide exact feature-level reasoning for every blocked transaction.

## 📊 Model Performance (Unseen Test Data)

Because accuracy is a deceptive metric on imbalanced data, we optimized strictly for Precision-Recall metrics:

* **F1-Score:** `0.84` (Balanced harmonic mean)
* **Recall:** `80.0%` (Successfully caught 80% of all fraud)
* **Precision:** `88.0%` (Very low false positive rate)
* **PR-AUC Score:** `0.85` (Area Under the Precision-Recall Curve)

## 💻 The Translation Engine (UI)

Instead of forcing users to input raw, anonymized PCA vectors (V1-V28), we built a **Translation Engine** inside a custom Streamlit dashboard. 

Users input intuitive, real-world data points:
* Purchase Category (e.g., Electronics vs. Groceries)
* Merchant Location Anomaly
* Device Fingerprint (Trusted Device vs. Tor Browser)
* Transaction Velocity (Time since last purchase)

The backend mathematically translates these human choices into the abstract PCA feature space required by the XGBoost model to evaluate risk in real-time.

## 🚀 How to Run Locally

1. **Install Requirements:**
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap streamlit matplotlib seaborn joblib
```

2. **Train the Advanced Model:**
*(Note: You must provide your own `creditcard.csv` dataset in the root folder).*
```bash
python advanced_training.py
```
*This will train the XGBoost model, optimize the threshold, and generate the SHAP and Confusion Matrix charts.*

3. **Launch the Security Dashboard:**
```bash
python -m streamlit run app.py
```
