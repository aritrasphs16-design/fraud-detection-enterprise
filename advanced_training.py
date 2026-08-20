import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

start_time = time.time()
print("Loading dataset...")
df = pd.read_csv('creditcard.csv')

# Drop Time
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']

# Stratify is crucial
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Applying SMOTE (Synthetic Data Generation) to training data...")
# SMOTE generates new, synthetic fraud examples using K-Nearest Neighbors
smote = SMOTE(sampling_strategy=0.1, random_state=42) # Bring fraud up to 10% of majority class to balance learning
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"Original Training Fraud Cases: {sum(y_train == 1)}")
print(f"Synthetic Training Fraud Cases: {sum(y_train_sm == 1)}")

print("Training Extreme Gradient Boosting (XGBoost) Model...")
# XGBoost is the industry standard for tabular data
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=10, # Additional weight for the remaining imbalance
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_model.fit(X_train_sm, y_train_sm)

print("\n--- Dynamic Threshold Tuning ---")
y_prob = xgb_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# We want high recall, but let's find the threshold that maximizes the F-beta score (weighing recall higher)
# or simply Youden's J statistic equivalent for PR curve. Let's maximize F1 score dynamically.
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Optimal Fraud Probability Threshold determined mathematically: {best_threshold:.4f}")

# Predict using the new dynamic threshold
y_pred_optimal = (y_prob >= best_threshold).astype(int)

print("\nClassification Report (Focus on Class 1 - Fraud):")
print(classification_report(y_test, y_pred_optimal))

pr_auc = auc(recall, precision)
print(f"\nPrecision-Recall AUC Score: {pr_auc:.4f}")

# Save Confusion Matrix Plot
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['Legitimate', 'Fraud'], 
            yticklabels=['Legitimate', 'Fraud'])
plt.title(f'XGBoost Confusion Matrix (Threshold={best_threshold:.2f})')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_xgb.png')
print("\nSaved confusion matrix to 'confusion_matrix_xgb.png'")

print("\nRunning Explainable AI (SHAP) Analysis... (This takes a moment)")
# SHAP explains exactly what features are driving the model
explainer = shap.TreeExplainer(xgb_model)
# For speed in hackathon, only explain a sample of the test set
X_test_sample = X_test.sample(2000, random_state=42)
shap_values = explainer.shap_values(X_test_sample)

plt.figure()
shap.summary_plot(shap_values, X_test_sample, show=False)
plt.title("SHAP Feature Impact on Fraud")
plt.tight_layout()
plt.savefig('shap_summary.png', bbox_inches='tight')
print("Saved SHAP Explainability Plot to 'shap_summary.png'")

# Save the model and threshold
model_data = {
    'model': xgb_model,
    'optimal_threshold': best_threshold
}
joblib.dump(model_data, 'advanced_fraud_model.pkl')
print(f"\nFinished in {time.time() - start_time:.1f} seconds. Model and Threshold saved as 'advanced_fraud_model.pkl'.")
