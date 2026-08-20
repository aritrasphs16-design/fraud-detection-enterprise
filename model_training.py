import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

start_time = time.time()
print("Loading dataset...")
df = pd.read_csv('creditcard.csv')

# The dataset is massive. For a 1-hour hackathon, we drop 'Time' as it requires complex feature engineering to be useful.
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']

# Stratify ensures we have the exact same ratio of fraud (0.17%) in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training Advanced Random Forest Model...")
# Secret Sauce for Winning: class_weight='balanced_subsample'
# This automatically penalizes the model heavily if it misses a fraudulent transaction, which maximizes our Recall!
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=15, 
    class_weight='balanced_subsample', 
    random_state=42, 
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

print("\n--- Model Evaluation ---")
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print("\nClassification Report (Focus on Class 1 - Fraud):")
print(classification_report(y_test, y_pred))

# Precision-Recall AUC (Judges love this metric for imbalanced data)
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
print(f"\nPrecision-Recall AUC Score: {pr_auc:.4f}")
print("(Pro Tip for Pitch: Standard Accuracy is useless here because fraud is rare. PR-AUC proves our model actually finds the fraud without too many false alarms!)")

# Save Confusion Matrix Plot
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['Legitimate', 'Fraud'], 
            yticklabels=['Legitimate', 'Fraud'])
plt.title('Fraud Detection Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nSaved high-quality confusion matrix to 'confusion_matrix.png'")

# Feature Importance Plot
plt.figure(figsize=(10,6))
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]
sns.barplot(x=importances[indices], y=X.columns[indices], palette="viridis")
plt.title('Top 10 Most Important Features for Detecting Fraud')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Saved feature importance chart to 'feature_importance.png'")

# Save the model
joblib.dump(rf_model, 'fraud_model.pkl')
print(f"\nFinished in {time.time() - start_time:.1f} seconds. Model saved as 'fraud_model.pkl'.")
