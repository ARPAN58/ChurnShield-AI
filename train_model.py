import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load Data
df = pd.read_csv(r"C:\Users\VICTUS\OneDrive\Documents\Data\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Data Cleaning
# 'TotalCharges' is object but should be numeric. Coerce errors to NaN and fill with 0
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
# Drop customerID as it's not useful for prediction
df = df.drop('customerID', axis=1)

# 3. Encoding Categorical Data
# Identify categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()

# We create a dictionary to save encoders if we want to inverse transform later
encoders = {}
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 4. Define X (Features) and y (Target)
X = df.drop('Churn', axis=1)
y = df['Churn']

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Handle Imbalance with SMOTE (Synthetic Minority Over-sampling Technique)
# This creates synthetic samples of "Churners" so the model learns better
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 7. Train Model (XGBoost)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_resampled, y_train_resampled)

# 8. Evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Save Model and Column names (for the app)
joblib.dump(model, 'churn_model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')
print("Model and columns saved successfully!")
