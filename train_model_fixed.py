import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load Data
print("Loading data...")
df = pd.read_csv(r'C:\Users\VICTUS\OneDrive\Documents\Data\WA_Fn-UseC_-Telco-Customer-Churn.csv')
print("Data loaded successfully!")

# 2. Data Cleaning
print("\nCleaning data...")
# 'TotalCharges' is object but should be numeric. Coerce errors to NaN and fill with 0
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
# Drop customerID as it's not useful for prediction
df = df.drop('customerID', axis=1)

# 3. Encoding Categorical Data
print("Encoding categorical data...")
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
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Handle Imbalance with SMOTE (Synthetic Minority Over-sampling Technique)
print("Handling class imbalance with SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 7. Train Model (XGBoost)
print("\nTraining XGBoost model...")
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train_resampled, y_train_resampled)
print("Training completed!")

# 8. Evaluate
print("\nEvaluating model...")
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Save Model and Column names (for the app)
print("\nSaving model and columns...")
joblib.dump(model, 'churn_model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')
print("Model and columns saved successfully!")
