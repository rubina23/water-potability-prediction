
# **Water Potability Prediction System**

# Steps:

##**1. Data Loading**

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


# Dataset load
df = pd.read_csv("water_predict.csv")

print(df.head())
print(df.shape)

## **2. Data Preprocessing**

# Handle Missing Values
df = df.fillna(df.median())

# Outlier Detection & Removal (IQR method)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop("Potability", axis=1))

# Feature Engineering (e.g., Water Quality Index)
df["quality_index"] = df["ph"] * df["Hardness"] / df["Solids"]

# Train-Test Split
X = df.drop("Potability", axis=1)
y = df["Potability"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## 3. Pipeline Creation

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

## **4. Model Training**
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# **5. Cross-Validation**
# Apply 5-fold cross-validation on the training set
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
print("CV Mean: %.3f" % scores.mean())
print("CV Std: %.3f" % scores.std())

## 6. Hyperparameter Tuning 

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5, 10]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)

## 7. Best Model Selection 

# Select the best model from GridSearchCV
best_model = grid.best_estimator_

print("Final Best Model:", best_model)

## 8. Model Performance Evaluation

# Predict on the test set
y_pred = best_model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

## 9. **Save Model**

# Save the pipeline instead of only model
import pickle
with open("water_predict_model.pkl", "wb") as f:
    pickle.dump(grid.best_estimator_, f)

"""**See rest of task on app.py file**"""
