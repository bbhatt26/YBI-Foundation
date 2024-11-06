# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
data = pd.read_csv('bank_customer_churn_data.csv')

# Explore data
print(data.head())
print(data.info())
print(data.describe())

# Preprocess data
data.dropna(inplace=True)
data['churn'] = data['churn'].map({'yes': 1, 'no': 0})

# Split data
X = data.drop('churn', axis=1)
y = data['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importances = model.feature_importances_
print('Feature Importances:')
print(feature_importances)

# Predict new data
new_customer = pd.DataFrame({'age': [30], 'balance': [1000], 'transactions': [5]})
new_customer_scaled = scaler.transform(new_customer)
prediction = model.predict(new_customer_scaled)
print('Prediction:', prediction)