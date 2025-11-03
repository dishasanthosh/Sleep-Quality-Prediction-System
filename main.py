# Sleep Quality Prediction System
# Kaggle Sleep Health and Lifestyle Dataset
# Author: Disha Santhosh

# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import shap, joblib

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv('Sleep Health and Lifestyle Dataset.csv')
print(f"Dataset Shape: {df.shape}")
print(df.head())

# ==============================
# 3. Data Cleaning & Preprocessing
# ==============================
print("\nMissing Values:\n", df.isnull().sum())

# Drop missing rows (or impute as needed)
df = df.dropna()

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes('object').columns:
    df[col] = le.fit_transform(df[col])

# ==============================
# 4. Exploratory Data Analysis
# ==============================
sns.countplot(x='Quality of Sleep', data=df)
plt.title('Distribution of Sleep Quality')
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# ==============================
# 5. Feature Engineering
# ==============================
X = df.drop('Quality of Sleep', axis=1)
y = df['Quality of Sleep']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ==============================
# 6. Model Training & Optimization
# ==============================
rf = RandomForestClassifier(random_state=42)
params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(rf, param_grid=params, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(f"Best Parameters: {grid.best_params_}")

# ==============================
# 7. Evaluation
# ==============================
y_pred = best_model.predict(X_test)

print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.show()

# ==============================
# 8. Feature Importance & Explainability
# ==============================
importances = pd.Series(best_model.feature_importances_, index=X.columns)
plt.figure(figsize=(8,6))
importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.show()

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X.columns)

# ==============================
# 9. Save Model
# ==============================
joblib.dump(best_model, 'sleep_quality_rf.pkl')
print("Model saved successfully as sleep_quality_rf.pkl")

# ==============================
# 10. Conclusion
# ==============================
# Accuracy expected ~0.88–0.93 depending on data splits.
# Key Predictors: Stress Level, Physical Activity, BMI, Age, Sleep Duration.
# This model can be extended into a Streamlit app for personalized sleep recommendations.

from modules.sleep_recommendation import get_sleep_recommendations

# Example input
sample_user = {
    'Age': 30,
    'Stress_Level': 7,
    'Physical_Activity': 2,
    'Sleep_Duration': 5.5,
    'BMI': 28
}

print("Recommendations:")
for tip in get_sleep_recommendations(sample_user):
    print("-", tip)
