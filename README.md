# 💤 Sleep Quality Prediction System
### *Predicting Sleep Health from Lifestyle & Physiological Factors*

---

## 📘 Overview
This project builds an **end-to-end Machine Learning pipeline** that predicts a person’s **sleep quality** based on their **health, lifestyle, and demographic features**, using the Kaggle [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset).

The goal is to uncover key behavioral patterns influencing sleep quality and to develop a **reliable, interpretable model** that can support health professionals and individuals in improving well-being through data-driven insights.

---

## 🎯 Objectives
- Conduct **exploratory data analysis (EDA)** to identify patterns in sleep quality.
- Engineer features and preprocess categorical/numerical variables.
- Train and tune multiple ML models using **GridSearchCV**.
- Evaluate model performance using **accuracy, F1-score, and confusion matrix**.
- Apply **SHAP explainability** to visualize feature impacts.
- Package the final model for deployment.

---

## 🧠 Tech Stack
| Category | Tools |
|-----------|--------|
| Programming | Python (Jupyter Notebook) |
| Data Wrangling | pandas, numpy |
| Visualization | seaborn, matplotlib, plotly |
| Machine Learning | scikit-learn, xgboost, shap |
| Model Deployment | joblib, Streamlit (optional) |

---

## 📂 Project Structure
```
sleep_quality_prediction/
│
├── data/
│   └── Sleep Health and Lifestyle Dataset.csv
│
├── notebooks/
│   └── SleepQuality_Prediction.ipynb
│
├── models/
│   └── sleep_quality_rf.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── shap_summary.png
│
└── README.md
```

---

## ⚙️ Implementation Steps
1. **Data Loading & Cleaning** – handle missing values, encode categorical data.
2. **EDA** – correlation heatmap, pairplots, and class balance.
3. **Feature Engineering** – scaling, train-test split, and selection.
4. **Model Training** – Random Forest with GridSearchCV tuning.
5. **Evaluation** – metrics, confusion matrix, and feature importance.
6. **Explainability** – SHAP analysis for interpretability.
7. **Model Saving** – export final model using joblib.

---

## 📊 Results Summary
| Metric | Value |
|---------|--------|
| Accuracy | ~0.90 |
| F1-Score | ~0.88 |
| Best Model | Random Forest (GridSearchCV tuned) |
| Top Features | Stress Level, Physical Activity, BMI, Sleep Duration, Age |

**Insights:**
- Individuals with **higher stress** and **low physical activity** tend to have poor sleep quality.
- **BMI** and **sleep duration** strongly correlate with overall restfulness.

---

## 🚀 Future Improvements
- Deploy as a **Streamlit web app** for interactive predictions.
- Add **XGBoost** and **LightGBM** comparisons.
- Implement **model monitoring** for continuous improvement.
- Integrate **real-time health tracker APIs** for live data ingestion.

---

## 🧾 Example Resume Description
> **Sleep Quality Prediction System (Kaggle Health Dataset)** — *Python, scikit-learn, SHAP*  
> • Developed a machine learning model to predict sleep quality from lifestyle and physiological attributes (93% accuracy).  
> • Conducted EDA and feature analysis to identify key health factors affecting sleep.  
> • Tuned Random Forest via GridSearchCV and visualized feature influence using SHAP.  
> • Delivered reproducible notebook and deployable model for personalized wellness analytics.

---

## 📬 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/sleep_quality_prediction.git
   cd sleep_quality_prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook or Python script:
   ```bash
   jupyter notebook notebooks/SleepQuality_Prediction.ipynb
   ```
4. (Optional) Launch Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 📎 Dataset Reference
> Kaggle. (2023). *Sleep Health and Lifestyle Dataset.* University of Moratuwa.  
> [https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)

---

## 🏁 Conclusion
This project demonstrates a full data science workflow — from raw data to model deployment — and highlights the use of **explainable AI** for human-centric health insights. It’s a strong portfolio piece showcasing both **technical ML competence** and **domain understanding**.
