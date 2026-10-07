# Forest Cover Type Classification Using Explainable AI

An end-to-end machine learning project for multi-class forest cover type classification using ensemble machine learning models, hyperparameter tuning, class balancing, and Explainable AI (SHAP).

The final selected model is a **Balanced + Tuned LightGBM classifier**, selected based primarily on **Validation Macro F1**. The trained model is integrated into an interactive **Streamlit web application** that supports manual input and CSV-based prediction along with SHAP-based explanations.

---

## Project Overview

Forest ecosystems can be characterized using cartographic and environmental features such as elevation, slope, soil types, wilderness areas, and hydrological information.

The objective of this project is to classify each observation into one of **7 forest cover types**.

The project follows an end-to-end machine learning workflow:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Train / Validation / Test Split
   ↓
Multiple ML Models
   ↓
Baseline Evaluation
   ↓
Class Balancing + Hyperparameter Tuning
   ↓
Model Comparison using Macro F1
   ↓
Final LightGBM Selection
   ↓
Test Set Evaluation
   ↓
SHAP Explainability
   ↓
Streamlit Web Application