# 🧠 Auto-MachineLearning-Tabular-Workflow

A robust, interactive AutoML platform for tabular data—no code required. Upload any CSV, preprocess, train, optimize, compare models, visualize results, and make predictions on new data using a modern Streamlit UI.

---

## 🚀 Overview

**AutoML Tabular Pipeline** is a complete machine learning pipeline for tabular datasets. It allows users to:

- Upload their own data and preprocess it automatically
- Train and tune multiple top classifiers (RandomForest, XGBoost, LightGBM, CatBoost, SVM, Logistic Regression, and more)
- Handle class imbalance with SMOTE or class weights
- Compare and visualize model performance
- Predict on new, unseen data files in one click

All steps are managed through a friendly Streamlit web app—no code required!

---

## 🛠️ Setup & Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/automl-tabular-pipeline.git
    cd automl-tabular-pipeline
    ```

2. **Create a virtual environment and activate it:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    _Required libraries include: `streamlit`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `optuna`, `imbalanced-learn`, `seaborn`, etc._

---

## 🚦 How To Use

### 2. **Train Models on Your Data**
- **Upload your CSV file** (sample datasets are in the `/data` folder)
- **Select the target column** (what you want to predict)
- **Click "Run AutoML Pipeline"**

The app will:
- Preprocess and encode your data
- Handle class imbalance
- Train, tune, and compare multiple models automatically
- Display a leaderboard, model stats, and interactive charts

---

### 3. **Predict on New Data**
- Go to **Phase 4: Predict on New Data** in the app
- Upload a new CSV file (must have the same columns/format as your training data)

The app will:
- Apply the saved preprocessor
- Run your best model on this data
- Show predictions and allow you to download them as a CSV

---

## 📊 Features & Phases

- **Phase 1: Data Preprocessing**
  - Upload CSV, handle missing data, encode and scale features

- **Phase 2: Model Training & Tuning**
  - Feature selection (optional)
  - Train multiple classifiers, optimize hyperparameters with Optuna
  - Handle class imbalance (SMOTE/class_weight)

- **Phase 3: Model Evaluation**
  - Rank models by Accuracy, F1, ROC AUC
  - Visualize confusion matrix, ROC curve, and view the classification report
  - Save the best model and preprocessor for later use

- **Phase 4: Prediction Pipeline**
  - Upload new (unseen) data for prediction
  - Pipeline applies preprocessing and model automatically
  - Download predictions as a CSV

- **Phase 5: Deployment**
  - Use the app locally or deploy it 

---

## 📝 Example Datasets

Use built-in data in the `/data` folder (`titanic.csv`, `iris_flower.csv`, etc.)

Or try public datasets like:
- [Titanic](https://www.kaggle.com/c/titanic/data)
- [UCI Adult Income](https://archive.ics.uci.edu/ml/datasets/adult)
- [Iris](https://archive.ics.uci.edu/ml/datasets/iris)
- [Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)

---

## 🧭 Usage Flow (Step-by-Step)

1. **Start the app:**
   ```bash
   streamlit run app.py

### **Train a model:**
1. **Upload a training CSV**
2. **Select your target column**
3. **Click _Run AutoML Pipeline_**
4. **Review leaderboard, confusion matrix, ROC, and download results**

---

### **Predict on new data:**
1. In **Phase 4**, upload a new CSV file (same format as your training data)
2. Get predictions, review results, and download as CSV

---

- **All results, models, and predictions are saved in the `/outputs` folder**
- **You can also use the `Phase_1.ipynb` notebook for experimentation and step-by-step analysis**

---

## 💡 Notes & Tips

- The app will prompt you if you try to predict on new data before training any model
- Outputs (models, preprocessors, leaderboards, plots) are auto-saved in `/outputs`
- For reproducibility, always keep the same columns and order for new prediction data as in your original training data
- Extend or modify model and preprocessing logic in `src/` as needed

---

## 🙌 Credits

Created by **Dimitri Montgomery**  
Powered by Streamlit, scikit-learn, Optuna, XGBoost, LightGBM, CatBoost, imbalanced-learn, seaborn, and more.

---


## 🚀 Quickstart

```bash
git clone https://github.com/your-username/automl-tabular-pipeline.git
cd automl-tabular-pipeline
pip install -r requirements.txt
streamlit run app.py
