
import warnings
warnings.filterwarnings('ignore')

import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from imblearn.over_sampling import SMOTE

# --------------- Classifier Names ---------------
MODELS = [
    'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost',
    'LogisticRegression', 'KNN', 'SVM', 'ExtraTrees', 'NaiveBayes'
]

# --------------- Objective Function (for Optuna) ---------------
def objective(trial, model_name, X_train, y_train):
    # Classifiers that support class_weight
    balanced_kwargs = {"class_weight": "balanced"}

    if model_name == 'RandomForest':
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 300),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            random_state=42,
            **balanced_kwargs
        )
    elif model_name == 'XGBoost':
        model = xgb.XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 300),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
    elif model_name == 'LightGBM':
        model = lgb.LGBMClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 300),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            random_state=42
        )
    elif model_name == 'CatBoost':
        model = cb.CatBoostClassifier(
            iterations=trial.suggest_int('iterations', 100, 300),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            depth=trial.suggest_int('depth', 3, 10),
            verbose=0,
            random_seed=42
        )
    elif model_name == 'LogisticRegression':
        model = LogisticRegression(max_iter=500, random_state=42, **balanced_kwargs)
    elif model_name == 'KNN':
        model = KNeighborsClassifier()
    elif model_name == 'SVM':
        model = SVC(probability=True, random_state=42)
    elif model_name == 'ExtraTrees':
        model = ExtraTreesClassifier(random_state=42, **balanced_kwargs)
    elif model_name == 'NaiveBayes':
        model = GaussianNB()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)
    preds = model.predict(X_train)
    return accuracy_score(y_train, preds)

# --------------- Main Model Training and Evaluation Pipeline ---------------
def run_all_models(X_train, y_train, X_test, y_test, tune_with_optuna=True, n_trials=20, use_smote=True):
    """
    Trains and evaluates a suite of classifiers on the dataset.
    Handles class imbalance using both class_weight and SMOTE.
    """
    results = []
    best_models = {}

    # --- (A) Apply SMOTE to the training set, if enabled ---
    if use_smote:
        print("[INFO] Applying SMOTE to the training set to balance classes...")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train

    for name in MODELS:
        try:
            print(f"\n🧪 Training {name}...")

            if tune_with_optuna:
                def optuna_objective(trial):
                    return objective(trial, name, X_train_res, y_train_res)
                study = optuna.create_study(direction='maximize')
                study.optimize(optuna_objective, n_trials=n_trials)
                best_params = study.best_params
                print(f"✅ Best params for {name}: {best_params}")

                if name == 'RandomForest':
                    model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)
                elif name == 'XGBoost':
                    model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
                elif name == 'LightGBM':
                    model = lgb.LGBMClassifier(**best_params, random_state=42)
                elif name == 'CatBoost':
                    model = cb.CatBoostClassifier(**best_params, verbose=0, random_seed=42)
                elif name == 'LogisticRegression':
                    model = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
                elif name == 'KNN':
                    model = KNeighborsClassifier()
                elif name == 'SVM':
                    model = SVC(probability=True, random_state=42)
                elif name == 'ExtraTrees':
                    model = ExtraTreesClassifier(class_weight='balanced', random_state=42)
                elif name == 'NaiveBayes':
                    model = GaussianNB()
            else:
                # Default no-tuning models
                if name == 'RandomForest':
                    model = RandomForestClassifier(class_weight='balanced', random_state=42)
                elif name == 'XGBoost':
                    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                elif name == 'LightGBM':
                    model = lgb.LGBMClassifier(random_state=42)
                elif name == 'CatBoost':
                    model = cb.CatBoostClassifier(verbose=0, random_seed=42)
                elif name == 'LogisticRegression':
                    model = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
                elif name == 'KNN':
                    model = KNeighborsClassifier()
                elif name == 'SVM':
                    model = SVC(probability=True, random_state=42)
                elif name == 'ExtraTrees':
                    model = ExtraTreesClassifier(class_weight='balanced', random_state=42)
                elif name == 'NaiveBayes':
                    model = GaussianNB()

            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            try:
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            except Exception:
                auc = 0.0

            results.append({
                'Model': name,
                'Accuracy': acc,
                'F1 Score': f1,
                'ROC AUC': auc
            })

            best_models[name] = (model, y_pred)

        except Exception as e:
            print(f"❌ {name} failed: {str(e)}")

    # --- (B) Show Results Table ---
    leaderboard = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    print("\n🏆 Model Leaderboard:")
    print(leaderboard)
    leaderboard.to_csv("outputs/model_leaderboard.csv", index=False)

    # --- (C) Save Best Model ---
    best_model_name = leaderboard.iloc[0]['Model']
    best_model = best_models[best_model_name][0]
    joblib.dump(best_model, "outputs/best_model.pkl")
    print(f"💾 Best model saved as 'best_model.pkl'")

    # --- (D) Plots ---
    y_best_pred = best_models[best_model_name][1]
    cm = confusion_matrix(y_test, y_best_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {leaderboard.iloc[0]["ROC AUC"]:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig("outputs/roc_curve.png")
        plt.close()

    joblib.dump(X_train_res, "outputs/X_train.pkl")
    joblib.dump(X_test, "outputs/X_test.pkl")
    joblib.dump(y_train_res, "outputs/y_train.pkl")
    joblib.dump(y_test, "outputs/y_test.pkl")

    

    return leaderboard

# -----------------------------------
# END OF FILE
# -----------------------------------
