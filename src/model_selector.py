# model_selector.py

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

import pandas as pd
import numpy as np

# Accepts preprocessed data
def run_all_models(X_train, y_train, X_test, y_test, tune=False):

    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42),
        'CatBoost': cb.CatBoostClassifier(verbose=0, random_seed=42)
    }

    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 6]
        },
        'LightGBM': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1]
        },
        'CatBoost': {
            'iterations': [100, 200],
            'depth': [4, 6],
            'learning_rate': [0.05, 0.1]
        }
    }

    results = []

    for name, model in models.items():
        try:
            print(f"\n🧪 Training {name}...")

            if tune:
                print("🔍 Running GridSearchCV...")
                grid = GridSearchCV(model, param_grids[name], cv=3, n_jobs=-1, scoring='accuracy')
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                print(f"✅ Best params for {name}: {grid.best_params_}")
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            auc = roc_auc_score(y_test, y_pred)

            results.append({
                'Model': name,
                'Accuracy': acc,
                'F1 Score': f1,
                'ROC AUC': auc
            })

        except Exception as e:
            print(f"❌ {name} failed: {str(e)}")

    leaderboard = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    print("\n🏆 Model Leaderboard:")
    print(leaderboard)

    return leaderboard


