from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

def get_feature_importance(model, X_train, top_n=20, feature_names=None):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 6))
    plt.title(f"Top {top_n} Feature Importances")
    
    # Plot with real names if given
    if feature_names is not None:
        names = [feature_names[i] for i in indices]
        plt.bar(range(top_n), importances[indices], align="center")
        plt.xticks(range(top_n), names, rotation=45, ha='right')
    else:
        plt.bar(range(top_n), importances[indices], align="center")
        plt.xlabel("Feature Index")
    
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.show()
    
    return indices
