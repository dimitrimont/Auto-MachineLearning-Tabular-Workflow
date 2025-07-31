import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from src.model_selector2 import run_all_models
from src.preprocess1 import preprocess_data
import os

st.set_page_config(page_title="AutoML Tabular Pipeline", layout="wide")

st.markdown(
    "<h1 style='color:#e84a5f;'>🧠 AutoML Tabular - ML Pipeline</h1>", unsafe_allow_html=True
)

st.markdown("### Upload your CSV dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    st.markdown("### Select Target Column")
    target_col = st.selectbox(
        "Which column is your prediction target?",
        df.columns,
        index=len(df.columns) - 1 if "target" not in df.columns else df.columns.get_loc("target")
    )

    if st.button("Run AutoML Pipeline"):
        with st.spinner("Processing data and running models... Please wait. 🧑‍💻"):
            try:
                # Preprocessing (from preprocess1.py)
                X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df, target_col)
                leaderboard = run_all_models(X_train, y_train, X_test, y_test, tune_with_optuna=True, n_trials=20, use_smote=True)

                # Save the preprocessor for Phase 4 predictions
                joblib.dump(preprocessor, 'outputs/preprocessor.pkl')

                st.success("AutoML pipeline complete! 🎉")
                st.markdown("### Model Leaderboard")
                st.dataframe(leaderboard)

                # Show stats for best model
                best_model_name = leaderboard.iloc[0]['Model']
                st.markdown(f"## Best Model: {best_model_name}")

                # Load the saved best model
                best_model = joblib.load("outputs/best_model.pkl")

                # Predict on test set for stats and graphs
                y_pred = best_model.predict(X_test)
                if hasattr(best_model, "predict_proba"):
                    y_proba = best_model.predict_proba(X_test)[:, 1]
                else:
                    y_proba = None

                from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title(f"Confusion Matrix - {best_model_name}")
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("True label")
                st.pyplot(fig)

                # ROC Curve
                if y_proba is not None:
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    fig2, ax2 = plt.subplots()
                    ax2.plot(fpr, tpr, label=f"{best_model_name} (AUC = {roc_auc:.2f})")
                    ax2.plot([0, 1], [0, 1], "k--", label="Random")
                    ax2.set_xlabel("False Positive Rate")
                    ax2.set_ylabel("True Positive Rate")
                    ax2.set_title(f"ROC Curve - {best_model_name}")
                    ax2.legend()
                    st.pyplot(fig2)

                # Classification report
                report = classification_report(y_test, y_pred, output_dict=True)
                st.markdown("#### Classification Report")
                st.dataframe(pd.DataFrame(report).transpose())

            except Exception as e:
                st.error(f"An error occurred during AutoML processing: {e}")
                st.stop()
st.markdown("")
st.markdown("")
st.markdown("---")
st.markdown("## Predict on New Data")
st.markdown("#### To run this you must first upload a CSV file and run the AutoML pipeline!")
pred_file = st.file_uploader("Upload a new CSV for prediction", key="predict_csv", type="csv")

preproc_exists = os.path.exists('outputs/preprocessor.pkl')
model_exists = os.path.exists('outputs/best_model.pkl')

if pred_file:
    if not (preproc_exists and model_exists):
        st.warning("⚠️ Please complete Step 1 first! You must train a model on your own data before you can use the prediction feature.")
    else:
        new_data = pd.read_csv(pred_file)
        st.write("New Data Preview:", new_data.head())

        # Load saved preprocessor and model
        preprocessor = joblib.load('outputs/preprocessor.pkl')
        model = joblib.load('outputs/best_model.pkl')

        # Preprocess and predict
        X_new = preprocessor.transform(new_data)
        preds = model.predict(X_new)

        # Show and offer download
        new_data['Prediction'] = preds
        st.markdown("### Predictions")
        st.dataframe(new_data)

        # Download link
        csv_out = new_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions as CSV", data=csv_out, file_name="predictions.csv", mime="text/csv")
else:
    if not (preproc_exists and model_exists):
        st.info("ℹ️ To use Phase 4 (predict on new data), please upload and train a model in Step 1 first.")

st.markdown("---")
st.markdown("Built with 🧠 by Dimitri. Powered by Streamlit + scikit-learn + Optuna + more.")
