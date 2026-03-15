import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------------------
# Load model
# ------------------------------
model_path = "model.pkl"
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
    st.stop()

model = joblib.load(model_path)

st.title("Smart Grid Electricity Theft Detection")
st.write("Enter the details below to predict electricity theft:")

# ------------------------------
# Load test dataset (optional)
# ------------------------------
test_csv_path = "dataset/test_data.csv"
if os.path.exists(test_csv_path):
    data = pd.read_csv(test_csv_path)
    label_col = "theft" if "theft" in data.columns else data.columns[-1]
    X_test = data.drop(label_col, axis=1)
    y_test = data[label_col]
else:
    st.warning("Test dataset not found. Metrics will be unavailable.")
    X_test = pd.DataFrame(columns=model.feature_names_in_ if hasattr(model, "feature_names_in_") else [])
    y_test = None

# ------------------------------
# Dynamic user input
# ------------------------------
user_input_dict = {}
feature_names = X_test.columns if not X_test.empty else model.feature_names_in_
for feature in feature_names:
    user_input_dict[feature] = st.number_input(feature, value=0.0)

user_input_df = pd.DataFrame([user_input_dict])

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Theft"):
    try:
        prediction = model.predict(user_input_df)
        predicted_label = "Theft" if prediction[0] == 1 else "No Theft"
        st.success(f"**Your Input Prediction:** {predicted_label}")

        # ------------------------------
        # Mini confusion for single input
        # ------------------------------
        st.write("**Mini Confusion Matrix for Your Input:**")
        st.write(pd.DataFrame(
            [[1 if prediction[0]==0 else 0, 0],
             [0, 1 if prediction[0]==1 else 0]],
            columns=["Predicted No Theft", "Predicted Theft"],
            index=["Actual No Theft?", "Actual Theft?"]
        ))

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ------------------------------
# Real model metrics
# ------------------------------
if y_test is not None and st.checkbox("Show model accuracy and confusion matrix"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    st.write("**Model Accuracy on Test Data:**", acc)
    st.write("**Confusion Matrix on Test Data:**")
    st.write(cm)