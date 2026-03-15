import streamlit as st
import pandas as pd
import joblib
import os

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
    # Detect features and label
    label_col = "theft" if "theft" in data.columns else data.columns[-1]
    X_test = data.drop(label_col, axis=1)
    y_test = data[label_col]
else:
    st.warning("Test dataset not found. Metrics will be unavailable.")
    X_test = pd.DataFrame(columns=model.feature_names_in_ if hasattr(model, "feature_names_in_") else [])
    y_test = None

# ------------------------------
# Dynamic user input based on model features
# ------------------------------
user_input_dict = {}
if X_test.empty and hasattr(model, "feature_names_in_"):
    feature_names = model.feature_names_in_
else:
    feature_names = X_test.columns

for feature in feature_names:
    user_input_dict[feature] = st.number_input(feature, value=0.0)

user_input_df = pd.DataFrame([user_input_dict])

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Theft"):
    try:
        prediction = model.predict(user_input_df)
        st.success(f"**Theft Prediction:** {'Yes' if prediction[0] == 1 else 'No'}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# ------------------------------
# Optional: Show model metrics
# ------------------------------
if y_test is not None and st.checkbox("Show model accuracy and confusion matrix"):
    from sklearn.metrics import accuracy_score, confusion_matrix
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    st.write("**Model Accuracy on Test Data:**", acc)
    st.write("**Confusion Matrix:**")
    st.write(cm)