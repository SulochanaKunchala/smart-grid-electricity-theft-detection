# app.py
import os
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------
# Load Dataset
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "dataset", "smart_grid_electricity_theft_dataset_1000.csv")

try:
    data = pd.read_csv(dataset_path)
except FileNotFoundError:
    st.error(f"Dataset not found at: {dataset_path}")
    st.stop()

# -----------------------
# Prepare Data for Training
# -----------------------
# Exclude non-feature columns
X = data.drop(['consumer_id', 'theft_label'], axis=1)
y = data['theft_label']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# -----------------------
# Streamlit App
# -----------------------
st.title("Smart Grid Electricity Theft Detection")
st.write("Predict electricity theft based on consumer usage patterns.")

# Sidebar Input
st.sidebar.header("Enter Consumer Data")
avg_daily_usage = st.sidebar.number_input("Average Daily Usage (kWh)", min_value=0.0)
peak_usage = st.sidebar.number_input("Peak Usage (kWh)", min_value=0.0)
night_usage = st.sidebar.number_input("Night Usage (kWh)", min_value=0.0)
voltage = st.sidebar.number_input("Voltage", min_value=0.0)
current = st.sidebar.number_input("Current", min_value=0.0)
power_factor = st.sidebar.number_input("Power Factor", min_value=0.0)
anomaly_score = st.sidebar.number_input("Anomaly Score", min_value=0.0)

input_df = pd.DataFrame(
    [[avg_daily_usage, peak_usage, night_usage, voltage, current, power_factor, anomaly_score]],
    columns=X.columns  # ensures feature names match training
)

# Predict Button
if st.button("Predict Theft"):
    prediction = model.predict(input_df)[0]
    result = "Theft Detected ⚠️" if prediction == 1 else "No Theft ✅"
    st.subheader("Prediction Result:")
    st.write(result)

# Show Model Metrics
st.subheader("Model Accuracy")
st.write(f"{accuracy*100:.2f}%")

st.subheader("Confusion Matrix")
st.write(cm)