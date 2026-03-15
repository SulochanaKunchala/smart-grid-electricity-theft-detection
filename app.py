import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.title("⚡ Smart Grid Electricity Theft Detection System")

# Load dataset
data = pd.read_csv("smart_grid_electricity_theft_dataset_1000.csv")

# Load trained model
model = joblib.load("model.pkl")

st.subheader("Dataset Preview")
st.dataframe(data.head())

# ------------------------------
# Prediction Section
# ------------------------------

st.subheader("Electricity Theft Prediction")

avg_daily_usage = st.number_input("Average Daily Usage (kWh)")
peak_usage = st.number_input("Peak Usage (kWh)")
night_usage = st.number_input("Night Usage (kWh)")
voltage = st.number_input("Voltage")
current = st.number_input("Current")
power_factor = st.number_input("Power Factor")
anomaly_score = st.number_input("Anomaly Score")

if st.button("Predict Theft"):

    features = np.array([[1, avg_daily_usage, peak_usage, night_usage,
                          voltage, current, power_factor, anomaly_score]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠ Electricity Theft Detected")
    else:
        st.success("✅ No Theft Detected")

# ------------------------------
# Graph Section
# ------------------------------

st.subheader("Data Visualization")

# Graph 1 - Theft distribution
fig1, ax1 = plt.subplots()
data['theft_label'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_title("Electricity Theft Distribution")
st.pyplot(fig1)

# Graph 2 - Average vs Peak usage
fig2, ax2 = plt.subplots()
ax2.scatter(data['avg_daily_usage_kwh'], data['peak_usage_kwh'])
ax2.set_title("Average vs Peak Usage")
st.pyplot(fig2)

# Graph 3 - Voltage distribution
fig3, ax3 = plt.subplots()
data['voltage'].plot(kind='hist', bins=20, ax=ax3)
ax3.set_title("Voltage Distribution")
st.pyplot(fig3)

# Graph 4 - Current vs Power Factor
fig4, ax4 = plt.subplots()
ax4.scatter(data['current'], data['power_factor'])
ax4.set_title("Current vs Power Factor")
st.pyplot(fig4)

# Graph 5 - Anomaly score
fig5, ax5 = plt.subplots()
data['anomaly_score'].plot(kind='box', ax=ax5)
ax5.set_title("Anomaly Score Analysis")
st.pyplot(fig5)

# ------------------------------
# Model Evaluation
# ------------------------------

st.subheader("Model Evaluation")

X = data.drop(columns=['theft_label','consumer_id'])
y = data['theft_label']

pred = model.predict(X)

cm = confusion_matrix(y, pred)

fig6, ax6 = plt.subplots()
ConfusionMatrixDisplay(cm).plot(ax=ax6)
st.pyplot(fig6)
