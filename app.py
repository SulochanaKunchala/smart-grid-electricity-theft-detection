
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

model = joblib.load("model.pkl")

st.title("Smart Grid Electricity Theft Detection")

st.write("Enter Electricity Usage Details")

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
        st.success("✔ No Theft Detected")

    usage = [avg_daily_usage, peak_usage, night_usage]

    fig, ax = plt.subplots()
    ax.bar(["Average", "Peak", "Night"], usage)
    ax.set_title("Electricity Usage Pattern")

    st.pyplot(fig)