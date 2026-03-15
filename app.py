from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs("static/graphs", exist_ok=True)
app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

data = pd.read_csv("dataset/smart_grid_electricity_theft_dataset_1000.csv")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    values = [float(x) for x in request.form.values()]
    features = np.array(values).reshape(1, -1)

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Electricity Theft Detected"
    else:
        result = "Normal Electricity Usage"

    # Graph 1
    plt.figure()
    data["theft_label"].value_counts().plot(kind="bar")
    plt.title("Theft vs Normal Consumers")
    plt.savefig("static/graphs/graph1.png")
    plt.close()

    # Graph 2
    plt.figure()
    sns.histplot(data["avg_daily_usage_kwh"])
    plt.title("Average Usage Distribution")
    plt.savefig("static/graphs/graph2.png")
    plt.close()

    # Graph 3
    plt.figure()
    sns.boxplot(x="theft_label", y="peak_usage_kwh", data=data)
    plt.title("Peak Usage vs Theft")
    plt.savefig("static/graphs/graph3.png")
    plt.close()

    # Graph 4
    plt.figure()
    sns.histplot(data["anomaly_score"])
    plt.title("Anomaly Score Distribution")
    plt.savefig("static/graphs/graph4.png")
    plt.close()

    # Graph 5
    plt.figure()
    sns.heatmap(data.corr(), annot=True)
    plt.title("Correlation Heatmap")
    plt.savefig("static/graphs/graph5.png")
    plt.close()

    return render_template(
        "index.html",
        prediction_text=result,
        graph1="graphs/graph1.png",
        graph2="graphs/graph2.png",
        graph3="graphs/graph3.png",
        graph4="graphs/graph4.png",
        graph5="graphs/graph5.png"
    )


if __name__ == "__main__":
    app.run(debug=True)