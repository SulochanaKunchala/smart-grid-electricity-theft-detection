import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("dataset/smart_grid_electricity_theft_dataset_1000.csv")

X = data.drop("theft_label", axis=1)
y = data["theft_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200)

model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully")