from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import joblib

X, y = fetch_california_housing(return_X_y=True)
_, X_test, _, _ = train_test_split(X, y, random_state=42)

model = joblib.load("src/model.joblib")
preds = model.predict(X_test[:5])
print("Sample predictions:", preds)