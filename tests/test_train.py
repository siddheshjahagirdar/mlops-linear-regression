from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

def test_data_loading():
    X, y = fetch_california_housing(return_X_y=True)
    assert X.shape[0] > 0

def test_model_training():
    X, y = fetch_california_housing(return_X_y=True)
    X_train, _, y_train, _ = train_test_split(X, y, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    assert hasattr(model, "coef_")

def test_saved_model():
    model = joblib.load("src/model.joblib")
    assert isinstance(model, LinearRegression)
    assert hasattr(model, "coef_")