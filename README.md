# MLOps Linear Regression Pipeline

## Description
End-to-end MLOps pipeline using scikit-learn and California Housing dataset.

## Structure
- `src/train.py`: Training
- `src/quantize.py`: Manual quantization
- `src/predict.py`: Inference
- `tests/test_train.py`: Unit tests
- `Dockerfile`: Containerization
- `.github/workflows/ci.yml`: CI/CD via GitHub Actions

## Metrics
| Metric   | Example Value |
|----------|----------------|
| R2 Score | 0.61           |
| MSE      | 0.52           |