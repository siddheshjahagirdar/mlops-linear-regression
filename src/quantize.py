import joblib
import numpy as np

model = joblib.load("src/model.joblib")
params = {"coef": model.coef_, "intercept": model.intercept_}
joblib.dump(params, "src/unquant_params.joblib")

coef_q = np.round(model.coef_ * 100).astype(np.uint8)
intercept_q = np.round(model.intercept_ * 100).astype(np.uint8)
quant_params = {"coef": coef_q, "intercept": intercept_q}
joblib.dump(quant_params, "src/quant_params.joblib")

# Dequantize
coef_dq = coef_q.astype(float) / 100
intercept_dq = intercept_q.astype(float) / 100
sample = np.random.rand(1, len(coef_dq))
pred = np.dot(sample, coef_dq) + intercept_dq
print("Sample inference output:", pred)