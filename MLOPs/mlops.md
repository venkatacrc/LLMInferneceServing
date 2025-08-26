# MLFlow

```python
pip install mlflow
mlflow server --host 127.0.0.1 --port 5001

# MLFlow UI
http://127.0.0.1:5001

Run this https://github.com/technoavengers/mlops_training/blob/main/Labs/Lab1/challenge/instructions.ipynb
```

# Model Serving

## Fast API

```python
pip install uvicorn
pip install fastapi
pip install <other dependencies>

uvicorn serving:app --host 0.0.0.0 --port 8080

# Testing the API using swagger docs
localhost:8080/docs

# preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import yaml

def preprocess_data(data, is_training=True, scaler_path=None):
    """
    Preprocess the data for training or inference.
    """
    ...
    return pd.DataFrame(data)

# serving.py
from fastapi import FastAPI
import pandas as pd
import joblib
from preprocessing import preprocess_data

# Load model and scaler
model_path = "shared/random_forest_model.pkl"
scaler_path = "shared/scaler.pkl"
model = joblib.load(model_path)

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Model Serving API"}

@app.post("/predict/")
def predict(data: dict):
    """
    Make predictions using the trained model.
    """
    df = pd.DataFrame([data])
    processed_data = preprocess_data(df, is_training=False, scaler_path=scaler_path)
    predictions = model.predict(processed_data)
    return {"predictions": predictions.tolist()}

# test_model.py
import requests

# API endpoint
url = "http://localhost:8080/predict"

# Test payload
payload = {

      "transaction_id": 1,
      "customer_id": 2824,
      "product_id": 843,
      "product_name": "Fridge",
      ...
}

# Send POST request
response = requests.post(url, json=payload)
print("Status Code:", response.status_code)
try:
    print("Response JSON:", response.json())
except Exception as e:
    print("Error parsing response:", e)

```

