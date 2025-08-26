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

# Containerization

```bash
Containers are an isolated environment to run any code+ packages together. Select the container, and go to the Files tab to see what's in it.

Images are used to run containers
git clone https://github.com/docker/welcome-to-docker
cd welcome-to-docker

# create a Dockerfile
% cat Dockerfile 
# Start your image with a node base image
FROM node:22-alpine

# The /app directory should act as the main application directory
WORKDIR /app

# Copy the app package and package-lock.json file
COPY package*.json ./

# Copy local directories to the current local directory of our docker image (/app)
COPY ./src ./src
COPY ./public ./public

# Install node packages, install serve, build the app, and remove dependencies at the end
RUN npm install \
    && npm install -g serve@latest \
    && npm run build \
    && rm -fr node_modules

EXPOSE 3000

# Start the app using serve command
CMD [ "serve", "-s", "build" ]%

# build the docker
You can build an image using the following docker build command via a CLI in your project folder.

docker build -t welcome-to-docker .
Breaking down this command
The -t flag tags your image with a name. (welcome-to-docker in this case). And the . lets Docker know where it can find the Dockerfile.

# Run the container
Once the build is complete, an image will appear in the Images tab. Select the image name to see its details. Select Run to run it as a container. In the Optional settings remember to specify a port number (something like 8089).
<img width="1083" height="735" alt="image" src="https://github.com/user-attachments/assets/aaa32e90-959b-4a1e-93be-a4f98694cbc1" />

docker push
docker pull

# Run Docker Hub images
docker run -d -p 8085:8080 technoavengers/model_serving
# other examples
docker run postgres
docker run mongodb
```

