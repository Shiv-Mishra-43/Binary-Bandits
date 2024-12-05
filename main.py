from flask import Flask, jsonify, request
import pandas as pd
import requests
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Global variables
MODEL_PATH = "lstm_model.h5"  # Path to the saved AI model
COVID_API_URL = "https://api.covid19api.com/dayone/country/india/status/confirmed/live"
THRESHOLD = 1000  # Example threshold for declaring a pandemic

# Function to fetch COVID-19 data
def fetch_covid_data():
    response = requests.get(COVID_API_URL)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['Date', 'Cases']]
        df = df.groupby('Date').sum().reset_index()
        df['Daily_New_Cases'] = df['Cases'].diff().fillna(0)
        return df
    else:
        return None

# Function to check pandemic threshold
def check_threshold(data, threshold):
    if data['Daily_New_Cases'].max() > threshold:
        return {"pandemic": True, "message": "Threshold crossed"}
    else:
        return {"pandemic": False, "message": "Below threshold"}

# Function to prepare data for AI prediction
def prepare_data_for_prediction(data, window_size=7):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['Daily_New_Cases'].values.reshape(-1, 1))
    X = []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
    X = np.array(X).reshape(-1, window_size, 1)
    return X, scaler

# Route: Home
@app.route('/')
def home():
    return jsonify({"message": "COVID-19 Trends and Prediction API"})

# Route: Get COVID-19 Data
@app.route('/data', methods=['GET'])
def get_data():
    data = fetch_covid_data()
    if data is not None:
        return data.to_json(orient="records")
    else:
        return jsonify({"error": "Failed to fetch data"}), 500

# Route: Get Pandemic Status
@app.route('/threshold', methods=['GET'])
def threshold_status():
    data = fetch_covid_data()
    if data is not None:
        status = check_threshold(data, THRESHOLD)
        return jsonify(status)
    else:
        return jsonify({"error": "Failed to fetch data"}), 500

# Route: Predict Future Cases
@app.route('/predict', methods=['GET'])
def predict_cases():
    data = fetch_covid_data()
    if data is not None:
        X, scaler = prepare_data_for_prediction(data)
        model = load_model(MODEL_PATH)
        future_cases_scaled = model.predict(X[-1].reshape(1, X.shape[1], 1))
        future_cases = scaler.inverse_transform(future_cases_scaled)
        return jsonify({"predicted_cases": int(future_cases[0][0])})
    else:
        return jsonify({"error": "Failed to fetch data"}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
