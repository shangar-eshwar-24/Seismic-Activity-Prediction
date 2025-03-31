from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Load the trained model
with open("rf_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load preprocessing tools
scaler = StandardScaler()
label_encoder = LabelEncoder()

def preprocess_input(data):
    # Convert Magnitude_type to encoded form
    data['Magnitude_type'] = label_encoder.fit_transform([data['Magnitude_type']])[0]
    
    # Convert data into DataFrame
    df = pd.DataFrame([data])
    
    # Apply standard scaling (excluding categorical columns)
    numerical_cols = ['Latitude(deg)', 'Longitude(deg)', 'Depth(km)', 'No_of_Stations', 'Gap', 'Close', 'RMS']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        user_input = {
            'Latitude(deg)': float(request.form['latitude']),
            'Longitude(deg)': float(request.form['longitude']),
            'Depth(km)': float(request.form['depth']),
            'Magnitude_type': request.form['magnitude_type'],
            'No_of_Stations': int(request.form['no_of_stations']),
            'Gap': int(request.form['gap']),
            'Close': int(request.form['close']),
            'RMS': float(request.form['rms'])
        }

        # Preprocess input
        input_data = preprocess_input(user_input)
        
        # Make prediction
        prediction = model.predict(input_data)
        result = f"Predicted Magnitude(ergs): {prediction[0]}"
    
    except Exception as e:
        result = f"Error: {str(e)}"
    
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0" , port=5000 ,debug=True)