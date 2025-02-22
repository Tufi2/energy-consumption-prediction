from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
from datetime import datetime

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize Flask app
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

# Load the trained model
model_path = os.path.join(BASE_DIR, 'models', 'trained_models', 'random_forest.pkl')
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.json
        
        # Extract hour and temperature
        hour = data['hour']
        temperature = data['temperature']
        
        # Convert hour to datetime to extract features
        date_obj = datetime.strptime(str(hour), "%H")
        
        # Feature engineering (MUST MATCH TRAINING)
        input_data = {
            'temperature': temperature,
            'hour': hour,
            'day_of_week': date_obj.weekday(),  # 0=Monday, 6=Sunday
            'day_period_morning': 1 if 5 <= hour < 12 else 0,
            'day_period_afternoon': 1 if 12 <= hour < 17 else 0,
            'day_period_evening': 1 if 17 <= hour < 21 else 0,
            'day_period_night': 1 if hour >= 21 or hour < 5 else 0,
        }
        
        # Create DataFrame with ALL features the model expects
        input_df = pd.DataFrame([input_data])
        
        # Ensure column order matches training
        input_df = input_df.reindex(columns=model.feature_names_in_)
        
        # Generate prediction
        prediction = model.predict(input_df)
        
        return jsonify({
            'prediction': prediction[0],
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)