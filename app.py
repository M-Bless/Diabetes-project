from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend apps (like Flutter)

# ======== Load Model and Metadata ========
try:
    model = joblib.load('diabetes_model.joblib')
    with open('model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    feature_names = metadata['feature_names']
    condition_map = metadata['condition_map']
except Exception as e:
    model, metadata, feature_names, condition_map = None, None, [], {}
    print(f"⚠️ Error loading model or metadata: {e}")

# ======== Routes ========

@app.route('/')
def home():
    """Check API status"""
    return jsonify({
        'status': 'success',
        'message': 'Diabetes Prediction API is running 🚀',
        'version': '1.0',
        'features_required': feature_names,
        'conditions': condition_map
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict glucose condition based on input features"""
    try:
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded properly on server.'
            }), 500

        # Parse input JSON
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided.'
            }), 400

        # Check for missing features
        missing_features = [f for f in feature_names if f not in data]
        if missing_features:
            return jsonify({
                'status': 'error',
                'message': f'Missing features: {missing_features}'
            }), 400

        # Convert input to DataFrame with correct column order
        input_df = pd.DataFrame([[data[f] for f in feature_names]], columns=feature_names)

        # Make prediction
        prediction = model.predict(input_df)[0]
        condition = condition_map.get(prediction, "Unknown")

        # Get prediction probabilities (if available)
        try:
            probabilities = model.predict_proba(input_df)[0].tolist()
        except Exception:
            probabilities = None

        # Return response
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'condition': condition,
            'probabilities': probabilities,
            'features_used': feature_names
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }), 500


# ======== Run the API ========
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')