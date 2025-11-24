# ============================================================
# DIABETES PREDICTION API (XGBoost Version)
# ============================================================
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter or other frontends

# ============================================================
# LOAD MODEL AND METADATA
# ============================================================
try:
    # Load the trained XGBoost model
    model = joblib.load('diabetes_xgboost_model.joblib')
    
    # Load metadata
    with open('xgboost_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    feature_names = metadata['feature_names']
    condition_map = metadata['condition_map']
    
    print(" XGBoost model and metadata loaded successfully.")
    print(f"   Features expected: {feature_names}")
    print(f"   Condition map: {condition_map}")
except Exception as e:
    model, metadata, feature_names, condition_map = None, None, [], {}
    print(f" Error loading model or metadata: {e}")

# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def home():
    """API status check"""
    return jsonify({
        'status': 'success',
        'message': 'Type 1 Diabetes Prediction API (XGBoost) is running',
        'version': '3.0',
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
        
        # Get JSON data from request
        data = request.get_json()
        
        # Debugging info
        print("\n" + "=" * 60)
        print(" RECEIVED FROM CLIENT:")
        for key, value in data.items():
            print(f"   {key}: {value} (type: {type(value).__name__})")
        print("=" * 60)
        
        # Validate input
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
        
        # Convert input to DataFrame (in correct column order)
        input_df = pd.DataFrame([[data[f] for f in feature_names]], columns=feature_names)
        
        # Debug: show input data
        print("\n CONVERTED TO DATAFRAME:")
        print(input_df)
        print(f"\n Data types:")
        print(input_df.dtypes)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        condition = condition_map.get(prediction, "Unknown")
        
        # Debug output
        print(f"\n PREDICTION: {prediction} ({condition})")
        print(" PROBABILITIES:")
        for i, prob in enumerate(probabilities):
            print(f"   {condition_map[i]}: {prob*100:.1f}%")
        print("=" * 60 + "\n")
        
        # Send back response
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'condition': condition,
            'probabilities': probabilities.tolist(),
            'features_used': feature_names
        })
        
    except Exception as e:
        print(f" ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }), 500

# ============================================================
# RUN THE API
# ============================================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)