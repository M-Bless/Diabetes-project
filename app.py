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
    print(f" Error loading model or metadata: {e}")

# ======== Routes ========

@app.route('/')
def home():
    """Check API status"""
    return jsonify({
        'status': 'success',
        'message': 'Diabetes Prediction API is running ',
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
        
        data = request.get_json()
        
        #  DEBUG: Print received data
        print("\n" + "="*60)
        print(" RECEIVED FROM FLUTTER:")
        for key, value in data.items():
            print(f"   {key}: {value} (type: {type(value).__name__})")
        print("="*60)
        
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
        
        #  DEBUG: Print DataFrame
        print("\n CONVERTED TO DATAFRAME:")
        print(input_df)
        print(f"\n Data types:")
        print(input_df.dtypes)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        condition = condition_map.get(prediction, "Unknown")
        
        #  DEBUG: Print prediction
        print(f"\n PREDICTION: {prediction} ({condition})")
        print(f"PROBABILITIES:")
        for i, prob in enumerate(probabilities):
            print(f"   {condition_map[i]}: {prob*100:.1f}%")
        print("="*60 + "\n")
        
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


# ======== Run the API ========
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')