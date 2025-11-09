
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class TsunamiPredictor:
    def __init__(self):
        self.models = {}
        self.feature_names = [
            'magnitude', 'depth', 'sig', 'mmi', 'cdi', 'nst', 'dmin', 'gap',
            'latitude', 'longitude', 'Month'
        ]
        self.load_models()
    
    def load_models(self):
        """Load trained models or create demo models"""
        try:
            # Try to load pre-trained models
            models_to_load = {
                'Random Forest': 'random_forest_model.pkl',
                'XGBoost': 'xgboost_model.pkl',
                'Neural Network (MLP)': 'neural_network_(mlp)_model.pkl',
                'Gradient Boosting': 'gradient_boosting_model.pkl',
                'Logistic Regression': 'logistic_regression_model.pkl'
            }
            
            for name, filename in models_to_load.items():
                if os.path.exists(filename):
                    self.models[name] = joblib.load(filename)
                    print(f"✅ Loaded {name}")
                else:
                    print(f"⚠️  {filename} not found")
            
            # If no models loaded, create demo models
            if not self.models:
                self._create_demo_models()
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self._create_demo_models()
    
    def _create_demo_models(self):
        """Create demo models for testing"""
        print("🔄 Creating demo models...")
        
        # Create dummy training data
        np.random.seed(42)
        n_samples = 1000
        X_demo = np.random.randn(n_samples, len(self.feature_names))
        
        # Create realistic target based on magnitude and depth
        magnitude_idx = self.feature_names.index('magnitude')
        depth_idx = self.feature_names.index('depth')
        
        # Higher magnitude and shallower depth = higher tsunami probability
        tsunami_proba = (X_demo[:, magnitude_idx] * 0.4 + 
                        (1 - X_demo[:, depth_idx] * 0.3) + 
                        np.random.randn(n_samples) * 0.2)
        y_demo = (tsunami_proba > 0.5).astype(int)
        
        # Train demo models
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
        
        self.models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42)
        self.models['Logistic Regression'] = LogisticRegression(random_state=42)
        
        for name, model in self.models.items():
            model.fit(X_demo, y_demo)
        
        print("✅ Demo models created and trained")
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        try:
            # Create DataFrame with correct feature order
            processed_data = []
            for feature in self.feature_names:
                processed_data.append(input_data.get(feature, 0))
            
            return np.array([processed_data])
            
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            return None
    
    def predict(self, input_data, model_name='Random Forest'):
        """Make prediction using specified model"""
        try:
            if model_name not in self.models:
                model_name = list(self.models.keys())[0]
            
            model = self.models[model_name]
            processed_data = self.preprocess_input(input_data)
            
            if processed_data is None:
                return None
            
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(processed_data)[0, 1]
                prediction = model.predict(processed_data)[0]
            else:
                prediction = model.predict(processed_data)[0]
                probability = 0.5
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': self._get_risk_level(probability),
                'model_used': model_name,
                'confidence': self._get_confidence(probability)
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def _get_risk_level(self, probability):
        if probability >= 0.7:
            return "HIGH RISK"
        elif probability >= 0.4:
            return "MEDIUM RISK"
        else:
            return "LOW RISK"
    
    def _get_confidence(self, probability):
        distance_from_decision = abs(probability - 0.5)
        return min(100, int(distance_from_decision * 200))

# Initialize predictor
predictor = TsunamiPredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = {
            'magnitude': float(request.form.get('magnitude', 7.0)),
            'depth': float(request.form.get('depth', 50.0)),
            'sig': float(request.form.get('sig', 1000.0)),
            'mmi': float(request.form.get('mmi', 6.0)),
            'cdi': float(request.form.get('cdi', 5.0)),
            'latitude': float(request.form.get('latitude', 0.0)),
            'longitude': float(request.form.get('longitude', 0.0)),
            'nst': float(request.form.get('nst', 100.0)),
            'dmin': float(request.form.get('dmin', 1.0)),
            'gap': float(request.form.get('gap', 50.0)),
            'Month': int(request.form.get('month', 6))
        }
        
        model_name = request.form.get('model', 'Random Forest')
        
        # Make prediction
        result = predictor.predict(input_data, model_name)
        
        if result:
            # Calculate additional insights
            insights = generate_insights(input_data, result)
            result['insights'] = insights
            
            return jsonify({
                'success': True,
                'result': result,
                'input_data': input_data
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Prediction failed'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def generate_insights(input_data, result):
    """Generate insights based on input data and prediction"""
    insights = []
    
    magnitude = input_data['magnitude']
    depth = input_data['depth']
    latitude = input_data['latitude']
    longitude = input_data['longitude']
    probability = result['probability']
    
    # Magnitude insights
    if magnitude >= 8.0:
        insights.append("🌋 Very high magnitude earthquake - significant tsunami potential")
    elif magnitude >= 7.0:
        insights.append("⚠️ High magnitude earthquake - monitor tsunami potential")
    else:
        insights.append("📊 Moderate magnitude - lower tsunami risk but monitor")
    
    # Depth insights
    if depth < 30:
        insights.append("🔼 Very shallow depth - increases tsunami generation potential")
    elif depth < 70:
        insights.append("↗️ Shallow depth - favorable for tsunami generation")
    else:
        insights.append("🔽 Deeper earthquake - reduces tsunami potential")
    
    # Location insights
    if (-60 <= latitude <= 60) and ((-180 <= longitude <= -60) or (110 <= longitude <= 180)):
        insights.append("🌊 Pacific Ring of Fire location - higher tsunami probability")
    
    # Probability-based insights
    if probability > 0.7:
        insights.append("🚨 HIGH RISK: Consider immediate evacuation for coastal areas")
    elif probability > 0.4:
        insights.append("⚠️ MEDIUM RISK: Monitor closely and prepare for possible evacuation")
    else:
        insights.append("✅ LOW RISK: Continue normal monitoring")
    
    return insights

@app.route('/models')
def get_models():
    """Return available models"""
    models_list = list(predictor.models.keys())
    return jsonify({'models': models_list})

if __name__ == '__main__':
    print("🚀 Starting Tsunami Prediction Web Application...")
    print("🌐 Server will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)