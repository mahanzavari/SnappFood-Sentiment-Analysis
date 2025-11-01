from flask import Flask, request, render_template, jsonify
import sys
import os
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter

# Add src directory to Python path to import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import SentimentPredictor

app = Flask(__name__)

# Setup Prometheus metrics - automatically creates /metrics endpoint
metrics = PrometheusMetrics(app)

# Custom metric to count sentiment predictions by type
sentiment_counter = Counter(
    'sentiment_predictions_total',
    'Total number of sentiment predictions',
    ['sentiment']
)

try:
    predictor = SentimentPredictor(config_path="configs/params.yml")
except Exception as e:
    print(f"Error loading the model: {e}")
    predictor = None

@app.route('/', methods = ['GET', 'POST'])
def index():
    """Handles both displaying the form and processing the prediction."""
    if predictor is None:
        return "Model not loaded, Please check the server logs.", 500
    
    result = None
    if request.method == 'POST':
        comment_text = request.form['comment']
        if comment_text:
            result = predictor.predict(comment_text)
    
    return render_template('index.html', result=result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction. Expects JSON input."""
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid input. 'text' field is required."}), 400
    
    comment_text = data['text']
    prediction = predictor.predict(comment_text)
    
    # Record sentiment prediction metric
    sentiment_counter.labels(sentiment=prediction['sentiment']).inc()
    
    return jsonify(prediction)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for Kubernetes."""
    if predictor is None:
        return jsonify({"status": "unhealthy", "reason": "Model not loaded"}), 503
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host = 'localhost', port = 5000, debug=True) 
       
