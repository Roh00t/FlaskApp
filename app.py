from flask import Flask, request, render_template
import pickle
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

sys.modules["keras.preprocessing.text"] = tf.keras.preprocessing.text
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# --------------------------------------------------
# Define and Register Custom LSTM Class (if needed)
# --------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class CustomLSTM(tf.keras.layers.LSTM):
    @classmethod
    def from_config(cls, config):
        # Remove problematic configs
        config.pop("time_major", None)
        config.pop("zero_output_for_mask", None)
        return cls(**config)

# --------------------------------------------------
# Load Pretrained Models and Preprocessors
# --------------------------------------------------
def check_model_files():
    """Check if all required model files exist"""
    required_files = [
        "lstm_model.h5",
        "lstm_tokenizer.pkl",
        "naive_bayes.pkl",
        "random_forest.pkl",
        "vectorizer.pkl",
        "rf_vectorizer.pkl"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing model files: {missing_files}")
    return missing_files

def load_models():
    """Load all models with detailed error logging"""
    models = {}
    
    # Check for missing files first
    missing_files = check_model_files()
    if missing_files:
        return None
        
    try:
        # 1. Load Naïve Bayes Model
        logger.info("Loading Naïve Bayes model...")
        try:
            models['naive_bayes'] = pickle.load(open("naive_bayes.pkl", "rb"))
            logger.info("Naïve Bayes model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Naïve Bayes model: {str(e)}")
            models['naive_bayes'] = None
        
        # 2. Load LSTM Model
        logger.info("Loading LSTM model...")
        try:
            models['lstm'] = tf.keras.models.load_model(
                "lstm_model.h5",
                custom_objects={'CustomLSTM': CustomLSTM},
                compile=False
            )
            logger.info("LSTM model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading LSTM model: {str(e)}")
            models['lstm'] = None
            
        # Load LSTM tokenizer
        logger.info("Loading LSTM tokenizer...")
        try:
            models['lstm_tokenizer'] = pickle.load(open("lstm_tokenizer.pkl", "rb"))
            logger.info("LSTM tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading LSTM tokenizer: {str(e)}")
            models['lstm_tokenizer'] = None
        
        # 3. Load Random Forest Model
        logger.info("Loading Random Forest model...")
        try:
            models['random_forest'] = pickle.load(open("random_forest.pkl", "rb"))
            logger.info("Random Forest model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Random Forest model: {str(e)}")
            models['random_forest'] = None
        
        # 4. Load TF-IDF Vectorizers
        logger.info("Loading vectorizers...")
        try:
            models['vectorizer_nb'] = pickle.load(open("vectorizer.pkl", "rb"))
            models['rf_vectorizer'] = pickle.load(open("rf_vectorizer.pkl", "rb"))
            logger.info("Vectorizers loaded successfully")
        except Exception as e:
            logger.error(f"Error loading vectorizers: {str(e)}")
            models['vectorizer_nb'] = None
            models['rf_vectorizer'] = None
        
        return models
        
    except Exception as e:
        logger.error(f"Unexpected error in load_models: {str(e)}")
        return None

# Load models at startup
logger.info("Starting model loading process...")
models = load_models()
logger.info(f"Available models: {list(models.keys()) if models else 'None'}")

# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def preprocess_text(text: str) -> str:
    """Basic text preprocessing"""
    return text.lower().strip()

def predict_spam(email_text: str, model_choice: str):
    """Predicts spam probability using the specified model"""
    logger.info(f"Attempting prediction with model: {model_choice}")
    
    if not models:
        logger.error("No models available for prediction")
        return {"label": "Error: Models not loaded properly.", "probability": 0}
        
    email_text = preprocess_text(email_text)
    if not email_text:
        return {"label": "Please enter an email.", "probability": 0}
    
    try:
        if model_choice == "naive_bayes" and models.get('naive_bayes') and models.get('vectorizer_nb'):
            transformed_text = models['vectorizer_nb'].transform([email_text])
            prob = models['naive_bayes'].predict_proba(transformed_text)[0][1]
            logger.info("Naïve Bayes prediction successful")
        
        elif model_choice == "lstm" and models.get('lstm') and models.get('lstm_tokenizer'):
            sequences = models['lstm_tokenizer'].texts_to_sequences([email_text])
            padded_seq = pad_sequences(sequences, maxlen=100)
            prob = float(models['lstm'].predict(padded_seq)[0][0])
            logger.info("LSTM prediction successful")
        
        elif model_choice == "random_forest" and models.get('random_forest') and models.get('rf_vectorizer'):
            transformed_text = models['rf_vectorizer'].transform([email_text])
            prob = models['random_forest'].predict_proba(transformed_text)[0][1]
            logger.info("Random Forest prediction successful")
        
        else:
            logger.error(f"Required components not available for {model_choice}")
            return {"label": f"Model {model_choice} or its components not available.", "probability": 0}
        
        prediction_label = "Spam" if prob >= 0.6 else "Not Spam"
        return {"label": prediction_label, "probability": round(prob * 100, 2)}
        
    except Exception as e:
        logger.error(f"Error in prediction with {model_choice}: {str(e)}")
        return {"label": f"Error in {model_choice} prediction: {str(e)}", "probability": 0}

# --------------------------------------------------
# Flask Routes
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    predictions = {}
    model_keys = {
        "Naïve Bayes": "naive_bayes",
        "LSTM": "lstm",
        "Random Forest": "random_forest"
    }

    if request.method == "POST":
        email_text = request.form.get("email_text", "").strip()
        logger.info(f"Received prediction request for text length: {len(email_text)}")
        
        if email_text:
            valid_predictions = []
            for display_name, model_key in model_keys.items():
                prediction = predict_spam(email_text, model_key)
                predictions[display_name] = prediction
                if prediction["probability"] > 0:
                    valid_predictions.append(prediction["probability"])
            
            average_probability = sum(valid_predictions) / len(valid_predictions) if valid_predictions else 0
            model_count = len(valid_predictions)
        else:
            average_probability = 0
            model_count = 0
    else:
        average_probability = 0
        model_count = 0

    return render_template("index.html", 
                         predictions=predictions, 
                         average_probability=average_probability, 
                         model_count=model_count)

# --------------------------------------------------
# Main Entry
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)