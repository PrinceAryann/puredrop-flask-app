from flask import Flask, render_template, request, flash, url_for, redirect
import numpy as np
import pandas as pd # Import pandas
import torch
import torch.nn as nn
import os
import joblib
import logging

# ---------------------------------------------------------------------------
# Flask Application Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
# It's crucial to set a secret key for flash messages
app.secret_key = os.environ.get('SECRET_KEY', 'a_secure_random_secret_key_for_dev')

# Configure logging
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Model Definition (PyTorch) - MUST MATCH TRAINING/FINE-TUNING
# ---------------------------------------------------------------------------
class WaterQualityNet(nn.Module):
    """Neural network for water quality prediction."""
    def __init__(self, input_size=9): # Defaulting to 9 features
        super(WaterQualityNet, self).__init__()
        # This architecture MUST match the one used in training/fine-tuning
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128), # Added BatchNorm
            nn.Dropout(0.4),    # Updated Dropout
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64), # Added BatchNorm
            nn.Dropout(0.3),    # Updated Dropout
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Ensure input is 2D for BatchNorm1d (batch_size, num_features)
        if x.ndim == 1:
            x = x.unsqueeze(0) # Add batch dimension if single sample
        return self.network(x)

# ---------------------------------------------------------------------------
# Load Model, Preprocessing Tools, and Feature Info
# ---------------------------------------------------------------------------
model = None
imputer = None
scaler = None
expected_features = None # Will be loaded from tools
input_size_loaded = None

try:
    # --- UPDATED Paths ---
    model_dir = "model"
    # Load the fine-tuned model
    model_path = os.path.join(model_dir, "water_quality_finetuned.pth")
    # Load the tools saved by the training script
    tools_path = os.path.join(model_dir, "preprocessing_tools_test.pkl")

    # Load preprocessing tools first to get feature names and input size
    if os.path.exists(tools_path):
        tools = joblib.load(tools_path)
        imputer = tools.get('imputer')
        scaler = tools.get('scaler')
        if imputer is None or scaler is None:
            raise ValueError("Imputer or Scaler not found in tools file.")

        # --- Get expected features (should be lowercase) ---
        expected_features = getattr(scaler, 'feature_names_in_', getattr(imputer, 'feature_names_in_', None))
        if expected_features is None:
             raise ValueError("Could not determine expected feature names from preprocessing tools.")
        input_size_loaded = len(expected_features)
        app.logger.info(f"Loaded preprocessing tools. Expecting features: {list(expected_features)}")
    else:
        raise FileNotFoundError(f"Preprocessing tools file not found at {tools_path}")

    # Load the model state dict
    if os.path.exists(model_path):
        # Use the input size determined from the tools
        model = WaterQualityNet(input_size=input_size_loaded)
        # Load state dict - ensure map_location for CPU/GPU flexibility if needed
        checkpoint = torch.load(model_path, map_location=torch.device('cpu')) # Load to CPU initially
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set model to evaluation mode
        app.logger.info(f"Successfully loaded fine-tuned model from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

except FileNotFoundError as e:
    app.logger.error(f"Error loading files: {e}. Ensure model and tools are in the '{model_dir}' directory.")
    model = None # Ensure model is None if loading failed
except ValueError as e:
    app.logger.error(f"Error initializing model or tools: {e}")
    model = None # Ensure model is None if loading failed
except Exception as e:
    app.logger.error(f"An unexpected error occurred during loading: {e}", exc_info=True)
    model = None # Ensure model is None on any loading error

# ---------------------------------------------------------------------------
# Prediction Function
# ---------------------------------------------------------------------------
def predict_quality(input_data_dict):
    """
    Takes a dictionary of feature values, preprocesses, and predicts potability.

    Args:
        input_data_dict (dict): Dictionary with feature names (lowercase) as keys
                                 and user-provided values.

    Returns:
        tuple: (predicted_class (int), probability (float)) or (None, None) if error.
    """
    if model is None or imputer is None or scaler is None or expected_features is None:
        app.logger.error("Model or preprocessing tools not loaded. Cannot predict.")
        return None, None

    try:
        # 1. Create DataFrame in the correct order
        # Ensure all expected features are present in the input dict
        if not all(feat in input_data_dict for feat in expected_features):
             missing = [feat for feat in expected_features if feat not in input_data_dict]
             raise ValueError(f"Missing input features: {missing}")

        # Create DataFrame with columns in the exact order expected by tools
        input_df = pd.DataFrame([input_data_dict], columns=expected_features)

        # 2. Convert to numeric (should already be done by Flask form processing)
        # Add explicit check/conversion just in case
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        if input_df.isnull().values.any():
            raise ValueError("Invalid numeric input detected after conversion.")


        # 3. Apply Imputer (learned from training data)
        input_imputed = imputer.transform(input_df)

        # 4. Apply Scaler (learned from training data)
        input_scaled = scaler.transform(input_imputed)

        # 5. Convert to PyTorch Tensor
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        # 6. Predict using the loaded model
        with torch.no_grad():
            logits = model(input_tensor)
            probability = torch.sigmoid(logits).item() # Get single probability value
            predicted_class = 1 if probability >= 0.5 else 0 # 1 for Safe, 0 for Unsafe

        return predicted_class, probability

    except ValueError as ve:
        app.logger.error(f"Value Error during prediction preprocessing: {ve}")
        flash(f"Invalid input: {ve}", "danger")
        return None, None
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}", exc_info=True)
        flash("An error occurred during prediction.", "danger")
        return None, None


# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Renders the main input form."""
    if model is None:
         flash("Model is not available. Please check server logs.", "danger")
    # Pass expected features to the template for dynamic form generation (optional)
    return render_template("index.html", features=expected_features)


@app.route("/predict", methods=["POST"])
def predict():
    """Handles form submission and displays prediction results."""
    if model is None:
         flash("Model is not available, cannot make predictions.", "danger")
         return redirect(url_for('index'))

    form_data = request.form.to_dict()
    user_inputs = {}
    missing_fields = []

    # Validate and convert form data (expecting lowercase keys from form/dict)
    if expected_features:
        for feature in expected_features:
            value = form_data.get(feature)
            if value is None or value == '':
                missing_fields.append(feature.replace('_', ' ').title())
            else:
                try:
                    # Convert to float, keys are already lowercase
                    user_inputs[feature] = float(value)
                except ValueError:
                    flash(f"Invalid input for {feature.replace('_', ' ').title()}. Please enter a number.", "danger")
                    return render_template("index.html", features=expected_features, form_data=form_data)
    else:
        flash("Model feature information not loaded correctly.", "danger")
        return redirect(url_for('index'))

    if missing_fields:
        flash(f"Missing input for: {', '.join(missing_fields)}", "warning")
        return render_template("index.html", features=expected_features, form_data=form_data)

    # Get prediction
    predicted_class, probability = predict_quality(user_inputs)

    if predicted_class is not None:
        result_text = "Safe for Consumption" if predicted_class == 1 else "Not Safe for Consumption"
        probability_percent = probability * 100
        return render_template("index.html", features=expected_features, result=result_text,
                               probability=f"{probability_percent:.2f}%", form_data=form_data)
    else:
        # Error already flashed in predict_quality
        return render_template("index.html", features=expected_features, form_data=form_data)


# ---------------------------------------------------------------------------
# Simple About Route (Example)
# ---------------------------------------------------------------------------
@app.route("/about")
def about():
    return render_template("about.html") # You'll need to create this template


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404 # You'll need to create this template


@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Server Error: {error}", exc_info=True)
    # Generic error page for users
    return render_template("500.html"), 500 # You'll need to create this template


# ---------------------------------------------------------------------------
# Entry Point (Development server) - Use Gunicorn/Waitress for production
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Use 0.0.0.0 to make it accessible on the network if needed
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
# ---------------------------------------------------------------------------