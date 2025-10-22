from flask import Flask, render_template, request, flash
import numpy as np
import torch
import torch.nn as nn
import os
import joblib # Needed to load the scalers

app = Flask(__name__)
# SECURITY FIX: Use an environment variable for the secret key in production
app.secret_key = os.environ.get('SECRET_KEY', 'a_default_key_for_development_only')

# --- 1. DEFINE THE PYTORCH MODEL ARCHITECTURE (Matches your training script) ---
# NOTE: The Sigmoid activation at the end is unusual for regression with scaled targets,
# but we will keep it as it matches the saved model structure.
class WaterQualityNet(nn.Module):
    def __init__(self, input_size=9): # Assuming 9 input features based on FEATURES_MAP
        super(WaterQualityNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3), # Match dropout from training script
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3), # Match dropout from training script
            nn.Linear(64, 1)#,
            # Removed Sigmoid - If the model was trained with MSELoss on scaled y,
            # Sigmoid here doesn't make sense unless y was originally 0-1.
            # We will interpret the raw output after inverse scaling y.
            # If your original 'final' column *was* 0 or 1, add nn.Sigmoid() back here.
        )

    def forward(self, x):
        return self.network(x)

# --- 2. LOAD MODEL AND SCALERS ---
try:
    # Model Path (lowercase 'model' directory)
    model_path = os.path.join('model', 'water_quality_pytorch_model.pth')
    
    # Scaler Paths (assuming they are saved in the model directory)
    scaler_x_path = r'model\scaler_X.pkl'
    scaler_y_path = r'model\scaler_y.pkl'

    # Instantiate the model structure
    # We need to know the number of input features expected by the model
    # Inferring 9 features from the FEATURES_MAP keys
    num_features = len([
        'ph', 'hardness', 'solids', 'chloramines', 'sulfate', 
        'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity'
    ])
    model = WaterQualityNet(input_size=num_features)
    
    # Load the saved model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # Set the model to evaluation mode

    # Load the scalers saved during training
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    print("✅ Model and scalers loaded successfully.")

except FileNotFoundError as e:
    print(f"❌ Error loading model or scalers: {e}")
    print("Ensure 'water_quality_pytorch_model.pth', 'scaler_X.pkl', and 'scaler_y.pkl' are in the correct paths.")
    # Exit or handle appropriately if files are critical
    exit()
except Exception as e:
    print(f"❌ An unexpected error occurred during loading: {e}")
    exit()


# --- Water Quality Logic ---
# Keep this consistent with the features used during training
# The order MUST match the order expected by scaler_X and the model
FEATURE_ORDER = [
    'ph', 'hardness', 'solids', 'chloramines', 'sulfate', 
    'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity'
]

FEATURES_MAP = {
    'ph': {'limit': 8.5, 'risk': 'Stomach irritation or metabolic issues'},
    'hardness': {'limit': 300, 'risk': 'Kidney stones or cardiovascular problems'},
    'solids': {'limit': 50000, 'risk': 'Unpleasant taste or potential contamination'}, # Note: 50k TDS is extremely high, verify this limit
    'chloramines': {'limit': 4, 'risk': 'Skin/eye irritation, nausea'},
    'sulfate': {'limit': 500, 'risk': 'Diarrhea or dehydration'},
    'conductivity': {'limit': 500, 'risk': 'Indicates high mineral content'}, # Verify unit/limit
    'organic_carbon': {'limit': 15, 'risk': 'Potential for disinfection byproducts'},
    'trihalomethanes': {'limit': 100, 'risk': 'Long-term cancer risk'}, # Verify unit/limit (usually ug/L, 100 is high)
    'turbidity': {'limit': 5, 'risk': 'May harbor bacteria and pathogens'}
}

def check_water_quality(features_dict):
    """Analyzes water quality using the PyTorch model and scalers."""
    
    # --- PREPROCESSING ---
    # 1. Create feature array in the correct order
    features_list = [features_dict[key] for key in FEATURE_ORDER]
    features_np = np.array([features_list]) # Model expects a 2D array
    
    # 2. Impute missing values (using median strategy like in training)
    # Although the form requires all fields, this adds robustness
    # We need an imputer fitted on training data, ideally saved like scalers.
    # For now, we'll assume no missing values or handle simply:
    if np.isnan(features_np).any():
        # Basic handling: Replace NaN with 0 or median if available
        # Ideally, load a fitted imputer: imputer = joblib.load('imputer.pkl')
        # features_np = imputer.transform(features_np)
        print("⚠️ Warning: NaNs detected in input, replaced with 0. Consider using a fitted imputer.")
        features_np = np.nan_to_num(features_np, nan=0.0)

    # 3. Scale the features using the loaded scaler_X
    features_scaled_np = scaler_X.transform(features_np)
    
    # 4. Convert scaled features to a PyTorch Tensor
    features_tensor = torch.FloatTensor(features_scaled_np)
    
    # --- PREDICTION ---
    with torch.no_grad(): # Disables gradient calculation for inference
        prediction_scaled = model(features_tensor)

    # --- POSTPROCESSING ---
    # 1. Inverse transform the model's output using loaded scaler_y
    # Convert tensor to numpy array for scaler
    prediction_unscaled_np = scaler_y.inverse_transform(prediction_scaled.cpu().numpy())
    prediction_final_score = prediction_unscaled_np[0][0] # Get the single prediction value

    # --- INTERPRETATION ---
    # Determine drinkability based on the *unscaled* prediction score
    # **CRITICAL**: You need to define what the 'final' score means.
    # Let's assume the original 'final' column was 1 for drinkable, 0 for not.
    # We'll use 0.5 as the threshold on the *original scale* prediction. Adjust as needed.
    drinkable_threshold = 0.5
    drinkable = prediction_final_score >= drinkable_threshold
    result_text = "✅ Safe to Drink" if drinkable else "❌ Not Safe to Drink"
    
    # --- RISK ASSESSMENT --- (Uses original, unscaled user inputs)
    risks = []
    if not drinkable:
        for feature, props in FEATURES_MAP.items():
            # Check if the entered value exceeds the safe limit for that feature
            if features_dict[feature] > props['limit']:
                risks.append(f"High {feature.replace('_', ' ').title()}: {props['risk']}")
    
    # General warning if no specific high value triggered a risk message
    if not drinkable and not risks:
        risks.append(f"The predicted quality score ({prediction_final_score:.2f}) is below the drinkable threshold ({drinkable_threshold}).")
        
    return drinkable, result_text, risks

# --- Routes ---
@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_server_error(error):
    # Log the error for debugging
    app.logger.error(f"Server Error: {error}", exc_info=True)
    return render_template("404.html"), 500 # Or a dedicated 500.html

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register")
def register():
    return "Register page placeholder" # Placeholder

@app.route("/login")
def login():
    return "Login page placeholder" # Placeholder

@app.route("/analyze", methods=['GET', 'POST'])
def analyze():
    form_data = request.form if request.method == 'POST' else {}
    if request.method == 'POST':
        try:
            # Validate and convert inputs
            user_inputs = {}
            missing_fields = []
            for key in FEATURE_ORDER:
                value_str = request.form.get(key)
                if value_str is None or value_str.strip() == "":
                    missing_fields.append(key.replace('_', ' ').title())
                else:
                    user_inputs[key] = float(value_str) # Convert here

            if missing_fields:
                flash(f"Missing input for: {', '.join(missing_fields)}. Please fill all fields.", "warning")
                return render_template("analyze.html", form_data=form_data)

            # Perform analysis
            drinkable, result, risks = check_water_quality(user_inputs)
            
            # Pass results and original form data back to template
            return render_template("analyze.html", result=result, risks=risks, drinkable=drinkable, form_data=form_data)

        except (ValueError, TypeError) as e:
            app.logger.error(f"Input conversion error: {e}", exc_info=True)
            flash("Invalid input. Please ensure all fields contain valid numbers.", "danger")
            return render_template("analyze.html", form_data=form_data)
        except Exception as e:
            app.logger.error(f"Prediction error: {e}", exc_info=True)
            flash("An error occurred during analysis. Please try again later.", "danger")
            return render_template("analyze.html", form_data=form_data)

    # GET request: Just show the form
    return render_template("analyze.html", form_data=form_data)

# PRODUCTION FIX: The app.run() block should be removed for hosting.
# It's okay for local testing, but gunicorn handles it in production.
# if __name__ == "__main__":
#     app.run(debug=True)

