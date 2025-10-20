from flask import Flask, render_template, request, flash
import numpy as np
import torch
import torch.nn as nn
import os

app = Flask(__name__)
app.secret_key = 'your_super_secret_key'

# --- 1. DEFINE THE CORRECT PYTORCH MODEL ARCHITECTURE ---
# This structure now matches the layers and sizes from the .pth file error message.
class WaterQualityNet(nn.Module):
    def __init__(self):
        super(WaterQualityNet, self).__init__()
        self.network = nn.Sequential(
            # Layer 0: Linear (matches the size from the error message)
            nn.Linear(9, 128),
            # Layer 1: Activation Function
            nn.ReLU(),
            # Layer 2: Dropout (a common layer with no weights, helps explain the gap)
            nn.Dropout(0.5),
            # Layer 3: Linear (matches an expected key from the error message)
            nn.Linear(128, 64),
            # Layer 4: Activation Function
            nn.ReLU(),
            # Layer 5: Dropout
            nn.Dropout(0.5),
            # Layer 6: Linear (matches an expected key from the error message)
            nn.Linear(64, 1),
            # Layer 7: Final Activation for probability
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# --- 2. LOAD THE PYTORCH MODEL ---
model_path = os.path.join('Model', 'water_quality_prediction.pth')

# Instantiate the CORRECT model structure
model = WaterQualityNet()
# Load the saved weights into the structure
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# Set the model to evaluation mode
model.eval()


# --- Water Quality Logic (Updated for PyTorch) ---
FEATURES_MAP = {
    'ph': {'limit': 8.5, 'risk': 'Stomach irritation or metabolic issues'},
    'hardness': {'limit': 300, 'risk': 'Kidney stones or cardiovascular problems'},
    'solids': {'limit': 50000, 'risk': 'Unpleasant taste or potential contamination'},
    'chloramines': {'limit': 4, 'risk': 'Skin/eye irritation, nausea'},
    'sulfate': {'limit': 500, 'risk': 'Diarrhea or dehydration'},
    'conductivity': {'limit': 500, 'risk': 'Indicates high mineral content'},
    'organic_carbon': {'limit': 15, 'risk': 'Potential for disinfection byproducts'},
    'trihalomethanes': {'limit': 100, 'risk': 'Long-term cancer risk'},
    'turbidity': {'limit': 5, 'risk': 'May harbor bacteria and pathogens'}
}

def check_water_quality(features_dict):
    """Analyzes water quality using the PyTorch model."""
    features_list = [features_dict[key] for key in FEATURES_MAP.keys()]
    
    # Convert the list to a PyTorch Tensor
    features_tensor = torch.FloatTensor([features_list])
    
    # Get the prediction from the model
    with torch.no_grad(): # Disables gradient calculation for faster inference
        prediction_score = model(features_tensor).item()

    drinkable = prediction_score >= 0.5
    result_text = "✅ Safe to Drink" if drinkable else "❌ Not Safe to Drink"
    
    risks = []
    if not drinkable:
        for feature, props in FEATURES_MAP.items():
            if features_dict[feature] > props['limit']:
                risks.append(f"High {feature.replace('_', ' ').title()}: {props['risk']}")
    
    if not drinkable and not risks:
        risks.append("The combination of parameters suggests the water is not suitable for drinking.")
        
    return drinkable, result_text, risks

# --- Your Routes (No changes needed from here down) ---
@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_server_error(error):
    return render_template("404.html"), 500

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register")
def register():
    # You will need to create a register.html template for this
    return "Register page placeholder" 

@app.route("/login")
def login():
    # You will need to create a login.html template for this
    return "Login page placeholder"

@app.route("/analyze", methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        try:
            user_inputs = {key: float(request.form.get(key)) for key in FEATURES_MAP.keys()}
            drinkable, result, risks = check_water_quality(user_inputs)
            return render_template("analyze.html", result=result, risks=risks, drinkable=drinkable)
        except (ValueError, TypeError):
            flash("Invalid input. Please enter valid numbers for all fields.", "danger")
            return render_template("analyze.html")

    return render_template("analyze.html")

