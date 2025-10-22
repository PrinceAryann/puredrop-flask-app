from flask import Flask, render_template, request, flash
import numpy as np
import torch
import torch.nn as nn
import os
import joblib

# ---------------------------------------------------------------------------
# Flask Application Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret_key')  # Use env variable in production


# ---------------------------------------------------------------------------
# Model Definition (PyTorch)
# ---------------------------------------------------------------------------
class WaterQualityNet(nn.Module):
    """Neural network for water quality prediction."""

    def __init__(self, input_size=9):
        super(WaterQualityNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------------
# Model and Scaler Loading
# ---------------------------------------------------------------------------
try:
    model_dir = "model"
    model_path = os.path.join(model_dir, "water_quality_prediction.pth")
    scaler_x_path = os.path.join(model_dir, "scaler_X.pkl")
    scaler_y_path = os.path.join(model_dir, "scaler_y.pkl")

    num_features = len([
        "ph", "hardness", "solids", "chloramines", "sulfate",
        "conductivity", "organic_carbon", "trihalomethanes", "turbidity"
    ])

    model = WaterQualityNet(input_size=num_features)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    print("✅ Model and scalers loaded successfully.")

except FileNotFoundError as e:
    print(f"❌ Missing model or scaler file: {e}")
    exit()
except Exception as e:
    print(f"❌ Error loading model or scalers: {e}")
    exit()


# ---------------------------------------------------------------------------
# Feature Configuration
# ---------------------------------------------------------------------------
FEATURE_ORDER = [
    "ph", "hardness", "solids", "chloramines", "sulfate",
    "conductivity", "organic_carbon", "trihalomethanes", "turbidity"
]

FEATURES_MAP = {
    "ph": {"limit": 8.5, "risk": "May cause stomach irritation or metabolic issues."},
    "hardness": {"limit": 300, "risk": "Increases risk of kidney stones and heart disease."},
    "solids": {"limit": 50000, "risk": "High TDS may indicate contamination or poor taste."},
    "chloramines": {"limit": 4, "risk": "Can cause eye or skin irritation and nausea."},
    "sulfate": {"limit": 500, "risk": "May lead to diarrhea or dehydration."},
    "conductivity": {"limit": 500, "risk": "Suggests high mineral concentration."},
    "organic_carbon": {"limit": 15, "risk": "Promotes harmful disinfection byproducts."},
    "trihalomethanes": {"limit": 100, "risk": "Linked to long-term cancer risks."},
    "turbidity": {"limit": 5, "risk": "Can harbor pathogens or bacterial growth."}
}


# ---------------------------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------------------------
def check_water_quality(features_dict):
    """
    Evaluate water quality based on feature inputs using the trained PyTorch model.
    """

    # Convert user input into ordered NumPy array
    features_list = [features_dict[key] for key in FEATURE_ORDER]
    features_np = np.array([features_list])

    # Handle NaN values (basic imputation)
    if np.isnan(features_np).any():
        features_np = np.nan_to_num(features_np, nan=0.0)
        print("⚠️ NaN values found and replaced with 0.0")

    # Scale input data
    features_scaled = scaler_X.transform(features_np)
    features_tensor = torch.FloatTensor(features_scaled)

    # Model prediction
    with torch.no_grad():
        prediction_scaled = model(features_tensor)

    # Inverse transform the predicted output
    prediction_unscaled = scaler_y.inverse_transform(prediction_scaled.cpu().numpy())
    final_score = prediction_unscaled[0][0]

    # Determine drinkability
    threshold = 0.5
    drinkable = final_score >= threshold
    result_text = "✅ Safe to Drink" if drinkable else "❌ Not Safe to Drink"

    # Risk assessment for unsafe water
    risks = []
    if not drinkable:
        for feature, props in FEATURES_MAP.items():
            if features_dict[feature] > props["limit"]:
                risks.append(f"High {feature.replace('_', ' ').title()}: {props['risk']}")
        if not risks:
            risks.append(f"Quality score ({final_score:.2f}) is below the safety threshold ({threshold}).")

    return drinkable, result_text, risks


# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register")
def register():
    return "Register page placeholder"


@app.route("/login")
def login():
    return "Login page placeholder"


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    form_data = request.form if request.method == "POST" else {}

    if request.method == "POST":
        try:
            user_inputs = {}
            missing_fields = []

            # Validate and convert all inputs
            for key in FEATURE_ORDER:
                value = request.form.get(key)
                if not value or value.strip() == "":
                    missing_fields.append(key.replace("_", " ").title())
                else:
                    user_inputs[key] = float(value)

            # Handle missing inputs
            if missing_fields:
                flash(f"Missing input for: {', '.join(missing_fields)}", "warning")
                return render_template("analyze.html", form_data=form_data)

            # Generate prediction
            drinkable, result, risks = check_water_quality(user_inputs)
            return render_template("analyze.html", result=result, risks=risks,
                                   drinkable=drinkable, form_data=form_data)

        except ValueError:
            flash("Invalid input: please enter valid numerical values.", "danger")
        except Exception as e:
            app.logger.error(f"Prediction error: {e}", exc_info=True)
            flash("An internal error occurred. Please try again later.", "danger")

    return render_template("analyze.html", form_data=form_data)


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Server Error: {error}", exc_info=True)
    return render_template("404.html"), 500


# ---------------------------------------------------------------------------
# Entry Point (Development only)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
