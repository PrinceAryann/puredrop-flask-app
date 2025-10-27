# --------------------------------------------------------------------------
# Water Quality Model Testing Script
# --------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# 1. Model Definition (Must match train_water_quality_classifier.py and fine_tune.py)
# --------------------------------------------------------------------------
class WaterQualityNet(nn.Module):
    def __init__(self, input_size):
        super(WaterQualityNet, self).__init__()
        # This architecture MUST match the one used in training/fine-tuning
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4), # Matched from training script
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3), # Matched from training script (was 0.4, corrected based on latest train script)
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------------------------------
# 2. Paths (Adjust relative paths based on script location in /Testing)
# --------------------------------------------------------------------------
# Assuming the script is in Final Project/Testing/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Go up one level to Final Project/

MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "Dataset") # Where generated test data is
RESULTS_DIR = os.path.join(BASE_DIR, "Testing", "Results") # Output directory

# --- Select the model to test ---
# MODEL_PATH = os.path.join(MODEL_DIR, "water_quality_prediction_new_test.pth") # Pre-trained
MODEL_PATH = os.path.join(MODEL_DIR, "water_quality_prediction.pth") # Fine-tuned (Recommended)

PREPROCESS_PATH = os.path.join(MODEL_DIR, "preprocessing_tools.pkl")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_generated_hard.csv") # Using the hard test set
OUTPUT_CSV_PATH = os.path.join(RESULTS_DIR, "test_predictions_hard.csv")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Target column name in the test data
TARGET_COLUMN = 'Water_Quality_Label' # Should be numeric (0 or 1)

# --------------------------------------------------------------------------
# 3. Setup Device
# --------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------------------------------
# 4. Load Model and Preprocessing Tools
# --------------------------------------------------------------------------
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # Determine input size from checkpoint or tools
    input_size = checkpoint.get('input_size')
    if input_size is None:
         # Attempt to load tools to infer size FIRST
         try:
            tools = joblib.load(PREPROCESS_PATH)
            temp_scaler = tools.get('scaler')
            input_size = getattr(temp_scaler, 'n_features_in_', None)
            if input_size is None:
                temp_imputer = tools.get('imputer')
                input_size = getattr(temp_imputer, 'n_features_in_', None)
            if input_size is None:
                 raise ValueError("Cannot determine input size from checkpoint or tools.")
            print(f"   Inferred input size {input_size} from preprocessing tools.")
         except Exception as tool_err:
             print(f"❌ Error loading tools to infer input size: {tool_err}")
             raise ValueError("Input size missing and cannot be inferred.") from tool_err
    else:
        print(f"   Input size {input_size} found in checkpoint.")

    model = WaterQualityNet(input_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch_trained = checkpoint.get('epoch', -1) + 1
    print(f"✅ Successfully loaded model from {os.path.basename(MODEL_PATH)}")
    print(f"   (Trained for {epoch_trained} epochs)")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at '{MODEL_PATH}'.")
    exit()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

try:
    tools = joblib.load(PREPROCESS_PATH)
    imputer = tools.get('imputer')
    scaler = tools.get('scaler') # Expecting 'scaler' key
    if imputer is None or scaler is None:
        raise ValueError("'imputer' or 'scaler' key not found in .pkl file.")
    # Store expected feature names from the scaler/imputer
    expected_features = getattr(scaler, 'feature_names_in_', getattr(imputer, 'feature_names_in_', None))
    if expected_features is None:
        print("⚠️ Warning: Could not get expected feature names from tools.")
        # Fallback based on inferred input size - ASSUMES ORDER IS CORRECT
        if input_size:
             expected_features = [f'feature_{i}' for i in range(input_size)]
             print(f"   Assuming generic feature names based on input size: {expected_features}")
        else:
             print("❌ Error: Cannot determine expected features.")
             exit()

    print(f"✅ Successfully loaded preprocessing tools from {os.path.basename(PREPROCESS_PATH)}")
    print(f"   Model expects features: {list(expected_features)}")
except FileNotFoundError:
    print(f"❌ Error: Preprocessing tools not found at '{PREPROCESS_PATH}'.")
    exit()
except Exception as e:
    print(f"❌ Error loading preprocessing tools: {e}")
    exit()

# --------------------------------------------------------------------------
# 5. Load and Preprocess Test Data
# --------------------------------------------------------------------------
try:
    df = pd.read_csv(TEST_DATA_PATH)
    print(f"✅ Loaded test data '{os.path.basename(TEST_DATA_PATH)}' ({len(df)} rows)")
except FileNotFoundError:
    print(f"❌ Error: Test data file not found at '{TEST_DATA_PATH}'.")
    exit()
except Exception as e:
    print(f"❌ Error loading test data: {e}")
    exit()

# Store original columns before processing
original_columns = df.columns.tolist()

# --- Data Cleaning and Feature Selection ---
# Use lowercase for processing to match realistic generator
df.columns = [col.strip().lower() for col in df.columns]
print(f"   Test data columns (lowercase): {df.columns.tolist()}")

# Check if all expected features are in the test data (lowercase)
missing_test_features = [f for f in expected_features if f not in df.columns]
if missing_test_features:
    print(f"❌ Error: Test data is missing expected feature columns: {missing_test_features}")
    exit()

# Select only the features the model expects, in the correct order
X_test_df = df[expected_features].copy() # Ensure we work on a copy

# Convert feature columns to numeric, coercing errors
print("   Converting test features to numeric...")
for col in X_test_df.columns:
    X_test_df[col] = pd.to_numeric(X_test_df[col], errors='coerce')

# Check if target column exists for evaluation
y_true = None
target_column_lower = TARGET_COLUMN.lower()
if target_column_lower in df.columns:
    try:
        y_true = pd.to_numeric(df[target_column_lower], errors='coerce').values
        # Handle potential NaNs introduced by coerce if target wasn't numeric
        if np.isnan(y_true).any():
             print(f"⚠️ Warning: Found non-numeric or missing values in target column '{TARGET_COLUMN}'. Rows with invalid targets will be excluded from evaluation.")
             # Keep track of valid indices for later evaluation
             valid_target_indices = ~np.isnan(y_true)
             y_true = y_true[valid_target_indices] # Keep only valid numeric targets
             X_test_df = X_test_df.iloc[valid_target_indices] # Filter features accordingly
             df_original_filtered = df.iloc[valid_target_indices].copy() # Keep filtered original data for output
        else:
             y_true = y_true.astype(int) # Ensure integer type for classification labels
             df_original_filtered = df.copy() # Use all original data if target is clean

        print(f"   Found target column '{TARGET_COLUMN}' for evaluation.")
    except Exception as e:
        print(f"⚠️ Warning: Could not process target column '{TARGET_COLUMN}'. Skipping evaluation. Error: {e}")
        y_true = None
        df_original_filtered = df.copy() # Use all original data for output without evaluation
else:
    print(f"   Target column '{TARGET_COLUMN}' not found. Predictions will be generated without evaluation.")
    df_original_filtered = df.copy() # Use all original data for output

# Apply Imputation and Scaling using loaded tools
print("   Applying pre-fitted imputer and scaler...")
try:
    X_test_imp = imputer.transform(X_test_df)
    X_test_scaled = scaler.transform(X_test_imp)
    print("✅ Test data preprocessed successfully.")
except ValueError as e:
    print(f"❌ Error during imputation/scaling on test data: {e}")
    print("   This might indicate a mismatch between test data columns and the tools' expectations.")
    exit()
except Exception as e:
    print(f"❌ Unexpected error during preprocessing: {e}")
    exit()

# Convert to Tensor
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

# --------------------------------------------------------------------------
# 6. Run Predictions
# --------------------------------------------------------------------------
print("\nRunning predictions...")
model.eval()
with torch.no_grad():
    y_logits = model(X_test_tensor)
    y_probs_tensor = torch.sigmoid(y_logits)
    y_pred_classes = (y_probs_tensor > 0.5).cpu().numpy().astype(int).flatten() # Flatten for consistency
    y_probs = y_probs_tensor.cpu().numpy().flatten() # Get probabilities as well

print("✅ Predictions complete.")

# --------------------------------------------------------------------------
# 7. Evaluate if Ground Truth Available
# --------------------------------------------------------------------------
print(f"\n--- 📊 Evaluation Metrics (using '{TARGET_COLUMN}') ---")
if y_true is not None:
    if len(y_true) != len(y_pred_classes):
        print("❌ Error: Mismatch between number of true labels and predictions after handling invalid targets.")
        print(f"   True labels: {len(y_true)}, Predictions: {len(y_pred_classes)}")
        # Attempting to proceed, but results might be unreliable
    else:
        # --- UPDATED: Direct evaluation using numeric labels ---
        y_true_eval = y_true # Already numeric (0 or 1)
        y_pred_eval = y_pred_classes # Already numeric (0 or 1)

        # Calculate metrics
        acc = accuracy_score(y_true_eval, y_pred_eval)
        f1 = f1_score(y_true_eval, y_pred_eval, zero_division=0)
        precision = precision_score(y_true_eval, y_pred_eval, zero_division=0)
        recall = recall_score(y_true_eval, y_pred_eval, zero_division=0)

        print(f"Accuracy: {acc * 100:.2f}%")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"\n(Evaluation based on {len(y_true_eval)} rows with valid numeric targets)")

        print("\nClassification Report:")
        print(classification_report(
            y_true_eval,
            y_pred_eval,
            target_names=['Unsafe (0)', 'Safe (1)'], # Define names for report clarity
            zero_division=0
        ))
else:
    print(f"\n⚠️ No valid ground truth column '{TARGET_COLUMN}' found or processed. Skipping evaluation.")

# --------------------------------------------------------------------------
# 8. Save Results
# --------------------------------------------------------------------------
# Add predictions back to the (potentially filtered) original dataframe
output_df = df_original_filtered.copy()

# Ensure length match before assigning columns - important if targets were filtered
if len(output_df) == len(y_probs) == len(y_pred_classes):
    output_df['predicted_probability_safe'] = y_probs
    # Map numeric prediction back to text label for clarity in output file if desired
    output_df['predicted_label'] = np.where(y_pred_classes == 1, 'Safe', 'Unsafe')
    # Restore original column names for output consistency
    output_df.columns = [col if col not in ['predicted_probability_safe', 'predicted_label'] else col for col in output_df.columns] # Basic restore logic
    # Try to map back to original casing more robustly
    original_columns_lower = {col.lower(): col for col in original_columns}
    output_df.columns = [original_columns_lower.get(col.lower(), col) for col in output_df.columns]

    try:
        output_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\n✅ All {len(output_df)} predictions saved to {os.path.relpath(OUTPUT_CSV_PATH, BASE_DIR)}")
    except Exception as e:
        print(f"\n❌ Error saving predictions to '{OUTPUT_CSV_PATH}': {e}")
else:
     print("\n❌ Error: Mismatch in length between original data (after filtering) and predictions. Cannot save combined results.")
     print(f"   Filtered data rows: {len(output_df)}, Prediction probabilities: {len(y_probs)}, Predicted classes: {len(y_pred_classes)}")
     print("   Saving predictions separately...")
     try:
         preds_only_df = pd.DataFrame({
             'predicted_probability_safe': y_probs,
             'predicted_label_numeric': y_pred_classes
         })
         preds_only_df['predicted_label'] = np.where(preds_only_df['predicted_label_numeric'] == 1, 'Safe', 'Unsafe')
         preds_only_df.to_csv(OUTPUT_CSV_PATH, index=False)
         print(f"✅ Saved {len(preds_only_df)} predictions (only) to {os.path.relpath(OUTPUT_CSV_PATH, BASE_DIR)}")
     except Exception as e:
         print(f"\n❌ Error saving separate predictions to '{OUTPUT_CSV_PATH}': {e}")

