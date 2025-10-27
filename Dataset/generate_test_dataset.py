import numpy as np
import pandas as pd
import time
import os
from sklearn.impute import SimpleImputer # Needed to handle missing values in real data

# --- Configuration ---
N_SAMPLES = 1_000_000 # Number of synthetic samples to generate
OUTPUT_FILENAME = "train_generated_realistic.csv"
MODEL_DIR = 'model'
REAL_DATA_PATH = os.path.join(MODEL_DIR, 'water_potability_real.csv') # Path to your real data
SEED = 42

# Noise and Outlier Settings
NOISE_STD_FRACTION = 0.02 # Add noise with std dev = X% of feature's std dev
OUTLIER_PROBABILITY = 0.001 # 0.1% chance for a value to be an outlier
OUTLIER_MULTIPLIER = 5 # How many std deviations away outliers should be (can vary)

np.random.seed(SEED)
start_time = time.time()

print("--- Realistic Synthetic Data Generation ---")

# --- Step 1: Analyze Real Data to Get Statistics ---
print(f"Loading real data from '{REAL_DATA_PATH}' for analysis...")
try:
    real_data = pd.read_csv(REAL_DATA_PATH)
except FileNotFoundError:
    print(f"❌ Error: Real dataset not found at '{REAL_DATA_PATH}'.")
    print("   This script requires the real data to calculate realistic statistics.")
    print("   Please download 'water_potability.csv' from Kaggle, rename it to")
    print("   'water_potability_real.csv', and place it in the 'model' folder.")
    exit()

# Define the features we want to generate (matching previous scripts)
# Ensure columns exist in the real data (handle potential case differences)
real_data_columns_lower = [col.lower() for col in real_data.columns]
target_column_real = 'potability' # Lowercase target in real data
features_to_match = ['ph', 'hardness', 'solids', 'chloramines', 'sulfate',
                     'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity']

# Check if target column exists
if target_column_real not in real_data_columns_lower:
     print(f"❌ Error: Target column '{target_column_real}' not found in real data columns: {real_data.columns.tolist()}")
     exit()

# Find the actual column names in the real data file (case-insensitive)
real_feature_columns = []
column_map = {col.lower(): col for col in real_data.columns}
for feature in features_to_match:
    if feature in column_map:
        real_feature_columns.append(column_map[feature])
    else:
        print(f"❌ Error: Feature '{feature}' not found in real data columns: {real_data.columns.tolist()}")
        exit()

# Select only the numeric feature columns from real data
real_features_df = real_data[real_feature_columns].copy()

# Convert to numeric, coercing errors
for col in real_features_df.columns:
     real_features_df[col] = pd.to_numeric(real_features_df[col], errors='coerce')

# Impute missing values using median strategy (important for covariance calc)
print("Imputing missing values in real data (using median)...")
imputer_real = SimpleImputer(strategy='median')
real_features_imputed = imputer_real.fit_transform(real_features_df)
real_features_imputed_df = pd.DataFrame(real_features_imputed, columns=real_feature_columns)

# Calculate mean vector and covariance matrix from imputed real data
print("Calculating mean vector and covariance matrix from real data...")
mean_vector = real_features_imputed_df.mean().values
# Use np.cov with rowvar=False because pandas columns are variables
covariance_matrix = np.cov(real_features_imputed_df.values, rowvar=False)
feature_std_devs = real_features_imputed_df.std().values # Needed for noise/outliers

print(f"   Mean vector calculated: {mean_vector.round(2)}")
# print(f"   Covariance matrix calculated:\n{np.round(covariance_matrix, 2)}") # Optional: print matrix

# --- Step 2: Generate Core Synthetic Data using Multivariate Normal ---
print(f"\nGenerating {N_SAMPLES} synthetic samples using multivariate normal distribution...")
# Generate data points based on the mean and covariance from real data
synthetic_features = np.random.multivariate_normal(mean_vector, covariance_matrix, N_SAMPLES)

# Clip values to somewhat realistic ranges (optional, but prevents extreme negatives)
# Example: ph between 0 and 14, others non-negative
feature_mins = [0, 0, 0, 0, 0, 0, 0, 0, 0] # Min values for each feature
feature_maxs = [14, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf] # Max for pH
synthetic_features = np.clip(synthetic_features, feature_mins, feature_maxs)

df_synthetic = pd.DataFrame(synthetic_features, columns=features_to_match) # Use lowercase standard names

# --- Step 3: Add Noise ---
print("Adding realistic noise...")
noise_std_devs = feature_std_devs * NOISE_STD_FRACTION
noise = np.random.normal(0, noise_std_devs, size=df_synthetic.shape)
df_synthetic += noise
# Re-clip after adding noise
# --- FIX: Use 'lower' instead of 'min' ---
df_synthetic = df_synthetic.clip(lower=0) # Ensure non-negativity where applicable
df_synthetic['ph'] = df_synthetic['ph'].clip(0, 14) # Clip pH specifically


# --- Step 4: Add Outliers ---
print("Adding outliers...")
for i, col in enumerate(features_to_match):
    # Determine outlier values (e.g., mean +/- multiplier * std_dev)
    outlier_low = mean_vector[i] - OUTLIER_MULTIPLIER * feature_std_devs[i]
    outlier_high = mean_vector[i] + OUTLIER_MULTIPLIER * feature_std_devs[i]

    # Decide where to put outliers
    outlier_mask = np.random.rand(N_SAMPLES) < OUTLIER_PROBABILITY

    # Decide if outlier is low or high
    low_high_choice = np.random.rand(N_SAMPLES) < 0.5 # 50% chance low, 50% high

    # Apply outliers, ensuring clipping for realistic bounds
    current_col_values = df_synthetic[col].values
    current_col_values[outlier_mask & low_high_choice] = outlier_low
    current_col_values[outlier_mask & ~low_high_choice] = outlier_high
    df_synthetic[col] = current_col_values

# Re-clip final values after outliers
# --- FIX: Use 'lower' instead of 'min' ---
df_synthetic = df_synthetic.clip(lower=0)
df_synthetic['ph'] = df_synthetic['ph'].clip(0, 14)

# --- Step 5: Add Categorical Feature (Source Type) ---
# This remains independent for simplicity, could be correlated in more advanced versions
print("Adding 'source_type' column...")
sources = ["River", "Well", "Municipal", "Industrial", "Rainwater"]
probs = [0.25, 0.25, 0.25, 0.15, 0.10]
df_synthetic['source_type'] = np.random.choice(sources, size=N_SAMPLES, p=probs)

# --- Step 6: Apply Rule-Based Labeling ---
print("Applying rule-based labeling...")
# Vectorized labeling logic
conditions = [
    (df_synthetic["ph"] < 6.5) | (df_synthetic["ph"] > 8.5),
    (df_synthetic["hardness"] > 300),
    (df_synthetic["solids"] > 50000), # Example high threshold
    (df_synthetic["chloramines"] > 4),
    (df_synthetic["sulfate"] > 500), # Example high threshold
    (df_synthetic["conductivity"] > 500), # Example high threshold
    (df_synthetic["organic_carbon"] > 15), # Example high threshold
    (df_synthetic["trihalomethanes"] > 100), # Example high threshold
    (df_synthetic["turbidity"] > 5)
]
# Sum violations across conditions (True=1, False=0)
violations = np.sum(conditions, axis=0)

# Apply final label based on violation count (0=Unsafe, 1=Safe)
# Original script: violations == 0 -> Safe (1), else -> Unsafe (0)
df_synthetic['Water_Quality_Label'] = np.where(violations == 0, 1, 0).astype(int)

# --- Step 7: Final Touches and Save ---
print("Rounding values and saving...")
# Round numeric columns for cleaner output
numeric_cols = df_synthetic.select_dtypes(include=np.number).columns.tolist()
numeric_cols.remove('Water_Quality_Label') # Don't round the label
df_synthetic[numeric_cols] = df_synthetic[numeric_cols].round(3)

# Reorder columns to have label last
cols = [col for col in df_synthetic if col != 'Water_Quality_Label'] + ['Water_Quality_Label']
df_synthetic = df_synthetic[cols]

output_path = os.path.join(MODEL_DIR, OUTPUT_FILENAME)
df_synthetic.to_csv(output_path, index=False)

end_time = time.time()
print(f"\n✅ Synthetic dataset generated successfully!")
print(f"   Saved {len(df_synthetic)} samples to '{output_path}'")
duration = end_time - start_time
print(f"   Time taken: {duration // 60:.0f} minutes {duration % 60:.2f} seconds")

