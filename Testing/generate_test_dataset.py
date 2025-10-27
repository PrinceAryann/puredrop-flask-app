import numpy as np
import pandas as pd
import time
import os
from sklearn.impute import SimpleImputer # Needed to handle missing values in real data

# --- Configuration ---
# --- Target size for the HARD test set ---
TARGET_TEST_SIZE = 10000
# --- Generate a larger pool to select hard cases from ---
N_CANDIDATES = TARGET_TEST_SIZE * 5 # Generate 5x more candidates initially
OUTPUT_FILENAME = "test_generated_hard.csv" # Specific name for this challenging test set

# --- Define paths relative to the script's location in Scripts/ ---
# --- UPDATED: Path to real data based on new structure ---
REAL_DATA_DIR = r'Dataset'
REAL_DATA_FILENAME = 'water_potability_real.csv' # Assuming this is the real data file name
REAL_DATA_PATH = os.path.join(REAL_DATA_DIR, REAL_DATA_FILENAME)

# --- UPDATED: Path for the output file ---
OUTPUT_DIR = REAL_DATA_DIR
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

SEED = 99 # Use a DIFFERENT seed than training data generation

# Noise and Outlier Settings (Keep these for realism)
NOISE_STD_FRACTION = 0.02 # Add noise with std dev = X% of feature's std dev
OUTLIER_PROBABILITY = 0.001 # 0.1% chance for a value to be an outlier
OUTLIER_MULTIPLIER = 5 # How many std deviations away outliers should be

np.random.seed(SEED)
start_time = time.time()

print("--- Realistic HARD Synthetic Test Data Generation ---")

# --- Step 1: Analyze Real Data to Get Statistics ---
print(f"Loading real data from '{REAL_DATA_PATH}' for analysis...")
try:
    # --- Ensure the Raw Data directory exists ---
    if not os.path.exists(REAL_DATA_DIR):
        print(f"❌ Error: Real data directory not found at '{REAL_DATA_DIR}'.")
        exit()
    real_data = pd.read_csv(REAL_DATA_PATH)
    # Immediately convert columns to lower for consistency
    real_data.columns = [col.strip().lower() for col in real_data.columns]
    # Define expected feature columns based on common potability dataset (lowercase)
    expected_features = ['ph', 'hardness', 'solids', 'chloramines', 'sulfate', 'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity']
    # Check if all expected columns are present
    missing_real_features = [f for f in expected_features if f not in real_data.columns]
    if missing_real_features:
        raise ValueError(f"Real data is missing expected columns: {missing_real_features}")
    # Select only the features needed for stats calculation
    real_features_df = real_data[expected_features]

except FileNotFoundError:
    print(f"❌ Error: Real dataset not found at '{REAL_DATA_PATH}'.")
    print("   This script requires the real data to calculate realistic statistics.")
    print(f"   Please ensure '{REAL_DATA_FILENAME}' exists in the '{REAL_DATA_DIR}' folder.")
    exit()
except ValueError as e:
    print(f"❌ Error processing real data columns: {e}")
    exit()
except Exception as e:
    print(f"❌ Error loading real data: {e}")
    exit()

print("Imputing missing values in real data (using median)...")
# Use SimpleImputer on the selected features
imputer_real = SimpleImputer(strategy='median')
real_features_imputed = imputer_real.fit_transform(real_features_df)
real_features_imputed_df = pd.DataFrame(real_features_imputed, columns=expected_features) # Keep column names

print("Calculating mean vector and covariance matrix from real data...")
try:
    mean_vector = real_features_imputed_df.mean().values
    # Use numpy's cov function (expects rows=features, so transpose)
    # Ensure correct data types before calculating covariance
    covariance_matrix = np.cov(real_features_imputed_df.astype(float).T)
    # Add small value to diagonal for numerical stability if needed
    covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-6
    print(f"   Mean vector calculated: {np.round(mean_vector, 2)}")
except Exception as e:
    print(f"❌ Error calculating statistics from real data: {e}")
    exit()

# --- Step 2: Generate Candidate Samples ---
print(f"\nGenerating {N_CANDIDATES} candidate synthetic samples using multivariate normal distribution...")
synthetic_features = np.random.multivariate_normal(mean_vector, covariance_matrix, size=N_CANDIDATES)
df_synthetic = pd.DataFrame(synthetic_features, columns=expected_features)

# --- Step 3: Add Noise ---
print("Adding realistic noise...")
feature_stds = real_features_imputed_df.std().values
noise = np.random.normal(0, feature_stds * NOISE_STD_FRACTION, size=df_synthetic.shape)
df_synthetic += noise

# --- Step 4: Add Outliers ---
print("Adding outliers...")
for col_idx, col_name in enumerate(expected_features):
    is_outlier = np.random.rand(N_CANDIDATES) < OUTLIER_PROBABILITY
    # Generate outliers based on mean and std dev from real data
    outlier_values = mean_vector[col_idx] + np.random.choice([-1, 1], size=is_outlier.sum()) * OUTLIER_MULTIPLIER * feature_stds[col_idx]
    df_synthetic.loc[is_outlier, col_name] = outlier_values

# --- Step 5: Clip Values (Ensure physical constraints like non-negativity) ---
print("Clipping values to realistic ranges (e.g., non-negative)...")
# Use .clip(lower=...) which is the correct argument name
df_synthetic = df_synthetic.clip(lower=0) # Ensure non-negativity where applicable
# Specific clipping for pH if needed (though multivariate normal might keep it reasonable)
df_synthetic['ph'] = df_synthetic['ph'].clip(lower=0, upper=14)

# --- Step 6: Calculate Violations for ALL Candidates ---
print("Calculating violations for candidate samples...")
# Define conditions based on thresholds (same as before)
conditions = [
    ~((df_synthetic["ph"] >= 6.5) & (df_synthetic["ph"] <= 8.5)),
    (df_synthetic["hardness"] > 300),
    (df_synthetic["solids"] > 50000), # Example high threshold
    (df_synthetic["chloramines"] > 4),
    (df_synthetic["sulfate"] > 500), # Example high threshold
    (df_synthetic["conductivity"] > 500), # Example high threshold
    (df_synthetic["organic_carbon"] > 15), # Example high threshold
    (df_synthetic["trihalomethanes"] > 100), # Example high threshold
    (df_synthetic["turbidity"] > 5)
]
# Sum violations across conditions (True=1, False=0) for each row
violations = np.sum(conditions, axis=0)
df_synthetic['violations'] = violations # Store the count

# --- Step 7: Select Hard Cases and Safe Cases ---
print("Selecting hard cases (1 violation) and safe cases (0 violations)...")

# Select all samples with exactly 1 violation
df_hard_unsafe = df_synthetic[df_synthetic['violations'] == 1].copy()

# Select samples with 0 violations
df_safe = df_synthetic[df_synthetic['violations'] == 0].copy()

# Determine how many of each to keep for the target size
n_hard_unsafe = len(df_hard_unsafe)
n_safe = len(df_safe)
print(f"   Found {n_safe} safe samples and {n_hard_unsafe} marginal (1 violation) samples.")

target_hard_unsafe = TARGET_TEST_SIZE // 2 # Aim for 50% marginal cases
target_safe = TARGET_TEST_SIZE - target_hard_unsafe # Aim for 50% safe cases

# Adjust counts if we don't have enough of one type
if n_hard_unsafe < target_hard_unsafe:
    print(f"   Warning: Only found {n_hard_unsafe} marginal samples. Using all of them.")
    target_hard_unsafe = n_hard_unsafe
    target_safe = TARGET_TEST_SIZE - target_hard_unsafe # Increase safe samples to reach target size

if n_safe < target_safe:
    print(f"   Warning: Only found {n_safe} safe samples. Using all of them.")
    target_safe = n_safe
    # We might end up with fewer than TARGET_TEST_SIZE if both are scarce
    if n_hard_unsafe + target_safe < TARGET_TEST_SIZE:
         target_hard_unsafe = n_hard_unsafe # Use all available marginal too

# Sample the required number from each group
# Add checks to ensure sampling size is not greater than available data
n_sample_hard = min(target_hard_unsafe, len(df_hard_unsafe))
n_sample_safe = min(target_safe, len(df_safe))

if n_sample_hard > 0:
    df_final_hard = df_hard_unsafe.sample(n=n_sample_hard, random_state=SEED)
else:
    df_final_hard = pd.DataFrame(columns=df_synthetic.columns) # Empty df

if n_sample_safe > 0:
    df_final_safe = df_safe.sample(n=n_sample_safe, random_state=SEED)
else:
    df_final_safe = pd.DataFrame(columns=df_synthetic.columns) # Empty df


# Combine the selected samples
df_final_test = pd.concat([df_final_safe, df_final_hard], ignore_index=True)

# Apply final label based on violation count (0=Unsafe, 1=Safe)
df_final_test['Water_Quality_Label'] = np.where(df_final_test['violations'] == 0, 1, 0).astype(int)

# Drop the intermediate 'violations' column
df_final_test = df_final_test.drop('violations', axis=1)

# Shuffle the final dataset only if it's not empty
if not df_final_test.empty:
    df_final_test = df_final_test.sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"   Selected {len(df_final_safe)} safe samples and {len(df_final_hard)} marginal samples.")
print(f"   Final hard test set size: {len(df_final_test)}")

# --- Step 8: Final Touches and Save ---
print("Rounding values and saving...")
# Round numeric columns for cleaner output
numeric_cols = df_final_test.select_dtypes(include=np.number).columns.tolist()
if 'Water_Quality_Label' in numeric_cols:
    numeric_cols.remove('Water_Quality_Label') # Don't round the label

# Apply rounding only if there are numeric cols to round
if numeric_cols:
    df_final_test[numeric_cols] = df_final_test[numeric_cols].round(3)

# Ensure label is the last column (it should be already, but just in case)
if 'Water_Quality_Label' in df_final_test.columns:
    cols = [col for col in df_final_test if col != 'Water_Quality_Label'] + ['Water_Quality_Label']
    df_final_test = df_final_test[cols]

# --- Ensure output directory exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    df_final_test.to_csv(OUTPUT_PATH, index=False)
    end_time = time.time()
    print(f"\n✅ Hard synthetic test dataset generated successfully!")
    print(f"   Saved {len(df_final_test)} samples to '{OUTPUT_PATH}'")
    print(f"   Time taken: {(end_time - start_time):.2f} seconds")
except Exception as e:
    print(f"\n❌ Error saving dataset to '{OUTPUT_PATH}': {e}")

