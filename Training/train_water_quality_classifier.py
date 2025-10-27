import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os # Import os for path joining
import time # Import time for timing

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

# ---------------------------------------------------------------------------
# 1. Setup Device (GPU or CPU)
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 2. Constants
# ---------------------------------------------------------------------------
# --- FIX: Update target column name to match realistic generator ---
TARGET_COLUMN = 'Water_Quality_Label'
MODEL_DIR = 'model' # Define model directory
# --- Use os.path.join for cross-platform compatibility ---
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'water_quality_prediction_new_test.pth')
PREPROCESS_PATH = os.path.join(MODEL_DIR, 'preprocessing_tools_test.pkl')
DATA_PATH = os.path.join(MODEL_DIR, 'train_generated_realistic.csv') # Load the realistic data

# ---------------------------------------------------------------------------
# 3. Load and Preprocess Dataset
# ---------------------------------------------------------------------------
try:
    data = pd.read_csv(DATA_PATH)
    print(f"✅ Dataset '{os.path.basename(DATA_PATH)}' loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Dataset not found at '{DATA_PATH}'.")
    print("   Make sure the realistic synthetic data file exists.")
    exit()

# Clean column names (strip whitespace)
data.columns = [col.strip() for col in data.columns]

# --- FIX: Explicitly drop the non-numeric source_type column ---
if 'source_type' in data.columns:
    data = data.drop('source_type', axis=1)
    print("   Dropped 'source_type' column.")
else:
    print("   'source_type' column not found, skipping drop.")


# Check if target column exists after potential drops
if TARGET_COLUMN not in data.columns:
    print(f"❌ Target column '{TARGET_COLUMN}' not found. Available columns: {data.columns.tolist()}")
    exit()

# Separate features (X) and target (y)
X = data.drop(TARGET_COLUMN, axis=1)
y = data[TARGET_COLUMN].values

# Ensure all feature columns are numeric, coercing errors
print("   Converting feature columns to numeric...")
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce') # Coerce non-numeric to NaN


# --- FIX: Recalculate input_size AFTER dropping source_type ---
input_size = X.shape[1]
print(f"   Input size (number of features): {input_size}")
if input_size != 9:
    print(f"⚠️ Warning: Expected 9 features after dropping source_type, but found {input_size}. Check CSV columns.")
    print(f"   Features being used: {X.columns.tolist()}")


# ---------------------------------------------------------------------------
# 4. Train / Validation / Test Split
# ---------------------------------------------------------------------------
# Check if target variable has enough samples for stratification
# Ensure y is integer type for bincount
y_int = y.astype(int)
if len(np.unique(y_int)) > 1 and np.min(np.bincount(y_int)) >= 2:
    stratify_param = y
else:
    print("⚠️ Warning: Not enough samples in minority class for stratification or only one class present. Stratify set to None.")
    # Check counts if only one class
    if len(np.unique(y_int)) <= 1:
        print(f"   Class counts: {np.bincount(y_int)}")
    stratify_param = None


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=stratify_param # Split off 30% for val+test
)
# Ensure y_temp is integer for stratification in the second split
y_temp_int = y_temp.astype(int)
if stratify_param is not None and len(np.unique(y_temp_int)) > 1 and np.min(np.bincount(y_temp_int)) >= 2:
     stratify_temp = y_temp
else:
     stratify_temp = None # Don't stratify second split if first wasn't or if temp set is imbalanced

# Split the temp set into validation and test (e.g., 50/50 split of the 30%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=stratify_temp
)


print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# ---------------------------------------------------------------------------
# 5. Imputation and Scaling
# ---------------------------------------------------------------------------
print("Applying Imputation (median) and Scaling (StandardScaler)...")
# Imputer should only see numeric columns now
imputer = SimpleImputer(strategy='median')
# Use feature names during fit for better error messages if columns mismatch later
X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled = scaler.transform(X_val_imp)
X_test_scaled = scaler.transform(X_test_imp)

# ---------------------------------------------------------------------------
# 6. Convert to PyTorch Tensors
# ---------------------------------------------------------------------------
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

# Ensure y tensors are float32 for BCEWithLogitsLoss
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# ---------------------------------------------------------------------------
# 7. Datasets and DataLoaders
# ---------------------------------------------------------------------------
batch_size = 256 # Increased batch size for potentially faster training on GPU
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size*2, shuffle=False) # Larger batch for validation
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size*2, shuffle=False) # Test loader

# ---------------------------------------------------------------------------
# 8. Model Definition
# ---------------------------------------------------------------------------
class ClassificationNet(nn.Module):
    # --- Corrected __init__ to accept input_size argument ---
    def __init__(self, input_size):
        super(ClassificationNet, self).__init__()
        # Adding input validation
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError(f"input_size must be a positive integer, got {input_size}")

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1) # Output layer for binary classification
        )

    def forward(self, x):
        return self.network(x)

# ---------------------------------------------------------------------------
# 9. Initialize Model, Loss, Optimizer, Scheduler
# ---------------------------------------------------------------------------
# input_size is now correctly calculated before this step
model = ClassificationNet(input_size).to(device)

# Initialize weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(init_weights)

# Handle potential class imbalance in training set
# Ensure y_train is integer for calculations
y_train_int = y_train.astype(int)
train_safe_count = np.sum(y_train_int == 1)
train_unsafe_count = np.sum(y_train_int == 0)

# Add epsilon to prevent division by zero if one class count is zero
epsilon = 1e-6
if train_safe_count > 0 and train_unsafe_count > 0:
     pos_weight_val = train_unsafe_count / (train_safe_count + epsilon)
elif train_unsafe_count == 0 and train_safe_count > 0:
     print("⚠️ Warning: Training data contains only the positive class (1). Setting pos_weight=1.")
     pos_weight_val = 1.0
elif train_safe_count == 0 and train_unsafe_count > 0:
     print("⚠️ Warning: Training data contains only the negative class (0). Setting pos_weight=1.")
     pos_weight_val = 1.0
else: # Should not happen if data loaded
     print("⚠️ Warning: Could not calculate class counts. Setting pos_weight=1.")
     pos_weight_val = 1.0

pos_weight = torch.tensor([pos_weight_val]).to(device)
print(f"   Calculated pos_weight for BCEWithLogitsLoss: {pos_weight_val:.4f}")
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Scheduler monitors validation loss
# --- FIX: Removed verbose=True ---
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)


# ---------------------------------------------------------------------------
# 10. Training Loop with Early Stopping
# ---------------------------------------------------------------------------
num_epochs = 100 # Reduced epochs as realistic data might train faster/overfit sooner
best_val_loss = float('inf')
patience = 20 # Number of epochs to wait for improvement before stopping
epochs_no_improve = 0 # Counter for early stopping

print("\n🚀 Starting model training...\n")
training_start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    total_train_loss = 0.0

    # Training phase
    for i, (features, labels) in enumerate(train_loader):
        # features, labels already moved to device in DataLoader preparation if device='cuda'
        # If device='cpu', this moves them:
        features, labels = features.to(device), labels.to(device)


        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping can help stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item()

        # Print batch progress occasionally (e.g., every 100 batches)
        # if (i + 1) % 100 == 0:
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


    avg_train_loss = total_train_loss / len(train_loader)

    # Validation phase
    model.eval()
    total_val_loss = 0.0
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            val_outputs = model(features)
            val_loss = criterion(val_outputs, labels)
            total_val_loss += val_loss.item()

            # Get predictions (apply sigmoid and threshold)
            val_probs = torch.sigmoid(val_outputs)
            # Use .detach() before .cpu() if using GPU
            val_preds = (val_probs > 0.5).detach().cpu().numpy().astype(int)
            all_val_preds.extend(val_preds)
            all_val_labels.extend(labels.detach().cpu().numpy().astype(int))

    avg_val_loss = total_val_loss / len(val_loader)
    # Ensure labels/preds are flattened numpy arrays for sklearn metrics
    all_val_labels_np = np.array(all_val_labels).flatten()
    all_val_preds_np = np.array(all_val_preds).flatten()

    # Calculate F1 score on validation set
    val_f1 = f1_score(all_val_labels_np, all_val_preds_np, zero_division=0)


    epoch_duration = time.time() - epoch_start_time
    # Step the scheduler based on validation loss
    scheduler.step(avg_val_loss)

    # Get current learning rate AFTER scheduler step
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.1E} | Time: {epoch_duration:.2f}s")


    # Early stopping and save best model
    if avg_val_loss < best_val_loss:
        # --- FIX: Apply formatting conditionally ---
        previous_best_loss_str = f"{best_val_loss:.4f}" if best_val_loss != float('inf') else "inf"
        print(f"   Validation loss improved ({previous_best_loss_str} --> {avg_val_loss:.4f}). Saving model...")
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        # Save model checkpoint
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'input_size': input_size # Save input size with model
            }, BEST_MODEL_PATH)
        except Exception as e:
            print(f"❌ Error saving model checkpoint: {e}")

    else:
        epochs_no_improve += 1
        print(f"   Validation loss did not improve for {epochs_no_improve} epoch(s).")


    if epochs_no_improve >= patience:
        print(f"\n🛑 Early stopping triggered after epoch {epoch + 1} due to no improvement in validation loss for {patience} epochs.")
        print(f"   Best validation loss achieved: {best_val_loss:.4f}")
        break

training_duration = time.time() - training_start_time
print(f"\n✅ Training complete! Total time: {training_duration // 60:.0f}m {training_duration % 60:.2f}s")

# ---------------------------------------------------------------------------
# 11. Evaluation on Test Set
# ---------------------------------------------------------------------------
print("\nLoading best model for final evaluation on test set...")
try:
    # Load the best model saved during training
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    # --- Ensure model is re-initialized with correct input_size before loading state_dict ---
    saved_input_size = checkpoint.get('input_size', input_size) # Default to current if not saved
    if saved_input_size != input_size:
         print(f"⚠️ Warning: Model was saved with input_size {saved_input_size}, but current script calculated {input_size}. Using saved size {saved_input_size}.")
         model = ClassificationNet(saved_input_size).to(device) # Use saved size
    else:
         model = ClassificationNet(input_size).to(device) # Use current size

    model.load_state_dict(checkpoint['model_state_dict'])
    best_epoch = checkpoint.get('epoch', -1) + 1 # Adjust for 0-based index
    loaded_val_loss = checkpoint.get('val_loss', float('nan')) # Use nan if not found
    print(f"   Loaded best model from epoch {best_epoch} (Val Loss: {loaded_val_loss:.4f}).")


    model.eval()
    all_test_preds = []
    all_test_labels = []
    with torch.no_grad():
        for features, labels in test_loader: # Use test_loader
            features, labels = features.to(device), labels.to(device)
            test_outputs = model(features)
            test_probs = torch.sigmoid(test_outputs)
             # Use .detach() before .cpu() if using GPU
            test_preds = (test_probs > 0.5).detach().cpu().numpy().astype(int)
            all_test_preds.extend(test_preds)
            all_test_labels.extend(labels.detach().cpu().numpy().astype(int))

    y_test_orig = np.array(all_test_labels).flatten()
    y_pred = np.array(all_test_preds).flatten()

    acc = accuracy_score(y_test_orig, y_pred)
    f1 = f1_score(y_test_orig, y_pred, zero_division=0)
    precision = precision_score(y_test_orig, y_pred, zero_division=0)
    recall = recall_score(y_test_orig, y_pred, zero_division=0)

    print("\n--- 📊 Test Set Evaluation ---")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nClassification Report:\n", classification_report(
        y_test_orig, y_pred, target_names=['Not Safe (0)', 'Safe (1)'], zero_division=0 # Updated target names
    ))

except FileNotFoundError:
    print(f"❌ Error: Best model file not found at '{BEST_MODEL_PATH}'. Cannot evaluate test set.")
except Exception as e:
    print(f"❌ An error occurred during test set evaluation: {e}")
    # Optional: Print traceback for more details
    # import traceback
    # traceback.print_exc()


# ---------------------------------------------------------------------------
# 12. Save Preprocessing Tools
# ---------------------------------------------------------------------------
try:
    # Ensure directory exists before saving
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Save the imputer and scaler used on the training data
    # Add feature names to scaler for potential future use/debugging
    joblib.dump({'imputer': imputer, 'scaler': scaler}, PREPROCESS_PATH)
    print(f"\n💾 Preprocessing tools (imputer, scaler) saved to '{PREPROCESS_PATH}'")
    print(f"🏆 Best model checkpoint saved to '{BEST_MODEL_PATH}'")
except Exception as e:
    print(f"❌ Error saving preprocessing tools: {e}")

# ---------------------------------------------------------------------------