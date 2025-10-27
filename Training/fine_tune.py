import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time

# Assuming metrics are needed for evaluation, though not explicitly in original
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split # Needed if splitting real data
from sklearn.impute import SimpleImputer # Needed for imputation

# ---------------------------------------------------------------------------
# 1. Setup Device (GPU or CPU)
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 2. Model Definition (MUST match your pre-training script)
# ---------------------------------------------------------------------------
class ClassificationNet(nn.Module):
    def __init__(self, input_size):
        super(ClassificationNet, self).__init__()
        # Adding input validation
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError(f"input_size must be a positive integer, got {input_size}")

        # This architecture MUST exactly match the one saved in BEST_MODEL_PATH
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
# 3. Constants and Paths
# ---------------------------------------------------------------------------
MODEL_DIR = 'model'
# Path to the model PRE-TRAINED on synthetic data
PRETRAINED_MODEL_PATH = os.path.join(MODEL_DIR, 'water_quality_prediction_new_test.pth')
# Path to the preprocessing tools fitted on synthetic data
PREPROCESS_PATH = os.path.join(MODEL_DIR, 'preprocessing_tools_test.pkl')
# Path to the REAL-WORLD dataset for fine-tuning
REAL_DATA_PATH = os.path.join(MODEL_DIR, 'water_potability_real.csv')
# Path to save the FINAL fine-tuned model
FINETUNED_MODEL_PATH = os.path.join(MODEL_DIR, 'water_quality_finetuned1.pth')

# --- Optimized Fine-tuning Hyperparameters ---
FINETUNE_LR = 1e-5  # CRITICAL: Very small learning rate
FINETUNE_EPOCHS = 30 # Fewer epochs needed
FINETUNE_BATCH_SIZE = 64 # Smaller batch size for smaller dataset
FINETUNE_WEIGHT_DECAY = 1e-4 # Can keep the same or slightly reduce
FINETUNE_PATIENCE = 7 # Early stopping patience
SCHEDULER_PATIENCE = 5 # LR scheduler patience

# Target column in the REAL dataset (must match the CSV)
REAL_TARGET_COLUMN = 'Potability' # Common name in Kaggle dataset

# ---------------------------------------------------------------------------
# 4. Load Preprocessing Tools
# ---------------------------------------------------------------------------
print(f"Loading preprocessing tools from '{PREPROCESS_PATH}'...")
try:
    tools = joblib.load(PREPROCESS_PATH)
    imputer = tools.get('imputer')
    scaler = tools.get('scaler') # Assuming key is 'scaler' based on recent training script
    if imputer is None or scaler is None:
        raise ValueError("'imputer' or 'scaler' key not found in .pkl file.")
    print("✅ Preprocessing tools loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Preprocessing tools file not found at '{PREPROCESS_PATH}'.")
    print("   Make sure you have run the main training script on synthetic data first.")
    exit()
except Exception as e:
    print(f"❌ Error loading preprocessing tools: {e}")
    exit()

# ---------------------------------------------------------------------------
# 5. Load and Preprocess Real-World Dataset
# ---------------------------------------------------------------------------
print(f"\nLoading real-world data from '{REAL_DATA_PATH}' for fine-tuning...")
try:
    real_data = pd.read_csv(REAL_DATA_PATH)
    print(f"✅ Real-world dataset loaded successfully ({len(real_data)} rows).")
except FileNotFoundError:
    print(f"❌ Error: Real-world data file not found at '{REAL_DATA_PATH}'.")
    print(f"   Please download the dataset (e.g., from Kaggle) and save it as '{os.path.basename(REAL_DATA_PATH)}' in the '{MODEL_DIR}' directory.")
    exit()
except Exception as e:
    print(f"❌ Error loading real-world data: {e}")
    exit()

# Basic Preprocessing for Real Data
print("Preprocessing real-world data...")
real_data.columns = [col.strip().lower() for col in real_data.columns] # Convert to lowercase, strip spaces
print("   Renamed columns to lowercase for compatibility.")

# Ensure target column exists
real_target_column_lower = REAL_TARGET_COLUMN.lower()
if real_target_column_lower not in real_data.columns:
    print(f"❌ Target column '{real_target_column_lower}' (expected from REAL_TARGET_COLUMN) not found in real data.")
    print(f"   Available columns: {real_data.columns.tolist()}")
    exit()

# Get feature names expected by the scaler (learned from synthetic data)
# These should ideally be lowercase now based on realistic generator
expected_features = getattr(scaler, 'feature_names_in_', None)
if expected_features is None:
     # Fallback: try to get from imputer or assume order if not available
     expected_features = getattr(imputer, 'feature_names_in_', None)
     if expected_features is None:
         print("⚠️ Warning: Cannot determine expected feature names from scaler/imputer.")
         # Inferring from pre-trained model path if possible (might require loading checkpoint first - complex)
         # For now, we MUST assume the real data columns (after lowercasing) match the order and names used in pre-training
         # Let's try to infer from the number of features the imputer/scaler expect
         n_expected_features = getattr(scaler, 'n_features_in_', getattr(imputer, 'n_features_in_', None))
         if n_expected_features:
             potential_features = [col for col in real_data.columns if col != real_target_column_lower]
             if len(potential_features) == n_expected_features:
                 expected_features = potential_features
                 print(f"   Inferred expected features based on count ({n_expected_features}): {expected_features}")
             else:
                 print(f"❌ Error: Number of features in real data ({len(potential_features)}) doesn't match expected ({n_expected_features}).")
                 exit()
         else:
              print("❌ Error: Cannot determine expected features. Preprocessing tools might be invalid.")
              exit()


# Ensure all expected feature columns exist in the real data (after lowercasing)
missing_features = [f for f in expected_features if f not in real_data.columns]
if missing_features:
    print(f"❌ Error: Real data is missing expected feature columns: {missing_features}")
    exit()

# Select only the expected feature columns IN THE CORRECT ORDER
X_real = real_data[expected_features]
y_real = real_data[real_target_column_lower].values

# Convert features to numeric, coercing errors
print("   Converting real data features to numeric...")
for col in X_real.columns:
    X_real[col] = pd.to_numeric(X_real[col], errors='coerce')

# --- Split real data: Fine-tune on train, evaluate on validation ---
# We need a separate validation set from the real data to monitor fine-tuning
X_ft_train, X_ft_val, y_ft_train, y_ft_val = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42, stratify=y_real if len(np.unique(y_real)) > 1 and np.min(np.bincount(y_real.astype(int))) >= 2 else None
)

print(f"   Fine-tuning training samples: {len(X_ft_train)}")
print(f"   Fine-tuning validation samples: {len(X_ft_val)}")

# Apply PRE-FITTED imputer and scaler
print("   Applying pre-fitted imputer and scaler to real data...")
try:
    X_ft_train_imp = imputer.transform(X_ft_train)
    X_ft_val_imp = imputer.transform(X_ft_val)

    X_ft_train_scaled = scaler.transform(X_ft_train_imp)
    X_ft_val_scaled = scaler.transform(X_ft_val_imp)
except ValueError as e:
    print(f"❌ Error during imputation/scaling: {e}")
    print("   This often means the columns in 'water_potability_real.csv' (after lowercasing)")
    print(f"   do not match the features the tools were trained on: {expected_features}")
    exit()

# Convert to Tensors
X_train_tensor = torch.tensor(X_ft_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_ft_train, dtype=torch.float32).unsqueeze(1).to(device)
X_val_tensor = torch.tensor(X_ft_val_scaled, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_ft_val, dtype=torch.float32).unsqueeze(1).to(device)

# Create DataLoaders
ft_train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=FINETUNE_BATCH_SIZE, shuffle=True)
ft_val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=FINETUNE_BATCH_SIZE * 2, shuffle=False)


# ---------------------------------------------------------------------------
# 6. Load Pre-trained Model
# ---------------------------------------------------------------------------
print(f"\nLoading pre-trained model from '{PRETRAINED_MODEL_PATH}'...")
try:
    checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
    # Determine input size from checkpoint or tools
    saved_input_size = checkpoint.get('input_size')
    if saved_input_size is None:
        # Try inferring from scaler/imputer again
        saved_input_size = getattr(scaler, 'n_features_in_', getattr(imputer, 'n_features_in_', None))
        if saved_input_size is None:
             raise ValueError("Cannot determine input size from checkpoint or preprocessing tools.")
        print(f"   Inferred input size {saved_input_size} from preprocessing tools.")
    else:
         print(f"   Input size {saved_input_size} found in checkpoint.")

    # Instantiate model with the correct input size
    model = ClassificationNet(saved_input_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    pretrain_epoch = checkpoint.get('epoch', -1) + 1
    pretrain_val_loss = checkpoint.get('val_loss', float('nan'))
    print(f"✅ Pre-trained model loaded successfully (trained for {pretrain_epoch} epochs, val_loss: {pretrain_val_loss:.4f}).")

    # --- Crucial: Ensure model input size matches preprocessed real data ---
    if saved_input_size != X_ft_train_scaled.shape[1]:
        print(f"❌ CRITICAL Error: Model input size ({saved_input_size}) does not match preprocessed real data feature count ({X_ft_train_scaled.shape[1]}).")
        print("   This likely means the real data columns don't match the synthetic data used for pre-training.")
        exit()

except FileNotFoundError:
    print(f"❌ Error: Pre-trained model file not found at '{PRETRAINED_MODEL_PATH}'.")
    print("   Make sure you have run the main training script first.")
    exit()
except Exception as e:
    print(f"❌ Error loading pre-trained model: {e}")
    exit()

# ---------------------------------------------------------------------------
# 7. Setup Fine-tuning Optimizer, Scheduler, Loss
# ---------------------------------------------------------------------------
# Use a NEW optimizer with the fine-tuning LR
optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=FINETUNE_WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=SCHEDULER_PATIENCE) # Adjusted patience

# Loss function - can recalculate pos_weight based on the FT training split if desired
# Or reuse the one from pre-training if synthetic distribution is trusted more
y_ft_train_int = y_ft_train.astype(int)
ft_safe_count = np.sum(y_ft_train_int == 1)
ft_unsafe_count = np.sum(y_ft_train_int == 0)
epsilon = 1e-6
if ft_safe_count > 0 and ft_unsafe_count > 0:
     pos_weight_val = ft_unsafe_count / (ft_safe_count + epsilon)
else:
     pos_weight_val = 1.0 # Default

pos_weight = torch.tensor([pos_weight_val]).to(device)
print(f"   Calculated pos_weight for fine-tuning: {pos_weight_val:.4f}")
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ---------------------------------------------------------------------------
# 8. Fine-tuning Loop
# ---------------------------------------------------------------------------
print("\n🚀 Starting fine-tuning...\n")
best_ft_val_loss = float('inf')
epochs_no_improve = 0
start_time = time.time()

for epoch in range(FINETUNE_EPOCHS):
    epoch_start_time = time.time()
    model.train()
    total_ft_train_loss = 0.0

    for features, labels in ft_train_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_ft_train_loss += loss.item()

    avg_ft_train_loss = total_ft_train_loss / len(ft_train_loader)

    # Validation phase (on real validation set)
    model.eval()
    total_ft_val_loss = 0.0
    all_ft_val_preds = []
    all_ft_val_labels = []
    with torch.no_grad():
        for features, labels in ft_val_loader:
            features, labels = features.to(device), labels.to(device)
            val_outputs = model(features)
            val_loss = criterion(val_outputs, labels)
            total_ft_val_loss += val_loss.item()

            val_probs = torch.sigmoid(val_outputs)
            val_preds = (val_probs > 0.5).detach().cpu().numpy().astype(int)
            all_ft_val_preds.extend(val_preds)
            all_ft_val_labels.extend(labels.detach().cpu().numpy().astype(int))

    avg_ft_val_loss = total_ft_val_loss / len(ft_val_loader)
    all_ft_val_labels_np = np.array(all_ft_val_labels).flatten()
    all_ft_val_preds_np = np.array(all_ft_val_preds).flatten()
    ft_val_f1 = f1_score(all_ft_val_labels_np, all_ft_val_preds_np, zero_division=0)

    epoch_duration = time.time() - epoch_start_time
    scheduler.step(avg_ft_val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch [{epoch+1}/{FINETUNE_EPOCHS}] | Train Loss: {avg_ft_train_loss:.4f} | Val Loss: {avg_ft_val_loss:.4f} | Val F1: {ft_val_f1:.4f} | LR: {current_lr:.1E} | Time: {epoch_duration:.2f}s")

    # Early stopping and save best fine-tuned model
    if avg_ft_val_loss < best_ft_val_loss:
        previous_best_loss_str = f"{best_ft_val_loss:.4f}" if best_ft_val_loss != float('inf') else "inf"
        print(f"   Validation loss improved ({previous_best_loss_str} --> {avg_ft_val_loss:.4f}). Saving fine-tuned model...")
        best_ft_val_loss = avg_ft_val_loss
        epochs_no_improve = 0
        try:
            # Save the state of the fine-tuned model
            torch.save({
                'epoch': epoch, # Fine-tuning epoch
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), # Optimizer state might be useful
                'val_loss': best_ft_val_loss,
                'input_size': saved_input_size # Use the original input size
             }, FINETUNED_MODEL_PATH)
        except Exception as e:
            print(f"❌ Error saving fine-tuned model checkpoint: {e}")

    else:
        epochs_no_improve += 1
        print(f"   Validation loss did not improve for {epochs_no_improve} epoch(s).")


    if epochs_no_improve >= FINETUNE_PATIENCE:
        print(f"\n🛑 Early stopping triggered after epoch {epoch + 1}.")
        print(f"   Best fine-tuning validation loss achieved: {best_ft_val_loss:.4f}")
        break

end_time = time.time()
print(f"\n✅ Fine-tuning complete! Total time: {(end_time - start_time) // 60:.0f}m {(end_time - start_time) % 60:.2f}s")
print(f"🏆 Best fine-tuned model saved to '{FINETUNED_MODEL_PATH}'")


# ---------------------------------------------------------------------------
# 9. Final Evaluation (Optional but Recommended)
# ---------------------------------------------------------------------------
# Evaluate the *final* fine-tuned model on the held-out real validation set
print("\nLoading best fine-tuned model for final evaluation on validation set...")
try:
    # Load the best fine-tuned model
    checkpoint = torch.load(FINETUNED_MODEL_PATH, map_location=device)
    # Re-initialize model with correct size just in case
    saved_input_size = checkpoint.get('input_size')
    if saved_input_size is None:
         print("⚠️ Warning: Input size not found in fine-tuned checkpoint. Attempting to proceed.")
         saved_input_size = X_ft_val_scaled.shape[1] # Assume current shape

    model = ClassificationNet(saved_input_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_epoch = checkpoint.get('epoch', -1) + 1
    loaded_val_loss = checkpoint.get('val_loss', float('nan'))
    print(f"   Loaded best fine-tuned model from epoch {best_epoch} (Val Loss: {loaded_val_loss:.4f}).")

    model.eval()
    all_final_val_preds = []
    all_final_val_labels = []
    with torch.no_grad():
        # Use the ft_val_loader created earlier
        for features, labels in ft_val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).detach().cpu().numpy().astype(int)
            all_final_val_preds.extend(preds)
            all_final_val_labels.extend(labels.detach().cpu().numpy().astype(int))

    y_val_orig = np.array(all_final_val_labels).flatten()
    y_pred = np.array(all_final_val_preds).flatten()

    acc = accuracy_score(y_val_orig, y_pred)
    f1 = f1_score(y_val_orig, y_pred, zero_division=0)
    precision = precision_score(y_val_orig, y_pred, zero_division=0)
    recall = recall_score(y_val_orig, y_pred, zero_division=0)

    print("\n--- 📊 Final Fine-Tuned Model Evaluation (on Real Validation Set) ---")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nClassification Report:\n", classification_report(
        y_val_orig, y_pred, target_names=['Not Safe (0)', 'Safe (1)'], zero_division=0
    ))

except FileNotFoundError:
    print(f"❌ Error: Fine-tuned model file not found at '{FINETUNED_MODEL_PATH}'. Cannot evaluate.")
except Exception as e:
    print(f"❌ An error occurred during final evaluation: {e}")
