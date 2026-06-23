import os
try:
    import pandas as pd
except ImportError as exc:
    raise ImportError("Required package 'pandas' is not installed. Install it with `pip install pandas`.") from exc
try:
    import numpy as np
except ImportError as exc:
    raise ImportError("Required package 'numpy' is not installed. Install it with `pip install numpy`.") from exc
try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("Required package 'matplotlib' is not installed. Install it with `pip install matplotlib`.") from exc
try:
    import seaborn as sns
except ImportError as exc:
    raise ImportError("Required package 'seaborn' is not installed. Install it with `pip install seaborn`.") from exc
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.metrics import f1_score
except ImportError as exc:
    raise ImportError("Required package 'scikit-learn' is not installed. Install it with `pip install scikit-learn`.") from exc
try:
    import xgboost as xgb
except ImportError as exc:
    raise ImportError("Required package 'xgboost' is not installed. Install it with `pip install xgboost`.") from exc
try:
    from imblearn.over_sampling import SMOTE
except ImportError as exc:
    raise ImportError("Required package 'imbalanced-learn' is not installed. Install it with `pip install imbalanced-learn`.") from exc

# =====================================================================
# 1. SETUP PATHS & REGISTRY
# =====================================================================
BASE_DIR = r"D:\DATABASEYO\results"

dataset_files = {
    'Paderborn': 'paderborn_master_results',
    'NASA_IMS': 'nasa_ims_master_results',
    'Kaggle1': 'kaggle_motor_master_results',
    'MaFAULda': 'mafaulda_master_results',
    'SDUST': 'sdust_master_results',
    'HIT': 'hit_master_results',
    'MCC5THU': 'mcc5_thu_targeted_results'
}

all_dfs = []

print("STATUS: Loading and aligning all feature matrices...")

for dataset_name, file_name in dataset_files.items():
    if not file_name.endswith('.csv'):
        file_name += '.csv'
    file_path = os.path.join(BASE_DIR, file_name)
    
    if os.path.exists(file_path):
        df_dataset = pd.read_csv(file_path)
        if not df_dataset.empty:
            df_dataset['dataset_origin_track'] = dataset_name
            all_dfs.append(df_dataset)

if len(all_dfs) == 0:
    print(f" CRITICAL ERROR: No data found in '{BASE_DIR}'. Verify your paths.")
    exit()

df_master = pd.concat(all_dfs, ignore_index=True)
df_master.columns = [c.strip().lower() for c in df_master.columns]

#  DOMAIN ISOLATION TRACKING ENGINE
# Enforce 0 (Vibration) on legacy datasets that lack an explicit tracker column
if 'is_electrical' not in df_master.columns:
    df_master['is_electrical'] = 0
else:
    df_master['is_electrical'] = df_master['is_electrical'].fillna(0).astype(int)

# =====================================================================
# 2. CHIP-LEVEL LABEL CONSOLIDATION & CORRUPT BOUNDARY FILTERING
# =====================================================================
print("\n-> Consolidating sub-component defects into hardware target classes...")

label_consolidation = {
    'ball': 'bearing_fault',
    'innerrace': 'bearing_fault',
    'outerrace': 'bearing_fault',
    'xjtu_degradation': 'bearing_fault',
    'bearing_fault': 'bearing_fault',
    'electrical_fault': 'electrical_fault',
    'structural_fault': 'structural_fault',
    'gear_fault': 'gear_fault',
    'nominal': 'nominal',
    
    # SDUST explicit routes
    'inner_race_fault': 'bearing_fault',
    'outer_race_fault': 'bearing_fault',
    'roller_fault': 'bearing_fault',
    'normal': 'nominal',
    'unknown': 'nominal'
}

df_master['label'] = df_master['label'].map(label_consolidation)
df_master = df_master.dropna(subset=['label'])

invalid_elec_mask = (df_master['dataset_origin_track'] == 'EVdata') & (df_master['label'] == 'electrical_fault')
df_master = df_master[~invalid_elec_mask]

print("Consolidated Hardware Targets (Cleaned Check):\n", df_master['label'].value_counts())
print(f"Total rows remaining for training: {df_master.shape[0]}")

print("\n --- DATA DISTRIBUTION PROFILE PER CSV ---")
distribution_matrix = pd.crosstab(df_master['dataset_origin_track'], df_master['label'])
print(distribution_matrix)
print("--------------------------------------------\n")

# =====================================================================
# 3. FEATURE SEPARATION & NEIGHBORHOOD RESAMPLING (SMOTE)
# =====================================================================
#  Added 'is_electrical' explicitly to feature matrices
feature_columns = [
    'kurtosis', 'skewness', 'crest_factor', 'shape_factor', 
    'impulse_factor', 'margin_factor', 'peak_frequency', 'spectral_centroid',
    'is_electrical'
]

print(" Cleaning infinite and missing feature values before resampling...")

# Clean raw math features safely
math_features = feature_columns[:-1]
df_master[math_features] = df_master[math_features].replace([np.inf, -np.inf], np.nan)

initial_rows = df_master.shape[0]
df_master = df_master.dropna(subset=feature_columns)
cleaned_rows = df_master.shape[0]

if initial_rows != cleaned_rows:
    print(f" Cleaned {initial_rows - cleaned_rows} bad rows containing NaN/Inf values.")

y_raw = df_master['label']
X_raw = df_master[feature_columns].copy()
origins = df_master['dataset_origin_track'].values  

le_target = LabelEncoder()
y = le_target.fit_transform(y_raw)
class_names = le_target.classes_

X_train, X_test, y_train, y_test, origins_train, origins_test = train_test_split(
    X_raw, y, origins, test_size=0.20, random_state=42, stratify=y
)

class_counts = pd.Series(y_train).value_counts()
majority_count = class_counts.max()

target_sampling = {}
for c_idx, count in class_counts.items():
    if count < (majority_count * 0.15):
        target_sampling[c_idx] = int(majority_count * 0.15)
    else:
        target_sampling[c_idx] = count

print(f"-> Dynamic resampling blueprint: {target_sampling}")

print("-> Restructuring target boundaries using neighborhood interpolation...")
# k_neighbors=1 safeguards minority classes like Kaggle1 electrical seeds from interpolation faults
smote = SMOTE(sampling_strategy=target_sampling, random_state=42, k_neighbors=1)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Round domain category back to crisp integers post-SMOTE
X_train_res['is_electrical'] = np.round(X_train_res['is_electrical']).astype(int)

# =====================================================================
# 4. TRAIN FPGA-CONSTRAINED MODEL WITH GRADIENT ISOLATION
# =====================================================================
print(f"\nSTATUS: Training Clean Hardware-Constrained XGBoost (Gradient Isolated)...")

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_res)

model = xgb.XGBClassifier(
    n_estimators=80,       
    max_depth=4,
    learning_rate=0.15,
    objective='multi:softprob',
    tree_method='hist',
    max_delta_step=3,       
    subsample=0.8,          
    colsample_bytree=0.8,
    reg_alpha=1.5,         
    reg_lambda=1.5,
    random_state=42
)

# FIXED: Removed the second unweighted fit() call which was overwriting weights
model.fit(X_train_res, y_train_res, sample_weight=sample_weights)

y_pred = model.predict(X_test)

# =====================================================================
# 5. THE ACCURACY REPORT SUITE
# =====================================================================
print("\n" + "="*60)
print("             GLOBAL PERFORMANCE DIAGNOSTICS                 ")
print("="*60)

macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f" SYSTEM GENERALIZATION SUITE (Macro F1): {macro_f1 * 100:.2f}%")
global_acc = accuracy_score(y_test, y_pred)
print(f" OVERALL CROSS-DOMAIN ACCURACY: {global_acc * 100:.2f}%\n")

print(" CLASS-SPECIFIC METRIC MATRIX:")
print(classification_report(y_test, y_pred, target_names=class_names))

print(" TEXT CONFUSION MATRIX:")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=[f"True {c}" for c in class_names], columns=[f"Pred {c}" for c in class_names]))

print("\n" + "="*60)
print("         PER-DATASET GENERALIZATION MATRIX                 ")
print("="*60)
print(f"{'Dataset Source':<20} | {'Test Rows':<10} | {'Accuracy Score':<15}")
print("-"*60)

for d_name in np.unique(origins_test):
    mask = (origins_test == d_name)
    if np.sum(mask) > 0:
        d_acc = accuracy_score(y_test[mask], y_pred[mask])
        print(f"{d_name:<20} | {np.sum(mask):<10} | {d_acc * 100:.2f}%")
print("="*60)

# =====================================================================
# 6. EXPORT METRIC PLOT & MODEL
# =====================================================================
plt.clf()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Zynq Hardware-Constrained Balanced Confusion Matrix')
plt.ylabel('Actual Fault State')
plt.xlabel('Predicted Fault State')
plt.tight_layout()

output_plot_path = os.path.join(BASE_DIR, 'zynq_model_confusion_matrix.png')
plt.savefig(output_plot_path)
print(f"\n  Visual Confusion Matrix plot saved to: '{output_plot_path}'")

model_json_path = os.path.join(BASE_DIR, 'zynq_optimized_fault_model.json')
model.save_model(model_json_path)
print(f" Hardware model binary exported successfully to: '{model_json_path}'")