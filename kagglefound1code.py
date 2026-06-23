import os
import numpy as np
import pandas as pd

# ==================================================
# 1. SETUP PATHS & HARDWARE PROPERTIES
# ==================================================
DATA_DIR = r"D:\DATABASEYO\kagglefound1"
OUTPUT_FILE = r"D:\DATABASEYO\results\kaggle_motor_master_results.csv"

SAMPLING_RATE = 20000  # 20 kHz tracking rate
WINDOW_SIZE = 2048     # Perfect window size match for your baseline model

print("STATUS: Initializing 3-Axis Triaxial Acceleration Engine...")

try:
    all_files = [f for f in os.listdir(DATA_DIR) if (f.endswith('.xlsx') or f.endswith('.csv')) and not f.startswith('._')]
    all_files.sort()
except Exception as e:
    raise ValueError(f"Could not open target directory: {e}")

if len(all_files) == 0:
    raise ValueError(f"Zero diagnostic logs found in '{DATA_DIR}'. Check folder files.")

all_features = []

# ==================================================
# 2. TRIAXIAL PROCESSING CORE
# ==================================================
for file_name in all_files:
    file_path = os.path.join(DATA_DIR, file_name)
    name_lower = file_name.lower()
    
    # Precise Text Boundary Mapping
    if "mechanically_imbalanced" in name_lower:
        label = "structural_fault"
    elif "electrically" in name_lower and "fault" in name_lower:
        label = "electrical_fault"
    elif "no_mechanical" in name_lower or "load" in name_lower:
        label = "nominal"
    else:
        continue  # Skip unmapped composite files smoothly
        
    print(f"-> Processing Triaxial Log: {file_name} [{label}]")
    
    try:
        # Load the data cleanly whether it's stored as .csv or .xlsx
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
            
        # Standardize columns to lowercase to prevent naming mismatch issues
        df.columns = [c.lower() for c in df.columns]
        
        # Isolate the exact sensor streams visible in your data sheet
        target_axes = ['accx', 'accy', 'accz']
        if not all(col in df.columns for col in target_axes):
            print(f"    Column structural mismatch in {file_name}. Attempting positional fallback...")
            # Fallback to structural positional array slicing if text names differ slightly
            signals = [pd.to_numeric(df.iloc[:, i].values, errors='coerce') for i in [1, 2, 3]]
        else:
            signals = [pd.to_numeric(df[col].values, errors='coerce') for col in target_axes]
            
        # Strip out any trailing unreadable text or NaN slots
        clean_signals = []
        for sig in signals:
            sig = sig[~np.isnan(sig)]
            clean_signals.append(sig)
            
        # Keep windows constrained to the shortest valid data channel length
        min_length = min(len(s) for s in clean_signals)
        if min_length < WINDOW_SIZE:
            continue
            
        num_windows = min_length // WINDOW_SIZE
        
        # Step through the time series slice by slice
        for w in range(num_windows):
            window_metrics = {
                'rms': [], 'kurtosis': [], 'skewness': [], 'peak': [], 'abs_mean': [],
                'crest_factor': [], 'shape_factor': [], 'impulse_factor': [], 'margin_factor': [],
                'spectral_energy': [], 'peak_frequency': [], 'spectral_centroid': []
            }
            
            # Loop through all 3 axes independently for this window position
            for axis_sig in clean_signals:
                window_data = axis_sig[w*WINDOW_SIZE : (w+1)*WINDOW_SIZE]
                
                # --- 1. TIME-DOMAIN AMPLITUDES ---
                rms = np.sqrt(np.mean(window_data**2))
                kurtosis = pd.Series(window_data).kurtosis()
                skewness = pd.Series(window_data).skew()
                peak = np.max(np.abs(window_data))
                abs_mean = np.mean(np.abs(window_data))
                
                # --- 2. TIME-DOMAIN RATIOS ---
                crest_factor = peak / rms if rms != 0 else 0
                shape_factor = rms / abs_mean if abs_mean != 0 else 0
                impulse_factor = peak / abs_mean if abs_mean != 0 else 0
                
                root_mean_sqrt = (np.mean(np.sqrt(np.abs(window_data))))**2
                margin_factor = peak / root_mean_sqrt if root_mean_sqrt != 0 else 0
                
                # --- 3. FREQUENCY-DOMAIN SPECTRAL METRICS ---
                fft_vals = np.abs(np.fft.fft(window_data))
                fft_freqs = np.fft.fftfreq(len(window_data), d=1/SAMPLING_RATE)
                
                pos_mask = fft_freqs > 0
                fft_vals = fft_vals[pos_mask]
                fft_freqs = fft_freqs[pos_mask]
                
                spectral_energy = np.sum(fft_vals**2)
                peak_frequency = fft_freqs[np.argmax(fft_vals)] if len(fft_vals) > 0 else 0
                
                sum_fft = np.sum(fft_vals)
                spectral_centroid = np.sum(fft_freqs * fft_vals) / sum_fft if sum_fft != 0 else 0
                
                # Collect values for this axis
                window_metrics['rms'].append(rms)
                window_metrics['kurtosis'].append(kurtosis)
                window_metrics['skewness'].append(skewness)
                window_metrics['peak'].append(peak)
                window_metrics['abs_mean'].append(abs_mean)
                window_metrics['crest_factor'].append(crest_factor)
                window_metrics['shape_factor'].append(shape_factor)
                window_metrics['impulse_factor'].append(impulse_factor)
                window_metrics['margin_factor'].append(margin_factor)
                window_metrics['spectral_energy'].append(spectral_energy)
                window_metrics['peak_frequency'].append(peak_frequency)
                window_metrics['spectral_centroid'].append(spectral_centroid)
            
            # Fuse the three physical axes together using a mean reduction step
            all_features.append({
                'rms': np.mean(window_metrics['rms']),
                'kurtosis': np.mean(window_metrics['kurtosis']),
                'skewness': np.mean(window_metrics['skewness']),
                'peak': np.mean(window_metrics['peak']),
                'abs_mean': np.mean(window_metrics['abs_mean']),
                'crest_factor': np.mean(window_metrics['crest_factor']),
                'shape_factor': np.mean(window_metrics['shape_factor']),
                'impulse_factor': np.mean(window_metrics['impulse_factor']),
                'margin_factor': np.mean(window_metrics['margin_factor']),
                'spectral_energy': np.mean(window_metrics['spectral_energy']),
                'peak_frequency': np.mean(window_metrics['peak_frequency']),
                'spectral_centroid': np.mean(window_metrics['spectral_centroid']),
                'label': label
            })
            
    except Exception as e:
        print(f"    Skipping unreadable block: {e}")
        continue

# ==================================================
# 3. CSV SAFE EXPORT ENGINE
# ==================================================
if all_features:
    df_kaggle = pd.DataFrame(all_features).dropna()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_kaggle.to_csv(OUTPUT_FILE, index=False)
    print(f"\n TRIAXIAL EXTRACTION COMPLETE: Exported {df_kaggle.shape[0]} windows.")
    print(df_kaggle['label'].value_counts())
else:
    print("\nCRITICAL: No matching features could be generated from the matrix layouts.")