import os
import numpy as np
import pandas as pd

# ==================================================
# 1. SETUP PATHS & HARDWARE METRICS
# ==================================================
DATA_DIR = r"D:\DATABASEYO\MafaulDa"   # Update to your actual unzipped MaFAULda folder path
OUTPUT_FILE = r"D:\DATABASEYO\results\mafaulda_master_results.csv"

SAMPLING_RATE = 50000  # MaFAULda default is 50 kHz
WINDOW_SIZE = 2048     

print("STATUS: Initializing MaFAULda 13-Feature Extraction Engine...")

all_features = []

if not os.path.exists(DATA_DIR):
    raise ValueError(f"Directory not found: '{DATA_DIR}'. Please check the path.")

# ==================================================
# 2. FILE TREE PARSER & SIGNAL CHUNKER
# ==================================================
# Walk through all nested subdirectories in MaFAULda
for root, dirs, files in os.walk(DATA_DIR):
    csv_files = [f for f in files if f.endswith('.csv') or f.endswith('.txt')]
    if not csv_files:
        continue
        
    # Determine the master label based on the folder path structure
    path_lower = root.lower().replace("\\", "/")
    
    if "normal" in path_lower:
        master_label = "nominal"
    elif "misalignment" in path_lower or "imbalance" in path_lower:
        master_label = "structural_fault"
    elif "bearing" in path_lower:
        master_label = "bearing_fault"
    else:
        # Fallback category if folders have modified names
        master_label = "structural_fault"

    print(f"-> Processing folder: {os.path.basename(root)} | Target: {master_label}")

    for file_name in csv_files:
        file_path = os.path.join(root, file_name)
        
        try:
            # MaFAULda files typically use commas, are headerless, and contain 8 columns
            df = pd.read_csv(file_path, header=None, nrows=100000) # Limit rows per file for memory safety
            
            # Select Column 1 as the primary radial accelerometer channel
            raw_signal = pd.to_numeric(df[1].values, errors='coerce')
            raw_signal = raw_signal[~np.isnan(raw_signal)]
            
            if len(raw_signal) < WINDOW_SIZE:
                continue
                
            num_windows = len(raw_signal) // WINDOW_SIZE
            
            for w in range(num_windows):
                window_data = raw_signal[w*WINDOW_SIZE : (w+1)*WINDOW_SIZE]
                
                # --- 1. TIME-DOMAIN CORE AMPLITUDES ---
                rms = np.sqrt(np.mean(window_data**2))
                kurtosis = pd.Series(window_data).kurtosis()
                kurtosis = 0.0 if np.isnan(kurtosis) else kurtosis
                skewness = pd.Series(window_data).skew()
                skewness = 0.0 if np.isnan(skewness) else skewness
                peak = np.max(np.abs(window_data))
                abs_mean = np.mean(np.abs(window_data))
                
                # --- 2. TIME-DOMAIN DIMENSIONLESS RATIOS ---
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
                
                spectral_energy = np.sum(fft_vals**2) if len(fft_vals) > 0 else 0
                peak_frequency = fft_freqs[np.argmax(fft_vals)] if len(fft_vals) > 0 else 0
                
                sum_fft = np.sum(fft_vals)
                spectral_centroid = np.sum(fft_freqs * fft_vals) / sum_fft if sum_fft != 0 else 0
                
                all_features.append({
                    'rms': rms, 'kurtosis': kurtosis, 'skewness': skewness,
                    'peak': peak, 'abs_mean': abs_mean, 'crest_factor': crest_factor,
                    'shape_factor': shape_factor, 'impulse_factor': impulse_factor,
                    'margin_factor': margin_factor, 'spectral_energy': spectral_energy,
                    'peak_frequency': peak_frequency, 'spectral_centroid': spectral_centroid,
                    'label': master_label
                })
                
        except Exception as e:
            # Gracefully continue if metadata or formatting lines cause an issue
            continue

# ==================================================
# 3. EXPORT RESULTS TO STORAGE
# ==================================================
if all_features:
    df_mafaulda = pd.DataFrame(all_features)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_mafaulda.to_csv(OUTPUT_FILE, index=False)
    print(f"\n SUCCESS: Compiled {df_mafaulda.shape[0]} windows into '{OUTPUT_FILE}'")
    print(df_mafaulda['label'].value_counts())
else:
    print("\n EXTRACTION FAILED: No valid feature vectors were compiled. Verify folder path content.")