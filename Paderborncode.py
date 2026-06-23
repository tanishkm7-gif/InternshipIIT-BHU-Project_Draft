import os
import scipy.io
import numpy as np
import pandas as pd

# ==================================================
# 1. SETUP PATHS AND MAPPINGS
# ==================================================
# This is your master Paderborn folder
DATA_DIR = r"D:\DATABASEYO\paderborn" 
OUTPUT_FILE = r"D:\DATABASEYO\results\paderborn_master_results.xlsx"

print("STATUS: Initializing Deep Subfolder Paderborn Pipeline...")

all_features = []
mat_files_paths = []

# --- NEW DEEP SEARCH LOGIC ---
# os.walk automatically digs into K001, KI21, etc.
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith('.mat'):
            full_path = os.path.join(root, file)
            mat_files_paths.append((file, full_path))

print(f"Found {len(mat_files_paths)} total .mat files hidden across subfolders.")

# ==================================================
# 2. DEFINE MAPPER FOR BEARING HEALTH CODES
# ==================================================
def get_bearing_label(filename):
    fn_upper = filename.upper()
    
    if any(code in fn_upper for code in ['K001', 'K002', 'K003', 'K004', 'K005', 'K006']):
        return 'nominal'
    elif 'KA' in fn_upper:
        return 'outerrace'
    elif 'KI' in fn_upper:
        return 'innerrace'
    elif 'KB' in fn_upper:
        return 'ball'
    else:
        return 'unknown_fault'

# ==================================================
# 3. PROCESSING LOOP
# ==================================================
for idx, (file_name, file_path) in enumerate(mat_files_paths):
    label = get_bearing_label(file_name)
    
    print(f"[{idx+1}/{len(mat_files_paths)}] Deep Processing: {file_name} -> {label}")
    
    try:
        mat = scipy.io.loadmat(file_path)
        
        data_keys = [k for k in mat.keys() if not k.startswith('__')]
        main_key = data_keys[0]
        
        structured_data = mat[main_key][0, 0]
        y_matrix = structured_data['Y']
        sensor_slot = y_matrix[0, 6] 
        raw_signal = sensor_slot['Data'][0].flatten()

        sampling_rate = 1.0
        if hasattr(structured_data, 'dtype') and structured_data.dtype.names is not None:
            for candidate in ['Fs', 'fs', 'sampling_rate', 'SamplingRate', 'SamplingFreq', 'samplingFreq']:
                if candidate in structured_data.dtype.names:
                    try:
                        sampling_rate = float(np.squeeze(structured_data[candidate]))
                        break
                    except Exception:
                        pass

        window_size = 2048
        num_windows = len(raw_signal) // window_size
        
        for w in range(num_windows):
            window_data = raw_signal[w*window_size : (w+1)*window_size]
            
            # --- EXPANDED 12-FEATURE MATHEMATICAL ENGINE ---
            rms = np.sqrt(np.mean(window_data**2))
            kurtosis = pd.Series(window_data).kurtosis()
            skewness = pd.Series(window_data).skew()  # New: Tracks structural asymmetry

            abs_mean = np.mean(np.abs(window_data))
            peak = np.max(np.abs(window_data))

            crest_factor = peak / rms if rms != 0 else 0
            shape_factor = rms / abs_mean if abs_mean != 0 else 0        # New: Tracks pure wave-shape deformation
            impulse_factor = peak / abs_mean if abs_mean != 0 else 0    # New: Captures sharp, tiny surface impacts
            margin_factor = peak / (np.mean(np.sqrt(np.abs(window_data)))**2) if abs_mean != 0 else 0  # New: Early spall detection

            # Advanced Frequency Analysis
            fft_vals = np.abs(np.fft.fft(window_data))
            fft_freqs = np.fft.fftfreq(len(window_data), d=1/sampling_rate)

            pos_mask = fft_freqs > 0
            fft_vals = fft_vals[pos_mask]
            fft_freqs = fft_freqs[pos_mask]

            spectral_energy = np.sum(fft_vals**2)
            peak_frequency = fft_freqs[np.argmax(fft_vals)] if len(fft_vals) > 0 else 0

            # Spectral Centroid (Center of Mass of Frequency Spectra)
            sum_fft = np.sum(fft_vals)
            spectral_centroid = np.sum(fft_freqs * fft_vals) / sum_fft if sum_fft != 0 else 0  # New: Catches friction shifting frequencies up

            # Append all 12 features now to your dictionary
            all_features.append({
                'rms': rms, 'kurtosis': kurtosis, 'skewness': skewness,
                'crest_factor': crest_factor, 'shape_factor': shape_factor, 
                'impulse_factor': impulse_factor, 'margin_factor': margin_factor,
                'spectral_energy': spectral_energy, 'peak_frequency': peak_frequency,
                'spectral_centroid': spectral_centroid, 'peak': peak, 'abs_mean': abs_mean,
                'label': label
            })
            
    except Exception as e:
        print(f" ERROR in file {file_name}: {e}")

# ==================================================
# 4. SAVE OUTPUT
# ==================================================
if all_features:
    df_paderborn = pd.DataFrame(all_features).dropna()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_paderborn.to_excel(OUTPUT_FILE, index=False)
    print(f"\n==================================================")
    print(f"SUCCESS: Generated {df_paderborn.shape[0]} feature rows.")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"==================================================")
else:
    print("\n Extraction failed. Check data structural format.")