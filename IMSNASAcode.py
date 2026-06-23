import os
import numpy as np
import pandas as pd

def extract_nasa_set(data_dir, output_file, test_set_num):
    """
    Universal Extractor for NASA IMS Set 1, 2, and 3/4.
    Automatically manages channels, axes, and target fault labels.
    """
    print(f"\nSTATUS: Commencing extraction on NASA IMS Set {test_set_num}...")
    
    # Standard settings uniform with your main XGBoost pipeline
    sampling_rate = 20000  
    window_size = 2048     
    
    try:
        all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and not f.startswith('._')]
        all_files.sort()
    except Exception as e:
        print(f"❌ Aborting: Unable to access path: {e}")
        return

    total_files = len(all_files)
    if total_files == 0:
        print(f"❌ Aborting: Zero files discovered in '{data_dir}'")
        return
        
    all_features = []

    for idx, file_name in enumerate(all_files):
        file_path = os.path.join(data_dir, file_name)
        
        # 1. SETUP CHRONOLOGICAL BOUNDARIES AND TARGET SELECTION BASED ON TEST SET
        if test_set_num == 1:
            # Set 1 Configuration: Dual-axis, maps Inner and Ball defects
            if idx < int(total_files * 0.40):
                # Healthy baseline across surviving channels
                targets = [(4, 'nominal'), (6, 'nominal')] # Ch 4=B2_Y, Ch 6=B3_Y
            elif idx >= int(total_files * 0.80):
                targets = [(5, 'innerrace'), (7, 'ball')] # Ch 5=B3_X, Ch 7=B4_X
            else:
                continue
                
        elif test_set_num == 2:
            # Set 2 Configuration: Single-axis, maps Outer Race defect
            if idx < int(total_files * 0.50):
                targets = [(0, 'nominal')]
            elif idx >= int(total_files * 0.80):
                targets = [(0, 'outerrace')]
            else:
                continue
                
        elif test_set_num in [3, 4]:
            # Set 3 Configuration: Single-axis, maps Outer Race defect on Bearing 3
            if idx < int(total_files * 0.50):
                targets = [(2, 'nominal')]
            elif idx >= int(total_files * 0.80):
                targets = [(2, 'outerrace')]
            else:
                continue
        else:
            raise ValueError("Invalid test set designator specified.")

        try:
            # Read tab-separated text documents
            df = pd.read_csv(file_path, sep='\t', header=None)
            
            for ch_idx, label in targets:
                raw_signal = pd.to_numeric(df.iloc[:, ch_idx].values, errors='coerce')
                raw_signal = raw_signal[~np.isnan(raw_signal)]
                
                if len(raw_signal) < window_size:
                    continue
                    
                num_windows = len(raw_signal) // window_size
                for w in range(num_windows):
                    window_data = raw_signal[w*window_size : (w+1)*window_size]
                    
                    # --- 1. TIME-DOMAIN CORE AMPLITUDES ---
                    rms = np.sqrt(np.mean(window_data**2))
                    kurtosis = pd.Series(window_data).kurtosis()
                    skewness = pd.Series(window_data).skew()
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
                    fft_freqs = np.fft.fftfreq(len(window_data), d=1/sampling_rate)
                    
                    pos_mask = fft_freqs > 0
                    fft_vals = fft_vals[pos_mask]
                    fft_freqs = fft_freqs[pos_mask]
                    
                    spectral_energy = np.sum(fft_vals**2)
                    peak_frequency = fft_freqs[np.argmax(fft_vals)] if len(fft_vals) > 0 else 0
                    
                    sum_fft = np.sum(fft_vals)
                    spectral_centroid = np.sum(fft_freqs * fft_vals) / sum_fft if sum_fft != 0 else 0
                    
                    all_features.append({
                        'rms': rms, 'kurtosis': kurtosis, 'skewness': skewness,
                        'peak': peak, 'abs_mean': abs_mean, 'crest_factor': crest_factor,
                        'shape_factor': shape_factor, 'impulse_factor': impulse_factor,
                        'margin_factor': margin_factor, 'spectral_energy': spectral_energy,
                        'peak_frequency': peak_frequency, 'spectral_centroid': spectral_centroid,
                        'label': label
                    })
        except Exception:
            continue # Skip corrupted lifecycle logs smoothly

    # Save data
    if all_features:
        df_export = pd.DataFrame(all_features).dropna()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Append mode allows you to run multiple tests sequentially into the same file
        file_exists = os.path.exists(output_file)
        df_export.to_csv(output_file, mode='a', index=False, header=not file_exists)
        print(f"✅ Success! Set {test_set_num} generated {df_export.shape[0]} windows.")
    else:
        print(f"❌ Extraction failed for Set {test_set_num}.")

# ==================================================
# EXECUTION ZONE: Process each dataset sequentially
# ==================================================
if __name__ == "__main__":
    # Target file where all parsed NASA data will be saved
    MASTER_NASA_CSV = r"D:\DATABASEYO\results\nasa_ims_master_results.csv"
    
    # Wipe the old master file clean if you want a fresh start
    if os.path.exists(MASTER_NASA_CSV):
        os.remove(MASTER_NASA_CSV)

    # Run Set 1 (Generates: nominal, innerrace, ball)
    extract_nasa_set(
        data_dir = r"D:\DATABASEYO\IMSNASA\1st_test\1st_test", 
        output_file = MASTER_NASA_CSV, 
        test_set_num = 1
    )
    
    # Run Set 2 (Generates: nominal, outerrace)
    extract_nasa_set(
        data_dir = r"D:\DATABASEYO\IMSNASA\2nd_test\2nd_test", 
        output_file = MASTER_NASA_CSV, 
        test_set_num = 2
    )

    # Run Set 3 (Generates: nominal, outerrace)
    extract_nasa_set(
        data_dir = r"D:\DATABASEYO\IMSNASA\3rd_test\4th_test\txt", # Or 4th_test depending on your folder name
        output_file = MASTER_NASA_CSV, 
        test_set_num = 3
    )
    
    # Verify the final master file contents
    if os.path.exists(MASTER_NASA_CSV):
        print("\n=== COMPLETE COMBINED NASA IMS METRICS ===")
        print(pd.read_csv(MASTER_NASA_CSV)['label'].value_counts())