import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# =========================================================
# 1. MATHEMATICAL FEATURE EXTRACTOR (FPGA-CONSTRAINED)
# =========================================================
def calculate_8_features(signal):
    mean_val = np.mean(signal)
    std_val = np.std(signal, ddof=1) if np.std(signal, ddof=1) != 0 else 1e-6
    max_val = np.max(np.abs(signal))
    
    kurtosis = np.mean((signal - mean_val) ** 4) / (np.mean((signal - mean_val) ** 2) ** 2)
    skewness = np.mean((signal - mean_val) ** 3) / (np.mean((signal - mean_val) ** 2) ** 1.5)
    crest_factor = max_val / std_val
    shape_factor = std_val / np.mean(np.abs(signal))
    impulse_factor = max_val / np.mean(np.abs(signal))
    margin_factor = max_val / (np.mean(np.sqrt(np.abs(signal))) ** 2)
    
    fft_vals = np.abs(np.fft.rfft(signal))
    fft_freqs = np.fft.rfftfreq(len(signal))
    peak_frequency = fft_freqs[np.argmax(fft_vals)]
    spectral_centroid = np.sum(fft_freqs * fft_vals) / (np.sum(fft_vals) + 1e-6)
    
    return [kurtosis, skewness, crest_factor, shape_factor, 
            impulse_factor, margin_factor, peak_frequency, spectral_centroid]

# =========================================================
# 2. FILE PROCESSING ENGINE
# =========================================================
def process_hit_file(file_path):
    base_name = os.path.basename(file_path)
    fn_upper = base_name.upper()
    
    # 💡 Smart Class Resolver based on HIT naming conventions
    if "MISALIGNMENT" in fn_upper or "UNBALANCE" in fn_upper or "LOOSENESS" in fn_upper:
        label = "structural_fault"
    elif "NORMAL" in fn_upper or "HEALTHY" in fn_upper:
        label = "nominal"
    else:
        label = "structural_fault"  # HIT typically centers on structural rotor diagnostics
        
    try:
        mat = sio.loadmat(file_path)
        # Pull the primary matrix array dynamically (e.g., 'xtrain_1')
        key = [k for k in mat.keys() if not k.startswith('__')][0]
        matrix = mat[key]
        
        feature_rows = []
        # HIT is pre-sliced! Iterate row by row through the pre-cut windows
        for i in range(matrix.shape[0]):
            window = matrix[i, :]
            features = calculate_8_features(window)
            feature_rows.append(features + [label])
            
        return feature_rows, f"✅ [SUCCESS] Extracted {len(feature_rows)} structural frames from {base_name}"
    except Exception as e:
        return None, f"❌ [ERROR] Failed to parse {base_name}: {str(e)}"

# =========================================================
# 3. CORE ORCHESTRATOR
# =========================================================
if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # Paths configuration
    HIT_DIR = r"D:\DATABASEYO\HITdataset"
    OUTPUT_CSV_DIR = r"D:\DATABASEYO\results"
    output_file = os.path.join(OUTPUT_CSV_DIR, "hit_master_results.csv")
    
    # Grab all .mat files recursively inside the HIT folder
    files = glob.glob(os.path.join(HIT_DIR, '**', '*.mat'), recursive=True)
    
    if not files:
        print(f"🛑 Error: No .mat files found in '{HIT_DIR}'. Verify your paths.")
        exit()
        
    print(f"🚀 Found {len(files)} files for the HIT Dataset. Launching parallel engine...")
    
    if os.path.exists(output_file):
        os.remove(output_file)
        
    cols = ['kurtosis', 'skewness', 'crest_factor', 'shape_factor', 
            'impulse_factor', 'margin_factor', 'peak_frequency', 'spectral_centroid', 'label']
            
    batch_rows = []
    max_workers = max(1, multiprocessing.cpu_count() - 2)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_hit_file, fp): fp for fp in files}
        
        for future in as_completed(futures):
            rows, log_msg = future.result()
            print(log_msg)
            if rows:
                batch_rows.extend(rows)
                
            # Periodic flush to disk to save RAM
            if len(batch_rows) >= 20000:
                df = pd.DataFrame(batch_rows, columns=cols)
                df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
                batch_rows = []
                
        # Final flush
        if batch_rows:
            df = pd.DataFrame(batch_rows, columns=cols)
            df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
            
    print(f"\n🎉 Completed! All HIT features safely written to: '{output_file}'")