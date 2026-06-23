import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# =========================================================
# 1. CORE PROCESSING FUNCTIONS
# =========================================================
def calculate_8_features(signal):
    """Computes the 8 statistical features for a single 1D window."""
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

def parse_label_from_filename(filename):
    """Dynamically identifies the fault class based on SDUST naming conventions."""
    fn_upper = filename.upper()
    if "NC" in fn_upper or "NORMAL" in fn_upper:
        return "normal"
    elif "IF" in fn_upper:
        return "inner_race_fault"
    elif "OF" in fn_upper:
        return "outer_race_fault"
    elif "RF" in fn_upper:
        return "roller_fault"
    return "unknown"

def process_single_file(file_path, window_size=2048, step_size=1024):
    """Processes a single file and returns a list of feature rows."""
    base_name = os.path.basename(file_path)
    label = parse_label_from_filename(base_name)
    
    try:
        mat_dict = sio.loadmat(file_path)
        signal_array = mat_dict['Signal'][0, 0]['y_values'][0, 0]['values'].flatten()
        
        if signal_array.dtype == 'O' and isinstance(signal_array[0], np.ndarray):
            signal_array = signal_array[0].flatten()

        if len(signal_array) < window_size:
            return None, f"[SKIP] Size too small ({len(signal_array)}) in: {base_name}"
            
    except Exception as e:
        return None, f"[ERROR] Failed parsing struct in {base_name}: {str(e)}"
        
    feature_rows = []
    for start in range(0, len(signal_array) - window_size, step_size):
        window = signal_array[start:start + window_size]
        features = calculate_8_features(window)
        feature_rows.append(features + [label])
        
    if not feature_rows:
        return None, f"[SKIP] No windows generated for: {base_name}"
        
    return feature_rows, f"[SUCCESS] Extracted {len(feature_rows)} windows from {base_name}"

# =========================================================
# 2. MULTIPROCESSING PIPELINE RUNNER
# =========================================================
if __name__ == '__main__':
    # Fix for Windows multiprocessing environments
    multiprocessing.freeze_support()
    
    RAW_SDUST_DIR = r"D:\DATABASEYO\SDUSTdataset\SDUST-Dataset-main"
    OUTPUT_CSV = r"D:\DATABASEYO\results\sdust_master_results.csv"
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    print(f"Scanning for SDUST .mat files recursively in: {RAW_SDUST_DIR}...")
    mat_files = glob.glob(os.path.join(RAW_SDUST_DIR, "**", "*.mat"), recursive=True)
    
    if not mat_files:
        print("❌ No .mat files found! Verify your RAW_SDUST_DIR path.")
        exit()
        
    print(f"Found {len(mat_files)} files. Spawning parallel workers across CPU threads...")
    
    # Define columns for the final dataframe
    cols = ['kurtosis', 'skewness', 'crest_factor', 'shape_factor', 
            'impulse_factor', 'margin_factor', 'peak_frequency', 'spectral_centroid', 'label']
            
    # Remove existing master file to avoid appending old data onto new runs
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
        
    # Set worker pool size (leaving 2 threads open so your computer stays responsive)
    max_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Running pipeline using {max_workers} parallel workers.")

    file_counter = 0
    
    # Process files in parallel batches
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, fp): fp for fp in mat_files}
        
        batch_rows = []
        for future in as_completed(futures):
            rows, log_msg = future.result()
            print(log_msg)
            
            if rows:
                batch_rows.extend(rows)
                
            # Write out data in batches to keep memory footprints low
            if len(batch_rows) >= 50000:
                df_batch = pd.DataFrame(batch_rows, columns=cols)
                file_exists = os.path.exists(OUTPUT_CSV)
                df_batch.to_csv(OUTPUT_CSV, mode='a', index=False, header=not file_exists)
                batch_rows = [] # Clear memory cache
                
        # Write any remaining rows left in the final batch
        if batch_rows:
            df_batch = pd.DataFrame(batch_rows, columns=cols)
            file_exists = os.path.exists(OUTPUT_CSV)
            df_batch.to_csv(OUTPUT_CSV, mode='a', index=False, header=not file_exists)

    print(f"\n🚀 Complete! Consolidated feature matrix saved to: {OUTPUT_CSV}")