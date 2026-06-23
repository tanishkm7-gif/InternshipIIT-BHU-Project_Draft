import os
import glob
import numpy as np
import pandas as pd
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
# 2. INTELLIGENT AUTOMATIC ROUTING MACHINE
# =========================================================
def process_thu_file(file_path, window_size=2048, step_size=1024):
    base_name = os.path.basename(file_path)
    fp_upper = file_path.upper()
    
    # 🎯 Label routing rules based on combined directory tags
    if any(k in fp_upper for k in ["SHORT", "ELEC", "STATOR", "UNBALANCE", "VOLTAGE", "CURRENT", "AMP", "ROTOR"]):
        label = "electrical_fault"
        target_keyword = "i"  # Target phase current sensors for electrical tracks
    elif any(k in fp_upper for k in ["GEAR", "PITTING", "WEAR", "CHIPPED", "TEETH", "BREAK", "CRACK"]):
        label = "gear_fault"
        target_keyword = "x"  # Target mechanical vibration channel for gears
    elif any(k in fp_upper for k in ["NOMINAL", "HEALTHY", "HEALTH", "NORMAL"]):
        label = "nominal"
        target_keyword = "x"
    else:
        label = "bearing_fault"
        target_keyword = "x"
        
    try:
        # Scan headers to find matching sensors regardless of case or spacing
        peek_df = pd.read_csv(file_path, nrows=1)
        actual_cols = list(peek_df.columns)
        normalized_cols = [str(c).strip().lower() for c in actual_cols]
        
        matched_col = None
        for orig_col, norm_col in zip(actual_cols, normalized_cols):
            if target_keyword == "i" and ("i1" in norm_col or "current" in norm_col or "phase" in norm_col or "cur" in norm_col):
                matched_col = orig_col
                break
            elif target_keyword == "x" and (norm_col == "x" or "vib" in norm_col or "acc" in norm_col or "de" in norm_col):
                matched_col = orig_col
                break
                
        if not matched_col:
            matched_col = actual_cols[0]
            
        df = pd.read_csv(file_path, usecols=[matched_col])
        signal = df[matched_col].to_numpy().flatten()
        
        feature_rows = []
        is_electrical = 1 if target_keyword == "i" else 0
        
        for start in range(0, len(signal) - window_size, step_size):
            window = signal[start:start + window_size]
            if len(window) == window_size and not np.isnan(window).any():
                features = calculate_8_features(window)
                # Combined row formatting matching the 10 target features column layout
                feature_rows.append(features + [is_electrical, label])
                
        if not feature_rows:
            return None, f"⚠️ [SKIP] Empty frames windowed in: {base_name}"
            
        return feature_rows, f"✅ [EXTRACTED] {len(feature_rows)} frames ({label} via '{matched_col}') from {base_name}"
        
    except Exception as e:
        return None, f"❌ [CRASH] Processing failure on {base_name}: {str(e)}"

# =========================================================
# 3. RUNTIME MANAGER
# =========================================================
if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # Paths pointing to your unified directory
    THU_ROOT_DIR = r"D:\DATABASEYO\Multi-mode Fault Diagnosis Datasets of Gearbox Under Variable Working Conditions"
    OUTPUT_CSV_DIR = r"D:\DATABASEYO\results"
    output_file = os.path.join(OUTPUT_CSV_DIR, "mcc5_thu_targeted_results.csv")
    
    files = glob.glob(os.path.join(THU_ROOT_DIR, '**', '*.csv'), recursive=True)
    
    if not files:
        print(f"🛑 Error: No files detected in '{THU_ROOT_DIR}'!")
        exit()
        
    print(f"🚀 Processing {len(files)} combined files (Multi-Modal Mapping Active)...")
    
    if os.path.exists(output_file):
        os.remove(output_file)
        
    cols = ['kurtosis', 'skewness', 'crest_factor', 'shape_factor', 
            'impulse_factor', 'margin_factor', 'peak_frequency', 'spectral_centroid', 
            'is_electrical', 'label']
            
    batch_rows = []
    max_workers = max(1, multiprocessing.cpu_count() - 2)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_thu_file, fp): fp for fp in files}
        
        for future in as_completed(futures):
            rows, log_msg = future.result()
            print(log_msg)
            if rows:
                batch_rows.extend(rows)
                
            if len(batch_rows) >= 20000:
                df_out = pd.DataFrame(batch_rows, columns=cols)
                df_out.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
                batch_rows = []
                
        if batch_rows:
            df_out = pd.DataFrame(batch_rows, columns=cols)
            df_out.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
            
    print(f"\n🎉 Extraction Complete! Combined matrices written to: {output_file}")