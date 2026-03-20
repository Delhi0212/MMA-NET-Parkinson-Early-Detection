import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import glob

# ==========================================
# 1. ROBUST PATH CONFIGURATION (FIXED)
# ==========================================
# Get the folder where this script is located (G:\VOICE_GAIT_PROJECT)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_CONFIG = {
    # CRITICAL FIX: We look inside the 'dataset' folder
    'uci_path': os.path.join(BASE_DIR, 'dataset', 'parkinsons.data'), 
    'gait_path': os.path.join(BASE_DIR, 'dataset', 'gaitpdb'),            
    'window_size': 300,                          
    'gait_stride': 150,                          
    'batch_size': 32,
    'test_split': 0.2,
    'seed': 42
}

# --- Auto-correction for Windows hidden extensions ---
# If 'parkinsons.data' isn't found, check if it's named 'parkinsons.data.txt'
if not os.path.exists(DATA_CONFIG['uci_path']):
    possible_txt = os.path.join(BASE_DIR, 'dataset', 'parkinsons.data.txt')
    if os.path.exists(possible_txt):
        print(f"⚠️ Note: Found file with .txt extension. Using that.")
        DATA_CONFIG['uci_path'] = possible_txt

# Final Check
if not os.path.exists(DATA_CONFIG['uci_path']):
    print(f"\n❌ CRITICAL ERROR: Could not find the file at: {DATA_CONFIG['uci_path']}")
    print(f"   I am looking in: {os.path.join(BASE_DIR, 'dataset')}")
    print("   Please ensure 'parkinsons.data' is inside the 'dataset' folder.\n")
    exit()

# ==========================================
# 2. VOICE STREAM PREPROCESSING (UCI)
# ==========================================
class VoicePreprocessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        
    def load_process(self):
        print(f"[Voice] Loading data from {self.csv_path}...")
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            exit()
        
        # 'status' is the target: 1 = PD, 0 = Healthy
        y = df['status'].values
        
        # Drop non-feature columns (name, status)
        X = df.drop(['name', 'status'], axis=1).values
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Separate into PD and Healthy pools
        X_pd = X_scaled[y == 1]
        X_hc = X_scaled[y == 0]
        
        print(f"[Voice] Processed: {len(X_pd)} PD samples, {len(X_hc)} Healthy samples.")
        print(f"[Voice] Feature Dimension: {X.shape[1]}")
        return X_pd, X_hc

# ==========================================
# 3. GAIT STREAM PREPROCESSING (PhysioNet)
# ==========================================
class GaitPreprocessor:
    def __init__(self, data_dir, window_size, stride):
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        
    def load_process(self):
        print(f"[Gait] Loading .txt files from {self.data_dir}...")
        
        # Check if folder exists
        if not os.path.exists(self.data_dir):
            print(f"⚠️ Warning: Gait folder not found at {self.data_dir}")
            print("   (This is expected if you haven't downloaded PhysioNet data yet)")
            # Return dummy data so the script doesn't crash
            return np.zeros((10, self.window_size, 16)), np.zeros((10, self.window_size, 16))

        pd_files = glob.glob(os.path.join(self.data_dir, "*Pt*.txt"))
        hc_files = glob.glob(os.path.join(self.data_dir, "*Co*.txt"))
        
        if not pd_files:
            print("⚠️ No Gait files found (Download from PhysioNet and put in dataset/gaitpdb/)")
            # Return dummy data so script runs
            return np.zeros((10, self.window_size, 16)), np.zeros((10, self.window_size, 16))

        X_pd_windows = self.process_files(pd_files)
        X_hc_windows = self.process_files(hc_files)
        
        print(f"[Gait] Processed: {len(X_pd_windows)} PD windows, {len(X_hc_windows)} Healthy windows.")
        return X_pd_windows, X_hc_windows

    def process_files(self, file_list):
        windows = []
        for fpath in file_list:
            try:
                # PhysioNet format: Time, L1..L8, R1..R8
                df = pd.read_csv(fpath, sep='\t', header=None, on_bad_lines='skip')
                
                # Check for header
                if isinstance(df.iloc[0,0], str): 
                    df = df.iloc[1:].astype(float)
                
                # Select Sensor Columns (1 to 16)
                sensor_data = df.iloc[:, 1:17].values 
                
                # Sliding Window
                for i in range(0, len(sensor_data) - self.window_size, self.stride):
                    window = sensor_data[i : i + self.window_size]
                    if window.shape == (self.window_size, 16):
                        windows.append(window)
            except Exception as e:
                print(f"Skipping {fpath}: {e}")
                
        if len(windows) == 0:
            return np.zeros((10, self.window_size, 16)) # Dummy return
            
        return np.array(windows)

# ==========================================
# 4. MULTIMODAL DATASET GENERATOR
# ==========================================
class MMADataset(Dataset):
    def __init__(self, voice_data, gait_data, label, size_multiplier=10):
        self.voice_data = torch.FloatTensor(voice_data)
        self.gait_data = torch.FloatTensor(gait_data)
        self.label = torch.LongTensor([label])
        self.size_multiplier = size_multiplier 
        self.length = max(len(voice_data), len(gait_data)) * size_multiplier

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Random Pairing
        v_idx = np.random.randint(0, len(self.voice_data))
        g_idx = np.random.randint(0, len(self.gait_data))
        
        return {
            'voice': self.voice_data[v_idx],       
            'gait': self.gait_data[g_idx],         
            'label': self.label                    
        }

# ==========================================
# 5. EXECUTION PIPELINE
# ==========================================
def get_dataloaders():
    # 1. Process Voice
    vp = VoicePreprocessor(DATA_CONFIG['uci_path'])
    v_pd, v_hc = vp.load_process()
    
    # 2. Process Gait
    gp = GaitPreprocessor(DATA_CONFIG['gait_path'], DATA_CONFIG['window_size'], DATA_CONFIG['gait_stride'])
    g_pd, g_hc = gp.load_process()
    
    # 3. Create Datasets
    dataset_pd = MMADataset(v_pd, g_pd, label=1)
    dataset_hc = MMADataset(v_hc, g_hc, label=0)
    
    # 4. Merge and Split
    full_dataset = torch.utils.data.ConcatDataset([dataset_pd, dataset_hc])
    
    train_size = int((1 - DATA_CONFIG['test_split']) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=DATA_CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=DATA_CONFIG['batch_size'], shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    train_dl, test_dl = get_dataloaders()
    sample = next(iter(train_dl))
    print("\n[Sanity Check] Batch Shapes:")
    print("Voice:", sample['voice'].shape) # Expected: [32, 22]
    print("Gait:", sample['gait'].shape)   # Expected: [32, 300, 16]
    print("Label:", sample['label'].shape) # Expected: [32, 1]