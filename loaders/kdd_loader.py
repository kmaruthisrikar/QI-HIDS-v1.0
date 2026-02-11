import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class KDDLoader:
    def __init__(self, data_path='data/kdd'):
        self.data_path = Path(data_path)
        self.scaler = StandardScaler()
        # Full KDD Cup 1999 column list (41 features + 1 label)
        self.col_names = [
            "duration","protocol_type","service","flag","src_bytes",
            "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
            "logged_in","num_compromised","root_shell","su_attempted","num_root",
            "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
            "is_host_login","is_guest_login","count","srv_count","serror_rate",
            "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
            "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
            "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
            "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"
        ]

    def load_data(self, sample_size=None):
        print(f"Loading KDD Cup 1999 (Pure Physics) from {self.data_path}...")
        files = list(self.data_path.glob("*"))
        if not files:
             print("WARNING: No KDD data files found.")
             return np.zeros((1, 38)), np.zeros(1) 

        # Load first KDD file found
        df = pd.read_csv(files[0], names=self.col_names, header=None)
        
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)

        # Labels: Binary classification (Normal vs Attack)
        y = (df['label'] != 'normal.').astype(int).values
        
        # FEATURE SELECTION: "Not Dummy Columns"
        # We drop the categorical metadata (protocol, service, flag) 
        # to focus on pure traffic physics (bytes, counts, rates)
        X = df.drop(['label', 'protocol_type', 'service', 'flag'], axis=1)
        
        # Ensure only numeric data remains
        X_numeric = X.select_dtypes(include=[np.number])
        
        print(f"  KDD Features Extracted: {X_numeric.shape[1]} numeric dimensions.")
        print(f"  KDD Features Extracted: {X_numeric.shape[1]} numeric dimensions.")
        
        # USE UNIVERSAL LIBRARY SCALING
        from models.tunneling_lib import UniversalTunnelingNetwork
        return UniversalTunnelingNetwork.normalize(X_numeric.values), y
