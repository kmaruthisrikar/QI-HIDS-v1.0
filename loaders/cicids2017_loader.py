import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class CICIDS2017Loader:
    def __init__(self, data_path='data/cicids2017'):
        self.data_path = Path(data_path)
        self.scaler = StandardScaler()
        self.feature_names = None

    def load_data(self, sample_size=None):
        print(f"Loading CICIDS2017 from {self.data_path}...")
        csv_files = list(self.data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_path}")
        
        # Optimize: If we need a small sample, load fewer files or sample while loading
        if sample_size and sample_size < 100000:
            # Load only the first file to get a quick sample
            f = csv_files[0]
            df = pd.read_csv(f, encoding='utf-8', low_memory=False, nrows=sample_size * 2)
        else:
            dataframes = []
            for f in csv_files:
                df = pd.read_csv(f, encoding='utf-8', low_memory=False)
                df.columns = df.columns.str.strip()
                dataframes.append(df)
            df = pd.concat(dataframes, ignore_index=True)
        
        df.columns = df.columns.str.strip()
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
            
        # Separate Labels
        label_col = [col for col in df.columns if 'label' in col.lower()][0]
        y = (df[label_col].str.strip() != 'BENIGN').astype(int).values
        
        # Clean Features
        X_raw = df.drop([label_col], axis=1).select_dtypes(include=[np.number])
        X_clean = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        self.feature_names = X_clean.columns.tolist()
        self.feature_names = X_clean.columns.tolist()
        
        # USE UNIVERSAL LIBRARY SCALING
        from models.tunneling_lib import UniversalTunnelingNetwork
        return UniversalTunnelingNetwork.normalize(X_clean.values), y
