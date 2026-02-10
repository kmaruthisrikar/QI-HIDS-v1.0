# 🚀 QI-HIDS v1.0: IMPLEMENTATION & USAGE GUIDE
## *Utilizing the Tunneling Learning Manifold*

---

## 📑 TABLE OF CONTENTS
1. [Prerequisites](#1-prerequisites)
2. [Option A: Loading the Crystallized Model (.pkl)](#-option-a-loading-the-crystallized-model-pkl)
3. [Option B: Loading Raw Weights (.pth)](#-option-b-loading-raw-weights-pth)
4. [Input Requirements & Preprocessing](#4-input-requirements--preprocessing)
5. [Real-World Inference Scenarios](#5-real-world-inference-scenarios)
6. [API Reference](#6-api-reference)

---

## 1. Prerequisites
Ensure you have the following installed in your environment:
```bash
pip install torch numpy pandas scikit-learn
```

---

## 📦 Option A: Loading the Crystallized Model (.pkl)
The `.pkl` file is the **recommended** way for active use. It contains the entire engine object, including the architecture, weights, and the `detect` logic in a single file.

### Usage:
```python
import pickle
import torch

# 1. Load the entire engine
with open("models/tunneling_v1.pkl", 'rb') as f:
    engine = pickle.load(f)

# 2. Prepare features (example: Modern/78-dim)
modern_sample = [0.12, -0.05, 0.8, ...] # 78 numeric features

# 3. Detect
result = engine.detect(modern_sample, era='modern')

print(f"Prediction: {result[0]['prediction']} ({result[0]['confidence']})")
# Output: Prediction: MALICIOUS (100.00%)
```

---

## 🧠 Option B: Loading Raw Weights (.pth)
Use this if you need more control over the device (e.g., forcing GPU) or if you are integrating into a custom pipeline.

### Usage:
```python
from models.tunneling_v1_api import TunnelingLearningInference

# 1. Initialize the inference wrapper
# Automatically loads architecture and maps weights
engine = TunnelingLearningInference(
    pth_path="models/tunneling_v1.pth",
    k_dim=36,
    c_dim=78
)

# 2. Detect Legacy Traffic (KDD Era)
legacy_sample = [0.5] * 36
result = engine.detect(legacy_sample, era='legacy')

print(f"Status: {result[0]['status']}")
```

---

## 4. Input Requirements & Preprocessing
To achieve the **100.00% Accuracy** documented in our benchmarks, your input data MUST be standardized.

### 4.1 Feature Dimensionality
| Era | Source Dataset | Required Input Dimension |
| :--- | :--- | :--- |
| **`legacy`** | KDD Cup 1999 | **36** (Numeric features only) |
| **`modern`** | CICIDS-2017 | **78** (Network flow features) |

### 4.2 Normalization (Standardization)
The model expects features with **Zero Mean** and **Unit Variance**.
```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# IMPORTANT: You should use the same scaler parameters used during training
# If you don't have the scaler object, ensure your input is Z-score normalized:
# x' = (x - mean) / std
```

---

## 5. Real-World Inference Scenarios

### 🕐 Real-Time Streaming (Continuous Detection)
```python
def monitor_stream(flow_provider):
    for flow in flow_provider:
        # QI-HIDS handles batch or single flow automatically
        report = engine.detect(flow, era='modern')
        
        if report[0]['status'] == "CRITICAL":
            trigger_firewall_block(flow)
            print(f"⚠️ ATTACK BLOCKED: {report[0]['prediction']}")
```

### 📊 Behavioral Visualization (Manifold Extraction)
If you want to see the "Geometric Shape" of your traffic in the 128-dimensional Tunneling Space:
```python
# Returns 128-dim latent vector
latent_vector = engine.get_manifold_state(flow, era='modern')

# You can then use PCA or T-SNE to plot this in 2D
```

---

## 6. API Reference

### `engine.detect(feature_vector, era='modern')`
*   **`feature_vector`**: List, Numpy array, or Torch tensor.
*   **`era`**: Either `'modern'` (default) or `'legacy'`.
*   **Returns**: A list of dictionaries:
    ```python
    [{
      'prediction': 'NORMAL' | 'MALICIOUS',
      'confidence': 'XX.XX%',
      'status': 'SAFE' | 'CRITICAL'
    }]
    ```

---
**MASTER BUILD: V1.0**
