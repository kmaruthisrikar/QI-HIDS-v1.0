"""
QI-HIDS ENTERPRISE PEAK & RESILIENCE TEST

Simulates real-world enterprise network stress:
• Throughput spikes
• Encryption-heavy traffic
• Sensor degradation
• Behavioral drift
• Low-and-slow attack patterns
• Hybrid adversarial noise
• Runtime stability metrics
"""

import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from colorama import init, Fore, Style

# --------------------------------------------------
# Setup
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from models.v1_engine import build_v1
from loaders.cicids2017_loader import CICIDS2017Loader

init()
DEVICE = torch.device("cpu")

# --------------------------------------------------
# Utility Metrics
# --------------------------------------------------

def evaluate_batch(model, X):
    start = time.time()
    with torch.no_grad():
        probs = model(X, era='modern')
        conf = probs.max(dim=1).values.mean().item()
    latency = (time.time() - start) * 1000  # ms
    return latency, conf


# --------------------------------------------------
# Traffic Transformations
# --------------------------------------------------

def simulate_encryption(x):
    return x + torch.randn_like(x) * 1.2  # high entropy noise

def simulate_packet_loss(x, drop_prob=0.3):
    return x * (torch.rand_like(x) > drop_prob).float()

def simulate_behavior_drift(x):
    scale = torch.linspace(0.8, 1.2, x.shape[1])
    return x * scale

def simulate_low_and_slow(x):
    x = x.clone()
    x[:, :5] *= 0.1  # suppress burst features
    x[:, 5:10] *= 1.5  # elongate timing-like features
    return x

def simulate_hybrid_attack(model, x):
    x_adv = x.clone().detach().requires_grad_(True)
    probs = model(x_adv, era='modern')
    loss = torch.log(probs + 1e-12).mean()
    loss.backward()
    return (x + 0.05 * x_adv.grad.sign()).detach()

# --------------------------------------------------
# Load Data & Model
# --------------------------------------------------

print(Fore.CYAN + "Loading dataset and model..." + Style.RESET_ALL)

loader = CICIDS2017Loader()
X, y = loader.load_data(sample_size=5000)
X = torch.tensor(X, dtype=torch.float32).to(DEVICE)

model = build_v1(legacy_dim=36, modern_dim=X.shape[1])
model.load_state_dict(torch.load(ROOT_DIR / "models" / "holographic_master_v1.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

print(Fore.GREEN + "System Ready.\n" + Style.RESET_ALL)

# --------------------------------------------------
# 1️⃣ Throughput Spike Test
# --------------------------------------------------

print(Fore.YELLOW + "THROUGHPUT SPIKE TEST" + Style.RESET_ALL)
for batch_size in [512, 1024, 2048, 4096]:
    batch = X[:batch_size]
    latency, conf = evaluate_batch(model, batch)
    print(f"Batch {batch_size:<4} | Latency: {latency:.2f} ms | Confidence: {conf:.4f}")

# --------------------------------------------------
# 2️⃣ Encryption Heavy Traffic
# --------------------------------------------------

print(Fore.YELLOW + "\nENCRYPTED TRAFFIC SIMULATION" + Style.RESET_ALL)
X_enc = simulate_encryption(X)
latency, conf = evaluate_batch(model, X_enc)
print(f"Encrypted Traffic | Latency: {latency:.2f} ms | Confidence: {conf:.4f}")

# --------------------------------------------------
# 3️⃣ Telemetry Degradation
# --------------------------------------------------

print(Fore.YELLOW + "\nTELEMETRY LOSS SIMULATION" + Style.RESET_ALL)
X_loss = simulate_packet_loss(X, 0.4)
latency, conf = evaluate_batch(model, X_loss)
print(f"40% Feature Loss | Latency: {latency:.2f} ms | Confidence: {conf:.4f}")

# --------------------------------------------------
# 4️⃣ Behavioral Drift
# --------------------------------------------------

print(Fore.YELLOW + "\nBEHAVIORAL DRIFT SIMULATION" + Style.RESET_ALL)
X_drift = simulate_behavior_drift(X)
latency, conf = evaluate_batch(model, X_drift)
print(f"Business Drift | Latency: {latency:.2f} ms | Confidence: {conf:.4f}")

# --------------------------------------------------
# 5️⃣ Low-and-Slow Attack Behavior
# --------------------------------------------------

print(Fore.YELLOW + "\nLOW-AND-SLOW ATTACK PATTERNS" + Style.RESET_ALL)
X_slow = simulate_low_and_slow(X)
latency, conf = evaluate_batch(model, X_slow)
print(f"Low & Slow | Latency: {latency:.2f} ms | Confidence: {conf:.4f}")

# --------------------------------------------------
# 6️⃣ Hybrid Adversarial + Noise
# --------------------------------------------------

print(Fore.YELLOW + "\nHYBRID ADVERSARIAL + ENCRYPTION" + Style.RESET_ALL)
X_hybrid = simulate_encryption(simulate_hybrid_attack(model, X))
latency, conf = evaluate_batch(model, X_hybrid)
print(f"Hybrid Attack | Latency: {latency:.2f} ms | Confidence: {conf:.4f}")

# --------------------------------------------------
# 7️⃣ Runtime Stability Loop
# --------------------------------------------------

print(Fore.YELLOW + "\nLONG-RUN STABILITY TEST (Simulated)" + Style.RESET_ALL)
conf_history = []

for i in range(20):  # simulate time steps
    X_step = simulate_behavior_drift(simulate_packet_loss(X, 0.2))
    _, conf = evaluate_batch(model, X_step)
    conf_history.append(conf)
    print(f"Step {i+1:02d} | Confidence: {conf:.4f}")
    time.sleep(0.2)

print(Fore.CYAN + f"\nConfidence Drift Range: {min(conf_history):.4f} → {max(conf_history):.4f}" + Style.RESET_ALL)

print(Fore.GREEN + Style.BRIGHT + "\nENTERPRISE PEAK STRESS TEST COMPLETE" + Style.RESET_ALL)
