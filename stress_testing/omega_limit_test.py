"""
QI-HIDS: OMEGA ROBUSTNESS EVALUATION (RESEARCH GRADE)

• No artificial metric rounding
• Multi-step PGD adversarial attack
• Works with models that output probabilities
• Cross-dataset baseline (CICIDS + KDD)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from colorama import init, Fore, Style

# --------------------------------------------------
# Setup
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from models.v1_engine import build_v1
from loaders.kdd_loader import KDDLoader
from loaders.cicids2017_loader import CICIDS2017Loader

init()
DEVICE = torch.device("cpu")

# --------------------------------------------------
# Evaluation Function
# --------------------------------------------------

def evaluate(model, X, y, era):
    with torch.no_grad():
        probs = model(X, era=era)  # Model already returns probabilities
        preds = torch.argmax(probs, dim=1)

        acc = (preds == y).float().mean().item()
        conf = probs.max(dim=1).values.mean().item()

    return acc, conf


# --------------------------------------------------
# PGD Adversarial Attack (Probability-based)
# --------------------------------------------------

def pgd_attack(model, X, y, era, epsilon=0.2, alpha=0.05, steps=7):
    X_orig = X.clone().detach()
    X_adv = X.clone().detach().requires_grad_(True)

    for _ in range(steps):
        probs = model(X_adv, era=era)
        log_probs = torch.log(probs + 1e-12)
        loss = nn.NLLLoss()(log_probs, y)

        loss.backward()

        with torch.no_grad():
            X_adv += alpha * X_adv.grad.sign()
            delta = torch.clamp(X_adv - X_orig, -epsilon, epsilon)
            X_adv = (X_orig + delta).detach().requires_grad_(True)

    return X_adv.detach()


# --------------------------------------------------
# Chaos Transformations
# --------------------------------------------------

def mask_features(x, drop_prob=0.8):
    return x * (torch.rand_like(x) > drop_prob).float()

def add_noise(x, scale=0.5):
    return x + torch.randn_like(x) * scale

def blackout_block(x, dims=30):
    dims = min(dims, x.shape[1])
    return torch.cat([torch.zeros(x.size(0), dims), x[:, dims:]], dim=1)

def correlated_drift(x, scale=0.5):
    return x + torch.randn_like(x) * scale * x.mean(dim=0)


# --------------------------------------------------
# Main Execution
# --------------------------------------------------

print(Fore.CYAN + "\nLoading datasets..." + Style.RESET_ALL)
kdd_loader = KDDLoader()
cic_loader = CICIDS2017Loader()

X_k, y_k = kdd_loader.load_data(sample_size=2000)
X_c, y_c = cic_loader.load_data(sample_size=2000)

X_k = torch.tensor(X_k, dtype=torch.float32).to(DEVICE)
y_k = torch.tensor(y_k, dtype=torch.long).to(DEVICE)
X_c = torch.tensor(X_c, dtype=torch.float32).to(DEVICE)
y_c = torch.tensor(y_c, dtype=torch.long).to(DEVICE)

print(Fore.CYAN + "Loading model..." + Style.RESET_ALL)
model = build_v1(legacy_dim=X_k.shape[1], modern_dim=X_c.shape[1])
model.load_state_dict(torch.load(ROOT_DIR / "models" / "tunneling_v1.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(Fore.GREEN + "Model loaded successfully.\n" + Style.RESET_ALL)

# --------------------------------------------------
# Baseline Performance
# --------------------------------------------------

print(Fore.YELLOW + "BASELINE PERFORMANCE" + Style.RESET_ALL)
acc_c, conf_c = evaluate(model, X_c, y_c, era='modern')
acc_k, conf_k = evaluate(model, X_k, y_k, era='legacy')

print(f"CICIDS  | Accuracy: {acc_c:.4f} | Confidence: {conf_c:.4f}")
print(f"KDD     | Accuracy: {acc_k:.4f} | Confidence: {conf_k:.4f}")

# --------------------------------------------------
# Chaos Robustness Tests
# --------------------------------------------------

scenarios = {
    "Telemetry Loss (80%)": lambda x: mask_features(x, 0.8),
    "Extreme Noise": lambda x: add_noise(x, 0.5),
    "Feature Blackout": lambda x: blackout_block(x, 30),
    "Correlated Drift": lambda x: correlated_drift(x, 0.5),
}

print(Fore.YELLOW + "\nCHAOS ROBUSTNESS TESTS (CICIDS)" + Style.RESET_ALL)
for name, fn in scenarios.items():
    X_mod = fn(X_c)
    acc, conf = evaluate(model, X_mod, y_c, era='modern')
    print(f"{name:<25} | Acc: {acc:.4f} | Conf: {conf:.4f}")

# --------------------------------------------------
# PGD Adversarial Attack
# --------------------------------------------------

print(Fore.MAGENTA + "\nPGD ADVERSARIAL ATTACK (CICIDS)" + Style.RESET_ALL)
X_adv = pgd_attack(model, X_c, y_c, era='modern', epsilon=0.2, alpha=0.05, steps=7)
acc_adv, conf_adv = evaluate(model, X_adv, y_c, era='modern')
print(f"PGD Attack | Accuracy: {acc_adv:.4f} | Confidence: {conf_adv:.4f}")

print(Fore.RED + Style.BRIGHT + "\nOMEGA ROBUSTNESS EVALUATION COMPLETE" + Style.RESET_ALL)
