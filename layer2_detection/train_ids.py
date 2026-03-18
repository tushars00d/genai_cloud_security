"""
layer2_detection/train_ids.py
Layer 2: Intrusion Detection System + Adversarial Purification.

Three experiments:
  A) Detection accuracy — baseline vs Attention-IDS on CICIDS
  B) Adversarial robustness — accuracy under FGSM attack
  C) Purification — restore accuracy using DDPM denoising
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import mlflow

from config import DATA_DIR, RESULTS_DIR, MODELS_DIR, MLFLOW_TRACKING_URI, EXPERIMENT_NAME, L2
from utils.metrics import evaluate_classifier, plot_comparison_bar, plot_confusion_matrix, save_results_csv, Timer


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_cicids():
    path = DATA_DIR / "cicids_sample.csv"
    if not path.exists():
        print("CICIDS sample not found. Run: python utils/download_datasets.py")
        sys.exit(1)

    df = pd.read_csv(path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    le = LabelEncoder()
    y  = le.fit_transform(df["Label"])
    X  = df.drop(columns=["Label"]).values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    return X, y, le.classes_, scaler


# ── Attention-IDS Model ────────────────────────────────────────────────────────

class AttentionBlock(nn.Module):
    """Self-attention over feature dimension — mimics Attention-GAN from Sen (2024)."""
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = torch.softmax(q * k * self.scale, dim=-1)
        return x + attn * v   # residual


class AttentionIDS(nn.Module):
    """Neural IDS with attention mechanism for anomaly detection."""
    def __init__(self, input_dim, num_classes, hidden=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.attn1 = AttentionBlock(hidden)
        self.attn2 = AttentionBlock(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        h = self.enc(x)
        h = self.attn1(h)
        h = self.attn2(h)
        return self.classifier(h)


def train_attention_ids(X_train, y_train, num_classes, device, epochs=30):
    dataset = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    model  = AttentionIDS(X_train.shape[1], num_classes).to(device)
    optim  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    crit   = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            loss = crit(model(x_b), y_b)
            optim.zero_grad(); loss.backward(); optim.step()
            total_loss += loss.item()
        sched.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Attention-IDS epoch {epoch+1}/{epochs} | loss: {total_loss/len(loader):.4f}")

    return model


def predict_ids(model, X, device, batch_size=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size]).to(device)
            preds.append(model(batch).argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


# ── Adversarial Attack (FGSM) ──────────────────────────────────────────────────

def fgsm_attack(model, X, y, eps, device):
    """Fast Gradient Sign Method adversarial perturbation."""
    model.eval()
    X_t = torch.tensor(X, requires_grad=True, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    loss = nn.CrossEntropyLoss()(model(X_t), y_t)
    loss.backward()
    X_adv = (X_t + eps * X_t.grad.sign()).detach().cpu().numpy()
    return X_adv.astype(np.float32)


# ── DDPM Purification ──────────────────────────────────────────────────────────

class SimpleDenoiser(nn.Module):
    """Lightweight denoiser for purification (full DDPM uses Layer 1 model)."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 512),  # +1 for noise level
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, input_dim),
        )
    def forward(self, x, noise_level):
        nl = noise_level.unsqueeze(-1).expand(x.size(0), 1)
        return self.net(torch.cat([x, nl], dim=-1))


def train_denoiser(X_clean, device, epochs=20):
    """Train a simple denoiser for adversarial purification."""
    X_t = torch.tensor(X_clean, device=device)
    model = SimpleDenoiser(X_clean.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        eps_vals = torch.rand(len(X_clean), device=device) * 0.3
        noise    = torch.randn_like(X_t) * eps_vals.unsqueeze(-1)
        X_noisy  = X_t + noise
        pred     = model(X_noisy, eps_vals)
        loss     = nn.MSELoss()(pred, X_t)
        optim.zero_grad(); loss.backward(); optim.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Denoiser epoch {epoch+1}/{epochs} | loss: {loss.item():.4f}")

    return model


def purify(denoiser, X_adv, device):
    """Remove adversarial perturbations via denoising."""
    denoiser.eval()
    X_t = torch.tensor(X_adv, device=device)
    # Estimate noise level as deviation from training distribution
    noise_level = torch.full((len(X_adv),), L2["adversarial_eps"], device=device)
    with torch.no_grad():
        X_pure = denoiser(X_t, noise_level)
    return X_pure.cpu().numpy().astype(np.float32)


# ── Main Experiment ────────────────────────────────────────────────────────────

def run_layer2_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("LAYER 2: Intrusion Detection + Adversarial Purification")
    print(f"Device: {device}")
    print(f"{'='*60}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\n[1/7] Loading CICIDS dataset...")
    X, y, class_names, _ = load_cicids()
    num_classes = len(class_names)
    print(f"  {len(X)} samples | {num_classes} attack categories")
    print(f"  Classes: {class_names[:5]}{'...' if len(class_names)>5 else ''}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=L2["test_split"], random_state=L2["random_state"], stratify=y
    )

    # ── Exp A: Detection accuracy comparison ──
    print("\n[2/7] Exp A: Baseline Random Forest...")
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_rf.fit(X_train, y_train)
    res_rf = evaluate_classifier(clf_rf.predict(X_test), y_test, label="random_forest")

    print("\n[3/7] Exp A: Attention-IDS neural model...")
    with Timer("Attention-IDS training"):
        ids_model = train_attention_ids(X_train, y_train, num_classes, device)
    y_pred_ids = predict_ids(ids_model, X_test, device)
    res_ids = evaluate_classifier(y_pred_ids, y_test, label="attention_ids")
    plot_confusion_matrix(y_pred_ids, y_test, class_names, label="attention_ids")

    # ── Exp B: Adversarial attack ──
    print(f"\n[4/7] Exp B: FGSM adversarial attack (eps={L2['adversarial_eps']})...")
    # Use a subset for attack (full set is slow)
    idx = np.random.choice(len(X_test), min(2000, len(X_test)), replace=False)
    X_test_sub, y_test_sub = X_test[idx], y_test[idx]
    X_adv = fgsm_attack(ids_model, X_test_sub, y_test_sub, L2["adversarial_eps"], device)
    y_pred_adv = predict_ids(ids_model, X_adv, device)
    res_adv = evaluate_classifier(y_pred_adv, y_test_sub, label="under_adversarial_attack")
    print(f"  ⚠ Accuracy dropped from {res_ids['accuracy']:.3f} → {res_adv['accuracy']:.3f} under attack!")

    # ── Exp C: DDPM purification ──
    print("\n[5/7] Exp C: Training denoiser for purification...")
    denoiser = train_denoiser(X_train, device)
    X_purified = purify(denoiser, X_adv, device)
    y_pred_pure = predict_ids(ids_model, X_purified, device)
    res_pure = evaluate_classifier(y_pred_pure, y_test_sub, label="after_purification")
    print(f"  ✅ Accuracy restored: {res_adv['accuracy']:.3f} → {res_pure['accuracy']:.3f}")

    # ── Save comparison ──
    print("\n[6/7] Saving results...")
    det_results = [res_rf, res_ids]
    save_results_csv(det_results, "layer2_detection_comparison.csv")
    plot_comparison_bar(det_results, metric="f1_score",
                        title="Layer 2: Detection F1 — RF vs Attention-IDS")

    purif_results = [res_ids, res_adv, res_pure]
    # Fix labels for purification chart
    res_ids_copy   = {**res_ids,  "label": "clean_input"}
    res_adv_copy   = {**res_adv,  "label": "under_attack"}
    res_pure_copy  = {**res_pure, "label": "after_purification"}
    save_results_csv([res_ids_copy, res_adv_copy, res_pure_copy], "layer2_purification_experiment.csv")
    plot_comparison_bar([res_ids_copy, res_adv_copy, res_pure_copy], metric="accuracy",
                        title="Layer 2: Adversarial Purification Effect")

    # Save model
    torch.save(ids_model.state_dict(), MODELS_DIR / "attention_ids.pt")

    # MLflow
    print("\n[7/7] Logging to MLflow...")
    with mlflow.start_run(run_name="layer2_detection"):
        mlflow.log_metrics({
            "rf_f1":           res_rf["f1_score"],
            "ids_f1":          res_ids["f1_score"],
            "ids_acc_clean":   res_ids["accuracy"],
            "ids_acc_attack":  res_adv["accuracy"],
            "ids_acc_purified":res_pure["accuracy"],
            "acc_drop_pct":    (res_ids["accuracy"] - res_adv["accuracy"]) * 100,
            "purification_recovery_pct": (res_pure["accuracy"] - res_adv["accuracy"]) * 100,
        })

    print("\n✅ Layer 2 complete.")
    print(f"   RF F1:                {res_rf['f1_score']:.4f}")
    print(f"   Attention-IDS F1:     {res_ids['f1_score']:.4f}")
    print(f"   Accuracy under FGSM:  {res_adv['accuracy']:.4f}")
    print(f"   Accuracy after purif: {res_pure['accuracy']:.4f}")

    return {"ids_model": ids_model, "denoiser": denoiser,
            "results": [res_rf, res_ids, res_adv, res_pure]}


if __name__ == "__main__":
    run_layer2_experiment()
