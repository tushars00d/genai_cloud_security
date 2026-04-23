"""
layer1_data/train_ddpm.py
Layer 1: Synthetic data generation using DDPM + GAN.
Fixed: all model dimensions are computed dynamically from actual data shape.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import mlflow
import json

from config import DATA_DIR, RESULTS_DIR, MODELS_DIR, MLFLOW_TRACKING_URI, EXPERIMENT_NAME, L1
from utils.metrics import evaluate_classifier, plot_comparison_bar, save_results_csv, Timer


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_nsl_kdd():
    path = DATA_DIR / "nsl_kdd" / "train.csv"
    if not path.exists():
        print("NSL-KDD not found. Run: python utils/download_datasets.py")
        sys.exit(1)

    df = pd.read_csv(path)
    categorical = ["protocol_type", "service", "flag"]
    for col in categorical:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    drop_cols = [c for c in ("label", "attack_category", "difficulty") if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values.astype(np.float32)
    
    le = LabelEncoder()
    y  = le.fit_transform(df["attack_category"])
    print(f"  Input feature dimension: {X.shape[1]}")
    return X, y, le.classes_


# ── DDPM Components ────────────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half   = self.dim // 2
        emb    = torch.log(torch.tensor(10000.0)) / (half - 1)
        emb    = torch.exp(torch.arange(half, device=device) * -emb)
        emb    = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TabularDDPM(nn.Module):
    """
    Lightweight DDPM denoiser for tabular data.
    input_dim is passed explicitly so dimensions always match.
    """
    def __init__(self, input_dim, time_dim=64, hidden_dim=256, num_classes=5):
        super().__init__()
        self.input_dim = input_dim
        self.time_emb  = SinusoidalPosEmb(time_dim)
        self.class_emb = nn.Embedding(num_classes, time_dim)

        # The denoiser conditions on the sum of time and class embeddings.
        net_input = input_dim + time_dim

        self.net = nn.Sequential(
            nn.Linear(net_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, t, class_labels):
        t_emb = self.time_emb(t)           # (B, time_dim)
        c_emb = self.class_emb(class_labels)  # (B, time_dim)
        cond  = t_emb + c_emb              # (B, time_dim)
        inp   = torch.cat([x, cond], dim=-1)  # (B, input_dim + time_dim)
        return self.net(inp)


class DDPMSampler:
    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T      = T
        self.device = device
        betas            = torch.linspace(beta_start, beta_end, T, device=device)
        alphas           = 1.0 - betas
        alphas_cumprod   = torch.cumprod(alphas, dim=0)
        self.betas               = betas
        self.sqrt_ac             = alphas_cumprod.sqrt()
        self.sqrt_one_minus_ac   = (1 - alphas_cumprod).sqrt()
        self.sqrt_recip_alphas   = (1.0 / alphas).sqrt()
        prev_ac                  = torch.cat([alphas_cumprod[:1], alphas_cumprod[:-1]])
        self.posterior_var       = betas * (1 - prev_ac) / (1 - alphas_cumprod)

    def add_noise(self, x0, t):
        noise       = torch.randn_like(x0)
        sqrt_ac     = self.sqrt_ac[t].unsqueeze(-1)
        sqrt_1mac   = self.sqrt_one_minus_ac[t].unsqueeze(-1)
        return sqrt_ac * x0 + sqrt_1mac * noise, noise

    @torch.no_grad()
    def sample(self, model, shape, class_labels):
        x = torch.randn(shape, device=self.device)
        for i in tqdm(reversed(range(self.T)), desc="DDPM sampling",
                      total=self.T, leave=False):
            t         = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            pred_noise = model(x, t, class_labels)
            coef1      = self.sqrt_recip_alphas[i]
            coef2      = self.betas[i] / self.sqrt_one_minus_ac[i]
            x          = coef1 * (x - coef2 * pred_noise)
            if i > 0:
                x += self.posterior_var[i].sqrt() * torch.randn_like(x)
        return x


# ── GAN Components ─────────────────────────────────────────────────────────────

class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim, num_classes, embed_dim=32):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(noise_dim + embed_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        return self.net(torch.cat([z, self.label_emb(labels)], dim=-1))


class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes, embed_dim=32):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + embed_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        return self.net(torch.cat([x, self.label_emb(labels)], dim=-1))


# ── Training ───────────────────────────────────────────────────────────────────

def train_ddpm(X_train, y_train, num_classes, device):
    T         = L1["ddpm_timesteps"]
    epochs    = L1["ddpm_epochs"]
    bs        = L1["ddpm_batch_size"]
    input_dim = X_train.shape[1]          # ← dynamic, never hardcoded

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_train).astype(np.float32)

    dataset = TensorDataset(
        torch.tensor(X_norm),
        torch.tensor(y_train, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)

    # Pass input_dim explicitly
    model   = TabularDDPM(input_dim=input_dim, num_classes=num_classes).to(device)
    sampler = DDPMSampler(T=T, device=device)
    optim   = torch.optim.Adam(model.parameters(), lr=2e-4)

    print(f"  DDPM model: input_dim={input_dim}, params={sum(p.numel() for p in model.parameters()):,}")

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            t       = torch.randint(0, T, (x_batch.size(0),), device=device)
            x_noisy, noise = sampler.add_noise(x_batch, t)
            pred    = model(x_noisy, t, y_batch)
            loss    = nn.MSELoss()(pred, noise)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        losses.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"  DDPM epoch {epoch+1}/{epochs} | loss: {avg:.4f}")

    return model, sampler, scaler, losses


def generate_ddpm_samples(model, sampler, scaler, num_classes, n_per_class, input_dim, device):
    model.eval()
    all_X, all_y = [], []
    for cls in range(num_classes):
        labels  = torch.full((n_per_class,), cls, dtype=torch.long, device=device)
        samples = sampler.sample(model, (n_per_class, input_dim), labels)
        samples = scaler.inverse_transform(samples.cpu().numpy())
        all_X.append(samples)
        all_y.extend([cls] * n_per_class)
    return np.vstack(all_X), np.array(all_y)


def train_gan(X_train, y_train, num_classes, device):
    epochs    = L1["gan_epochs"]
    bs        = L1["gan_batch_size"]
    noise_d   = L1["noise_dim"]
    input_dim = X_train.shape[1]          # ← dynamic

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_train).astype(np.float32)

    dataset = TensorDataset(
        torch.tensor(X_norm),
        torch.tensor(y_train, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)

    G     = Generator(noise_d, input_dim, num_classes).to(device)
    D     = Discriminator(input_dim, num_classes).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    crit  = nn.BCELoss()

    for epoch in range(epochs):
        for x_real, y_batch in loader:
            x_real   = x_real.to(device)
            y_batch  = y_batch.to(device)
            bs_actual = x_real.size(0)
            real_lbl = torch.ones(bs_actual, 1, device=device)
            fake_lbl = torch.zeros(bs_actual, 1, device=device)

            z      = torch.randn(bs_actual, noise_d, device=device)
            x_fake = G(z, y_batch)
            d_loss = (crit(D(x_real, y_batch), real_lbl) +
                      crit(D(x_fake.detach(), y_batch), fake_lbl))
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            z      = torch.randn(bs_actual, noise_d, device=device)
            x_fake = G(z, y_batch)
            g_loss = crit(D(x_fake, y_batch), real_lbl)
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

        if (epoch + 1) % 10 == 0:
            print(f"  GAN epoch {epoch+1}/{epochs} | "
                  f"D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

    return G, scaler


def generate_gan_samples(G, scaler, num_classes, n_per_class, device):
    G.eval()
    all_X, all_y = [], []
    with torch.no_grad():
        for cls in range(num_classes):
            z      = torch.randn(n_per_class, L1["noise_dim"], device=device)
            labels = torch.full((n_per_class,), cls, dtype=torch.long, device=device)
            samples = G(z, labels).cpu().numpy()
            samples = scaler.inverse_transform(samples)
            all_X.append(samples)
            all_y.extend([cls] * n_per_class)
    return np.vstack(all_X), np.array(all_y)


# ── Main Experiment ────────────────────────────────────────────────────────────

def run_layer1_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("LAYER 1: Synthetic Data Generation Experiment")
    print(f"Device: {device}")
    print(f"{'='*60}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\n[1/6] Loading NSL-KDD dataset...")
    X, y, class_names = load_nsl_kdd()
    num_classes = len(class_names)
    input_dim   = X.shape[1]
    print(f"  Loaded {len(X)} samples | {num_classes} classes | {input_dim} features")
    print(f"  Classes: {class_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Baseline ──
    print("\n[2/6] Baseline — training on original imbalanced data...")
    with Timer("Baseline RF training"):
        clf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_base.fit(X_train, y_train)
    res_baseline = evaluate_classifier(y_test, clf_base.predict(X_test),
                                       label="baseline_imbalanced")

    # ── GAN augmentation ──
    n_gen = L1["synthetic_samples"]
    print(f"\n[3/6] Training GAN (generating {n_gen} samples/class)...")
    with Timer("GAN training"):
        G, gan_scaler = train_gan(X_train, y_train, num_classes, device)
    X_gan, y_gan = generate_gan_samples(G, gan_scaler, num_classes, n_gen, device)
    X_tr_gan = np.vstack([X_train, X_gan])
    y_tr_gan = np.concatenate([y_train, y_gan])
    print(f"  Augmented set: {len(X_tr_gan)} samples")

    with Timer("GAN-augmented RF"):
        clf_gan = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_gan.fit(X_tr_gan, y_tr_gan)
    res_gan = evaluate_classifier(y_test, clf_gan.predict(X_test),
                                  label="gan_augmented")

    # ── DDPM augmentation ──
    print(f"\n[4/6] Training DDPM (input_dim={input_dim}, generating {n_gen}/class)...")
    with Timer("DDPM training"):
        ddpm_model, sampler, ddpm_scaler, losses = train_ddpm(
            X_train, y_train, num_classes, device
        )

    print(f"\n  Generating DDPM samples...")
    X_ddpm, y_ddpm = generate_ddpm_samples(
        ddpm_model, sampler, ddpm_scaler,
        num_classes, n_gen, input_dim, device   # ← pass input_dim explicitly
    )
    X_tr_ddpm = np.vstack([X_train, X_ddpm])
    y_tr_ddpm = np.concatenate([y_train, y_ddpm])
    print(f"  Augmented set: {len(X_tr_ddpm)} samples")

    with Timer("DDPM-augmented RF"):
        clf_ddpm = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_ddpm.fit(X_tr_ddpm, y_tr_ddpm)
    res_ddpm = evaluate_classifier(y_test, clf_ddpm.predict(X_test),
                                   label="ddpm_augmented")

    # ── Save ──
    all_results = [res_baseline, res_gan, res_ddpm]
    print("\n[5/6] Saving results...")
    save_results_csv(all_results, "layer1_augmentation_comparison.csv")
    plot_comparison_bar(all_results, metric="f1_score",
                        title="Layer 1: F1 Score — Imbalanced vs GAN vs DDPM")
    plot_comparison_bar(all_results, metric="accuracy",
                        title="Layer 1: Accuracy — Imbalanced vs GAN vs DDPM")

    torch.save({
        "model_state": ddpm_model.state_dict(),
        "input_dim":   input_dim,
        "num_classes": num_classes,
    }, MODELS_DIR / "ddpm_layer1.pt")
    print(f"  Model saved → {MODELS_DIR / 'ddpm_layer1.pt'}")

    print("\n[6/6] Logging to MLflow...")
    with mlflow.start_run(run_name="layer1_data_augmentation"):
        mlflow.log_params({
            "ddpm_epochs":   L1["ddpm_epochs"],
            "ddpm_timesteps": L1["ddpm_timesteps"],
            "gan_epochs":    L1["gan_epochs"],
            "input_dim":     input_dim,
            "num_classes":   num_classes,
            "synthetic_samples_per_class": n_gen,
        })
        mlflow.log_metrics({
            "baseline_f1":  res_baseline["f1_score"],
            "gan_f1":       res_gan["f1_score"],
            "ddpm_f1":      res_ddpm["f1_score"],
            "baseline_acc": res_baseline["accuracy"],
            "gan_acc":      res_gan["accuracy"],
            "ddpm_acc":     res_ddpm["accuracy"],
        })

    imp = (res_ddpm["f1_score"] - res_baseline["f1_score"]) * 100
    print(f"\n✅ Layer 1 complete.")
    print(f"   Baseline F1 : {res_baseline['f1_score']:.4f}")
    print(f"   GAN F1      : {res_gan['f1_score']:.4f}")
    print(f"   DDPM F1     : {res_ddpm['f1_score']:.4f}")
    print(f"   Improvement : +{imp:.2f}%")

    return {"ddpm_model": ddpm_model, "sampler": sampler,
            "scaler": ddpm_scaler, "results": all_results}


if __name__ == "__main__":
    run_layer1_experiment()
