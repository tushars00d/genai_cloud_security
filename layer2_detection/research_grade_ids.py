"""
Research-grade Layer 2 IDS experiment.

This module preserves the original Layer 2 pipeline and adds an improved,
explicitly reproducible experiment focused on macro F1 and robustness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import DATA_DIR, RESULTS_DIR, MODELS_DIR, MLFLOW_TRACKING_URI, EXPERIMENT_NAME, L2_RESEARCH
from evaluation.research_metrics import save_confusion_matrix, save_multiclass_roc, save_per_class_report
from models.tabular_ids import FocalLoss, TabularTransformerIDS
from training.imbalance import balanced_class_weights, random_oversample
from utils.metrics import plot_comparison_bar, save_results_csv


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cicids_frame():
    path = DATA_DIR / "cicids_sample.csv"
    if not path.exists():
        raise FileNotFoundError("CICIDS sample not found. Run python utils/download_datasets.py")
    df = pd.read_csv(path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    feature_names = [c for c in df.columns if c != "Label"]
    le = LabelEncoder()
    y = le.fit_transform(df["Label"])
    X = df[feature_names].values.astype(np.float32)
    return X, y, feature_names, le.classes_


def select_features(X_train, y_train, X_val, X_test, feature_names, k: int):
    if k <= 0 or k >= X_train.shape[1]:
        return X_train, X_val, X_test, feature_names, {}
    scores = mutual_info_classif(X_train, y_train, random_state=L2_RESEARCH["random_state"])
    top_idx = np.argsort(scores)[::-1][:k]
    selected_names = [feature_names[i] for i in top_idx]
    score_map = {feature_names[i]: float(scores[i]) for i in top_idx}
    return X_train[:, top_idx], X_val[:, top_idx], X_test[:, top_idx], selected_names, score_map


def metrics_dict(y_true, y_pred, label):
    return {
        "label": label,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }


def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def fgsm_attack(model, X, y, eps, device):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, requires_grad=True, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    loss = nn.functional.cross_entropy(model(X_t), y_t)
    loss.backward()
    return (X_t + eps * X_t.grad.sign()).detach().cpu().numpy().astype(np.float32)


def predict_model(model, X, device, batch_size=512):
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=-1)
            preds.append(prob.argmax(dim=-1).cpu().numpy())
            probs.append(prob.cpu().numpy())
    return np.concatenate(preds), np.vstack(probs)


def train_transformer(X_train, y_train, X_val, y_val, num_classes, device):
    cfg = L2_RESEARCH
    weights = torch.tensor(balanced_class_weights(y_train, num_classes), dtype=torch.float32, device=device)
    model = TabularTransformerIDS(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
    loss_fn = FocalLoss(gamma=cfg["focal_gamma"], weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    train_loader = make_loader(X_train, y_train, cfg["batch_size"], shuffle=True)

    best_state, best_macro, stale = None, -1.0, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if cfg["adversarial_training"]:
                xb_adv = xb.detach().clone().requires_grad_(True)
                adv_loss = nn.functional.cross_entropy(model(xb_adv), yb)
                grad = torch.autograd.grad(adv_loss, xb_adv, retain_graph=False)[0]
                xb_adv = (xb_adv + cfg["adversarial_eps"] * grad.sign()).detach()
                loss = 0.5 * loss + 0.5 * loss_fn(model(xb_adv), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        val_pred, _ = predict_model(model, X_val, device)
        val_macro = f1_score(y_val, val_pred, average="macro", zero_division=0)
        if val_macro > best_macro:
            best_macro = val_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if (epoch + 1) % 10 == 0:
            print(f"  Improved IDS epoch {epoch+1}/{cfg['epochs']} | val_macro_f1={val_macro:.4f}")
        if stale >= cfg["patience"]:
            print(f"  Early stopping at epoch {epoch+1}; best val macro F1={best_macro:.4f}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_macro


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_dae(X_train, device):
    cfg = L2_RESEARCH
    model = DenoisingAutoencoder(X_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loader = make_loader(X_train, np.zeros(len(X_train), dtype=np.int64), cfg["batch_size"], shuffle=True)
    for epoch in range(cfg["denoiser_epochs"]):
        model.train()
        total = 0.0
        for xb, _ in loader:
            xb = xb.to(device)
            noise = torch.randn_like(xb) * np.random.uniform(0.02, 0.15)
            recon = model(xb + noise)
            loss = nn.functional.mse_loss(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  DAE epoch {epoch+1}/{cfg['denoiser_epochs']} | loss={total / len(loader):.4f}")
    return model


def purify(dae, X, device):
    dae.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.tensor(X[i:i + 512], dtype=torch.float32, device=device)
            out.append(dae(xb).cpu().numpy())
    return np.vstack(out).astype(np.float32)


def run_research_grade_layer2():
    cfg = L2_RESEARCH
    set_seed(cfg["random_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 60)
    print("LAYER 2 RESEARCH-GRADE IDS EXPERIMENT")
    print(f"Device: {device}")
    print("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y, feature_names, class_names = load_cicids_frame()
    num_classes = len(class_names)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=cfg["test_split"], random_state=cfg["random_state"], stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=cfg["val_split"],
        random_state=cfg["random_state"], stratify=y_train_full
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    X_train, X_val, X_test, selected_features, mi_scores = select_features(
        X_train, y_train, X_val, X_test, feature_names, cfg["feature_selection_k"]
    )
    if cfg["oversample_minority"]:
        X_train_bal, y_train_bal = random_oversample(
            X_train, y_train, cfg["max_oversample_per_class"], cfg["random_state"]
        )
    else:
        X_train_bal, y_train_bal = X_train, y_train

    print(f"  Train/val/test: {len(X_train_bal)}/{len(X_val)}/{len(X_test)}")
    print(f"  Selected features: {len(selected_features)}")

    baselines = {
        "rf_balanced_selected": RandomForestClassifier(
            n_estimators=250, class_weight="balanced_subsample",
            random_state=cfg["random_state"], n_jobs=-1
        ),
        "hist_gradient_boosting_selected": HistGradientBoostingClassifier(
            max_iter=160, learning_rate=0.08, random_state=cfg["random_state"]
        ),
    }
    results = []
    fitted = []
    for label, clf in baselines.items():
        print(f"\n[Baseline] Training {label}...")
        clf.fit(X_train_bal, y_train_bal)
        pred = clf.predict(X_test)
        results.append(metrics_dict(y_test, pred, label))
        fitted.append((label, clf))

    if len(fitted) >= 2:
        vote = VotingClassifier(estimators=fitted, voting="soft")
        print("\n[Baseline] Training soft-voting ensemble...")
        vote.fit(X_train_bal, y_train_bal)
        pred = vote.predict(X_test)
        results.append(metrics_dict(y_test, pred, "soft_voting_tree_ensemble"))

    print("\n[Improved] Training TabularTransformerIDS...")
    model, best_val_macro = train_transformer(X_train_bal, y_train_bal, X_val, y_val, num_classes, device)
    pred_clean, prob_clean = predict_model(model, X_test, device)
    clean_res = metrics_dict(y_test, pred_clean, "tabular_transformer_focal_advtrain")
    clean_res["best_val_macro_f1"] = round(float(best_val_macro), 4)
    results.append(clean_res)

    print("\n[Robustness] Evaluating FGSM and DAE purification...")
    dae = train_dae(X_train_bal, device)
    robustness_rows = []
    for eps in cfg["attack_eval_eps"]:
        X_adv = fgsm_attack(model, X_test, y_test, eps, device)
        pred_adv, _ = predict_model(model, X_adv, device)
        adv_res = metrics_dict(y_test, pred_adv, f"fgsm_eps_{eps}")
        X_pure = purify(dae, X_adv, device)
        pred_pure, _ = predict_model(model, X_pure, device)
        pure_res = metrics_dict(y_test, pred_pure, f"purified_eps_{eps}")
        robustness_rows.extend([adv_res, pure_res])

    results_df = save_results_csv(results, "layer2_research_detection_comparison.csv")
    robustness_df = save_results_csv(robustness_rows, "layer2_research_robustness.csv")
    plot_comparison_bar(results, metric="macro_f1", title="Layer 2 Research: Macro F1 Comparison")

    save_per_class_report(
        y_test, pred_clean, class_names,
        RESULTS_DIR / "layer2_research_per_class_report.csv"
    )
    save_confusion_matrix(
        y_test, pred_clean, class_names,
        RESULTS_DIR / "layer2_research_confusion_matrix.png"
    )
    save_multiclass_roc(
        y_test, prob_clean, class_names,
        RESULTS_DIR / "layer2_research_roc_curves.png",
        RESULTS_DIR / "layer2_research_roc_auc.json",
    )
    (RESULTS_DIR / "layer2_research_feature_selection.json").write_text(
        json.dumps({"selected_features": selected_features, "mutual_information": mi_scores}, indent=2)
    )
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": X_train.shape[1],
        "num_classes": num_classes,
        "selected_features": selected_features,
        "class_names": [str(c) for c in class_names],
    }, MODELS_DIR / "tabular_transformer_ids.pt")

    with mlflow.start_run(run_name="layer2_research_grade"):
        mlflow.log_params({k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool))})
        mlflow.log_metrics({
            "research_macro_f1": clean_res["macro_f1"],
            "research_weighted_f1": clean_res["f1_score"],
            "research_balanced_accuracy": clean_res["balanced_accuracy"],
        })

    print("\nResearch-grade Layer 2 complete.")
    print(results_df.to_string(index=False))
    print("\nRobustness:")
    print(robustness_df.to_string(index=False))
    return {"results": results, "robustness": robustness_rows, "model": model}


if __name__ == "__main__":
    run_research_grade_layer2()
