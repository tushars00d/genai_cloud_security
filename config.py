"""
config.py — Central configuration for all layers.
Edit this file to switch datasets, LLM providers, and experiment settings.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "data"
RESULTS_DIR  = BASE_DIR / "results"
MODELS_DIR   = BASE_DIR / "results" / "models"

for d in [DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── LLM Provider ──────────────────────────────────────────────────────────────
# Options: "groq" | "ollama" | "openai" | "anthropic"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL    = os.getenv("LLM_MODEL") or {
    "groq":      "llama-3.1-8b-instant",  # Free-tier friendly Groq replacement for deprecated llama3-8b-8192
    "ollama":    "llama3.1",              # Fully local
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}.get(LLM_PROVIDER, "llama-3.1-8b-instant")

GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Layer 1: Data Generation ───────────────────────────────────────────────────
L1 = {
    "ddpm_epochs":       30,       # Reduce to 10 for quick test on Colab free
    "ddpm_timesteps":    200,
    "ddpm_batch_size":   256,
    "gan_epochs":        50,
    "gan_batch_size":    256,
    "noise_dim":         64,
    "synthetic_samples": 5000,     # Samples to generate per minority class
    "dataset":           "nsl_kdd",
}

# ── Layer 2: Detection ─────────────────────────────────────────────────────────
L2 = {
    "dataset":           "cicids2017",
    "test_split":        0.2,
    "random_state":      42,
    "adversarial_eps":   0.1,      # Perturbation strength for attack test
    "purification_steps": 50,      # DDPM denoising steps for purification
    "models": ["random_forest", "mlp", "attention_ids"],
}

# ── Layer 3: Cognitive Analysis ────────────────────────────────────────────────
L3 = {
    "chunk_size":        500,
    "chunk_overlap":     50,
    "top_k_retrieval":   5,
    "embedding_model":   "all-MiniLM-L6-v2",  # Free, runs locally
    "synthetic_incidents": 100,
    "mitre_techniques":  [
        "T1078", "T1190", "T1059", "T1053", "T1486",
        "T1071", "T1566", "T1110", "T1021", "T1083",
    ],
}

# ── Layer 4: Autonomous Response ───────────────────────────────────────────────
L4 = {
    "confidence_threshold_auto":  0.85,  # Above = autonomous execution
    "confidence_threshold_human": 0.60,  # Below = human escalation
    "max_agent_iterations":       5,
    "simulated_manual_mttr_mins": 480,   # 8 hours baseline (from literature)
    "num_test_incidents":         50,
}

# ── Layer 5: Governance ────────────────────────────────────────────────────────
L5 = {
    "shap_background_samples": 100,
    "audit_log_path": str(RESULTS_DIR / "audit_log.jsonl"),
    "autonomy_levels": {
        0: "fully_manual",
        1: "human_approved",
        2: "conditional_autonomous",
        3: "supervised_autonomous",
        4: "fully_autonomous",
    },
    "default_autonomy_level": 2,
}

# ── MLflow Tracking ────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = str(RESULTS_DIR / "mlruns")
EXPERIMENT_NAME     = "genai_cloud_security_dissertation"
