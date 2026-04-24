# Research-Grade Improvement Branch

Branch: `feature/research-grade-improvements`

This branch keeps the original dissertation pipeline reproducible and adds a separate research-grade path for stronger evaluation.

## What Changed

### Layer 5 Governance

- Fixed multiclass SHAP handling for tree models.
- Added SHAP summary plot, global feature-importance plot, and local explanation plot.
- Added JSON explanation logs for the generated explanations.
- Added LIME fallback if SHAP fails.
- Feature names are derived from the CICIDS-format dataframe when available.

Expected artifacts:

- `results/layer5_shap_importance_random_forest_ids.png`
- `results/layer5_shap_summary_random_forest_ids.png`
- `results/layer5_shap_local_random_forest_ids.png`
- `results/layer5_shap_explanation_random_forest_ids.json`

### Layer 2 Research-Grade Experiment

New runner:

```bash
python layer2_detection/research_grade_ids.py
```

or through the full pipeline:

```bash
python run_all.py --quick --skip-l1 --research-l2
```

The improved Layer 2 path includes:

- Stratified train/validation/test split.
- Standard scaling fit only on training data.
- Mutual-information feature selection.
- Capped random oversampling for minority classes.
- Class-balanced focal loss.
- Small FT-Transformer-style tabular IDS.
- Early stopping on validation macro F1.
- FGSM adversarial training.
- Denoising autoencoder purification.
- Per-class report, confusion matrix, and multiclass ROC curves.

Expected artifacts:

- `results/layer2_research_detection_comparison.csv`
- `results/layer2_research_robustness.csv`
- `results/layer2_research_per_class_report.csv`
- `results/layer2_research_confusion_matrix.png`
- `results/layer2_research_roc_curves.png`
- `results/layer2_research_roc_auc.json`
- `results/layer2_research_feature_selection.json`
- `results/models/tabular_transformer_ids.pt`

## Why This Is More Defensible

The original experiment was dominated by weighted F1 and raw accuracy, which can hide poor minority-class detection. The new primary target is macro F1, supported by balanced accuracy and per-class recall. This aligns better with IDS research because rare attacks are often the operationally important classes.

The improved neural IDS is not claimed to be state of the art by default. It must beat the baselines in generated results before any superiority claim is made. If it does not, the correct claim is that the architecture is feasible but the detector requires stronger feature engineering, real CICIDS validation, or additional tuning.

## Limitations

- The current CICIDS file is a Colab-scale synthetic stand-in, not the full UNB CICIDS 2017 dataset.
- Oversampling is capped random oversampling, not a proof that generated minority samples reflect real attack diversity.
- Denoising autoencoder purification is empirical and does not provide formal robustness guarantees.
- Layer 4 still measures simulated SOAR orchestration, not live containment in production.
- LLM-based Layer 3 scores depend on the incident set and selected provider/model.
