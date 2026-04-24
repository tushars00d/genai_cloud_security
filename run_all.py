"""
run_all.py — Run the complete 5-layer pipeline end-to-end.

Usage:
    python run_all.py                # Full pipeline
    python run_all.py --layer 1      # Single layer
    python run_all.py --quick        # Quick test (small datasets)
    python run_all.py --skip-l1      # Skip Layer 1 (slow DDPM training)
"""

import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import RESULTS_DIR, MLFLOW_TRACKING_URI, EXPERIMENT_NAME


def print_header(title):
    print(f"\n{'#'*60}")
    print(f"#  {title}")
    print(f"{'#'*60}")


def run_setup():
    print_header("SETUP: Downloading / generating datasets")
    from utils.download_datasets import (
        download_nsl_kdd, generate_cicids_sample, generate_synthetic_incidents
    )
    download_nsl_kdd()
    generate_cicids_sample()
    generate_synthetic_incidents()


def run_layer1():
    print_header("LAYER 1: Synthetic Data Generation (DDPM + GAN)")
    from layer1_data.train_ddpm import run_layer1_experiment
    return run_layer1_experiment()


def run_layer2():
    print_header("LAYER 2: Intrusion Detection + Adversarial Purification")
    from layer2_detection.train_ids import run_layer2_experiment
    return run_layer2_experiment()


def run_layer2_research():
    print_header("LAYER 2: Research-Grade IDS Improvements")
    from layer2_detection.research_grade_ids import run_research_grade_layer2
    return run_research_grade_layer2()


def run_layer3():
    print_header("LAYER 3: LLM + RAG Cognitive Analysis")
    from layer3_cognitive.run_rag_pipeline import run_layer3_experiment
    return run_layer3_experiment()


def run_layer4():
    print_header("LAYER 4: Autonomous Response Simulation")
    from layer4_response.autonomous_response import run_layer4_experiment
    return run_layer4_experiment()


def run_layer5():
    print_header("LAYER 5: Governance, Explainability, Dashboard")
    from layer5_governance.governance import run_layer5_experiment
    return run_layer5_experiment()


def print_final_summary(results: dict, total_time: float):
    print(f"\n{'='*60}")
    print("  DISSERTATION RESULTS SUMMARY")
    print(f"{'='*60}")

    if "l1" in results and results["l1"]:
        r = results["l1"].get("results", [])
        if r:
            baseline = next((x for x in r if "baseline" in x.get("label","")), None)
            ddpm     = next((x for x in r if "ddpm"     in x.get("label","")), None)
            if baseline and ddpm:
                imp = (ddpm["f1_score"] - baseline["f1_score"]) * 100
                print(f"\n  Layer 1 — Synthetic Data Generation:")
                print(f"    Baseline F1:  {baseline['f1_score']:.4f}")
                print(f"    DDPM F1:      {ddpm['f1_score']:.4f}")
                print(f"    Improvement:  +{imp:.1f}%")

    if "l2" in results and results["l2"]:
        r = results["l2"].get("results", [])
        ids = next((x for x in r if x.get("label") == "attention_ids"), None)
        adv = next((x for x in r if x.get("label") == "under_adversarial_attack"), None)
        pure = next((x for x in r if x.get("label") == "after_purification"), None)
        if ids and adv and pure:
            print(f"\n  Layer 2 — Detection + Purification:")
            print(f"    Attention-IDS F1:     {ids['f1_score']:.4f}")
            print(f"    Acc under attack:     {adv['accuracy']:.4f}")
            print(f"    Acc after purif:      {pure['accuracy']:.4f}")

    if "l4" in results and results["l4"]:
        mttr = results["l4"].get("mttr_reduction_pct", 0)
        print(f"\n  Layer 4 — Autonomous Response:")
        print(f"    MTTR reduction vs manual: {mttr:.1f}%")

    print(f"\n  Total runtime: {total_time/60:.1f} minutes")
    print(f"\n  All results saved to: {RESULTS_DIR}")
    print(f"  MLflow UI: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print(f"\n  📊 Main figure: {RESULTS_DIR / 'DISSERTATION_RESULTS_DASHBOARD.png'}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="GenAI Cloud Security — Full Pipeline")
    parser.add_argument("--layer",   type=int, help="Run only this layer (1-5)")
    parser.add_argument("--quick",   action="store_true", help="Quick test mode")
    parser.add_argument("--skip-l1", action="store_true", help="Skip slow DDPM training")
    parser.add_argument("--skip-l3", action="store_true", help="Skip LLM layer")
    parser.add_argument("--research-l2", action="store_true",
                        help="Run improved Layer 2 experiment after the baseline Layer 2 pipeline")
    args = parser.parse_args()

    if args.quick:
        import config
        config.L1["ddpm_epochs"]  = 5
        config.L1["gan_epochs"]   = 10
        config.L1["synthetic_samples"] = 500
        config.L4["num_test_incidents"] = 10
        print("Quick mode: reduced epochs and samples for fast testing.")

    total_start = time.time()
    results = {}

    print(f"\n{'#'*60}")
    print("#  GenAI as a Force Multiplier in Cloud Security")
    print("#  M.Tech Dissertation — Tushar Sood, JIIT 2026")
    print(f"{'#'*60}")

    if args.layer:
        # Single layer
        layer_map = {
            1: run_layer1, 2: run_layer2, 3: run_layer3,
            4: run_layer4, 5: run_layer5
        }
        if args.layer not in layer_map:
            print(f"Invalid layer. Choose 1-5.")
            sys.exit(1)
        run_setup()
        results[f"l{args.layer}"] = layer_map[args.layer]()
        if args.layer == 2 and args.research_l2:
            results["l2_research"] = run_layer2_research()
    else:
        # Full pipeline
        run_setup()
        if not args.skip_l1:
            results["l1"] = run_layer1()
        results["l2"] = run_layer2()
        if args.research_l2:
            results["l2_research"] = run_layer2_research()
        if not args.skip_l3:
            results["l3"] = run_layer3()
        results["l4"] = run_layer4()
        results["l5"] = run_layer5()

    print_final_summary(results, time.time() - total_start)


if __name__ == "__main__":
    main()
