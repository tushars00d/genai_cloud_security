"""
layer4_response/autonomous_response.py
Layer 4: Autonomous Agentic Response Simulation.

Simulates LLM-SOAR integration using:
- LangChain agents with custom security tools
- Probabilistic confidence gating for autonomous vs human approval
- MTTR measurement vs manual baseline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
import mlflow

from config import DATA_DIR, RESULTS_DIR, MLFLOW_TRACKING_URI, EXPERIMENT_NAME, L4, LLM_PROVIDER, LLM_MODEL, GROQ_API_KEY
from utils.metrics import save_results_csv, plot_comparison_bar


# ── Simulated SOAR Playbook Actions ────────────────────────────────────────────
# These represent real SOAR actions. In production, these call real APIs.

class SOARPlaybook:
    """Simulated SOAR platform — mock actions that return success/failure."""

    def __init__(self, audit_log_path):
        self.audit_log_path = audit_log_path
        self.action_log = []

    def isolate_host(self, resource_id: str, reason: str) -> dict:
        time.sleep(0.05)  # Simulate API call
        result = {"action": "isolate_host", "resource": resource_id,
                  "status": "success", "timestamp": datetime.utcnow().isoformat(),
                  "reason": reason}
        self._log(result)
        return result

    def block_ip(self, ip_address: str, reason: str) -> dict:
        time.sleep(0.03)
        result = {"action": "block_ip", "resource": ip_address,
                  "status": "success", "timestamp": datetime.utcnow().isoformat(),
                  "reason": reason}
        self._log(result)
        return result

    def revoke_iam_credentials(self, user_id: str, reason: str) -> dict:
        time.sleep(0.04)
        result = {"action": "revoke_iam_credentials", "resource": user_id,
                  "status": "success", "timestamp": datetime.utcnow().isoformat(),
                  "reason": reason}
        self._log(result)
        return result

    def enable_bucket_block_public(self, bucket_id: str, reason: str) -> dict:
        time.sleep(0.03)
        result = {"action": "enable_bucket_block_public", "resource": bucket_id,
                  "status": "success", "timestamp": datetime.utcnow().isoformat(),
                  "reason": reason}
        self._log(result)
        return result

    def create_security_group_rule(self, sg_id: str, action_type: str) -> dict:
        time.sleep(0.04)
        result = {"action": "update_security_group", "resource": sg_id,
                  "status": "success", "timestamp": datetime.utcnow().isoformat(),
                  "action_type": action_type}
        self._log(result)
        return result

    def notify_soc_team(self, incident_id: str, message: str) -> dict:
        result = {"action": "notify_soc", "incident_id": incident_id,
                  "status": "sent", "timestamp": datetime.utcnow().isoformat()}
        self._log(result)
        return result

    def escalate_to_human(self, incident_id: str, reason: str) -> dict:
        result = {"action": "human_escalation", "incident_id": incident_id,
                  "reason": reason, "status": "escalated",
                  "timestamp": datetime.utcnow().isoformat()}
        self._log(result)
        return result

    def _log(self, action: dict):
        self.action_log.append(action)
        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(action) + "\n")


# ── Playbook Selection Engine ──────────────────────────────────────────────────

PLAYBOOK_MAP = {
    "T1041": [("isolate_host",            "Stop exfiltration channel"),
              ("block_ip",                "Block destination IP"),
              ("notify_soc_team",         "Alert SOC of exfiltration event")],
    "T1110": [("block_ip",                "Block brute force source IP"),
              ("notify_soc_team",         "Alert on credential attack")],
    "T1078": [("revoke_iam_credentials",  "Revoke compromised credentials"),
              ("notify_soc_team",         "Alert on IAM compromise")],
    "T1530": [("enable_bucket_block_public", "Secure misconfigured S3 bucket"),
              ("notify_soc_team",            "Alert on data exposure")],
    "T1496": [("isolate_host",            "Stop cryptomining workload"),
              ("notify_soc_team",         "Alert on resource hijacking")],
    "T1562": [("notify_soc_team",         "Alert on defense evasion"),
              ("escalate_to_human",       "Requires manual audit review")],
    "T1133": [("create_security_group_rule", "Revoke public RDP access"),
              ("notify_soc_team",            "Alert on external service exposure")],
    "T1552": [("revoke_iam_credentials",  "Rotate exposed credentials"),
              ("notify_soc_team",         "Alert on credential exposure")],
    "T1190": [("block_ip",                "Block attack source"),
              ("notify_soc_team",         "Alert on web application attack")],
    "T1071": [("block_ip",                "Block C2 communication"),
              ("notify_soc_team",         "Alert on DNS tunneling")],
}

SEVERITY_PRIORITY = {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 3, "LOW": 4}


class AutonomousResponseAgent:
    """
    Autonomous response agent using LLM-guided playbook selection
    + probabilistic confidence gating.
    """

    def __init__(self, soar: SOARPlaybook, autonomy_level: int = 2):
        self.soar = soar
        self.autonomy_level = autonomy_level  # 0-4 per Srinivas et al. (2025)
        self.confidence_threshold_auto  = L4["confidence_threshold_auto"]
        self.confidence_threshold_human = L4["confidence_threshold_human"]

    def _select_playbook(self, incident: dict) -> list[tuple]:
        """Select appropriate playbook steps based on MITRE technique."""
        technique = incident.get("pred_mitre", "T1059")
        return PLAYBOOK_MAP.get(technique, [
            ("notify_soc_team", "Unknown technique — manual review required"),
            ("escalate_to_human", "No automated playbook available"),
        ])

    def _compute_confidence(self, incident: dict) -> float:
        """
        Compute response confidence score.
        Combines: LLM confidence + severity weight + technique match quality.
        """
        llm_conf  = float(incident.get("confidence", 0.5))
        sev       = incident.get("severity", "MEDIUM")
        sev_score = {1: 0.9, 2: 0.8, 3: 0.65, 4: 0.5}.get(SEVERITY_PRIORITY.get(sev, 3), 0.5)
        tech_ok   = 1.0 if incident.get("pred_mitre", "").startswith("T") else 0.4
        return round((llm_conf * 0.5 + sev_score * 0.3 + tech_ok * 0.2), 3)

    def _pgm_validate(self, playbook_steps: list, confidence: float) -> dict:
        """
        Probabilistic Graphical Model validation.
        Estimates risk of each action given confidence score.
        (Simplified: maps action type to risk weight × confidence)
        """
        high_risk_actions = {"isolate_host", "revoke_iam_credentials"}
        all_risks = []
        for action_name, _ in playbook_steps:
            risk = (1 - confidence) * (1.5 if action_name in high_risk_actions else 0.8)
            all_risks.append(risk)
        return {
            "max_risk": max(all_risks),
            "requires_human_approval": max(all_risks) > 0.5,
            "estimated_false_positive_cost": round(max(all_risks) * 0.3, 3),
        }

    def _get_llm_playbook_decision(self, incident: dict, playbook: list) -> str:
        """Ask LLM to validate and potentially modify the playbook."""
        if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
            # Offline mode: approve standard playbook
            return "APPROVE"

        prompt = f"""Security incident details:
Incident: {incident.get('description', '')[:200]}
MITRE Technique: {incident.get('pred_mitre', 'Unknown')}
Severity: {incident.get('severity', 'UNKNOWN')}
Confidence: {incident.get('confidence', 0.5):.2f}

Proposed automated response actions:
{chr(10).join([f"  {i+1}. {step[0]}: {step[1]}" for i, step in enumerate(playbook)])}

Should this automated response be APPROVED or ESCALATED to human review?
Respond with exactly one word: APPROVE or ESCALATE"""

        from layer3_cognitive.run_rag_pipeline import get_llm_response
        response = get_llm_response(prompt).strip().upper()
        return "APPROVE" if "APPROVE" in response else "ESCALATE"

    def respond(self, incident: dict) -> dict:
        """Execute full autonomous response for an incident."""
        start_time = time.time()
        incident_id = str(incident.get("incident_id", "unknown"))

        # 1. Select playbook
        playbook = self._select_playbook(incident)

        # 2. Compute confidence
        confidence = self._compute_confidence(incident)
        incident["confidence"] = confidence

        # 3. PGM validation
        pgm = self._pgm_validate(playbook, confidence)

        # 4. Decide: autonomous vs human escalation
        if self.autonomy_level == 0 or confidence < self.confidence_threshold_human:
            decision = "HUMAN_ESCALATION"
        elif self.autonomy_level >= 3 or (
            confidence >= self.confidence_threshold_auto and not pgm["requires_human_approval"]
        ):
            decision = "AUTONOMOUS"
        else:
            # Get LLM validation
            llm_decision = self._get_llm_playbook_decision(incident, playbook)
            decision = "AUTONOMOUS" if llm_decision == "APPROVE" else "HUMAN_ESCALATION"

        # 5. Execute actions
        actions_taken = []
        resource = incident.get("affected_resource", "unknown-resource")

        if decision == "AUTONOMOUS":
            for action_name, reason in playbook:
                action_fn = getattr(self.soar, action_name, None)
                if action_fn:
                    result = action_fn(resource, reason) if action_name != "create_security_group_rule" \
                             else action_fn(resource, "revoke")
                    actions_taken.append(result)
        else:
            # Human escalation
            result = self.soar.escalate_to_human(incident_id,
                f"Confidence {confidence:.2f} below threshold or high-risk action requires approval. "
                f"Technique: {incident.get('pred_mitre', 'unknown')}")
            actions_taken.append(result)
            # Still send notification
            self.soar.notify_soc_team(incident_id, f"Incident requires human review: {incident.get('severity', 'UNKNOWN')} severity")

        elapsed = time.time() - start_time
        mttr_minutes = elapsed / 60

        return {
            "incident_id":        incident_id,
            "decision":           decision,
            "confidence":         confidence,
            "pgm_max_risk":       pgm["max_risk"],
            "playbook_steps":     len(playbook),
            "actions_executed":   len(actions_taken),
            "mttr_seconds":       round(elapsed, 3),
            "mttr_minutes":       round(mttr_minutes, 4),
            "severity":           incident.get("severity", "UNKNOWN"),
            "mitre_technique":    incident.get("pred_mitre", "unknown"),
            "autonomy_level":     self.autonomy_level,
            "success":            all(a.get("status") in ("success","sent","escalated")
                                      for a in actions_taken),
        }


# ── Main Experiment ────────────────────────────────────────────────────────────

def run_layer4_experiment():
    print(f"\n{'='*60}")
    print("LAYER 4: Autonomous Response Simulation")
    print(f"Autonomy level: {L4.get('default_autonomy_level', 2)}")
    print(f"{'='*60}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load Layer 3 results (or generate test incidents)
    layer3_results_path = RESULTS_DIR / "layer3_rag_analysis_results.csv"
    if layer3_results_path.exists():
        incidents = pd.read_csv(layer3_results_path).to_dict("records")
        print(f"\n[1/4] Loaded {len(incidents)} analysed incidents from Layer 3.")
    else:
        # Generate test incidents directly
        from utils.download_datasets import generate_synthetic_incidents
        generate_synthetic_incidents()
        df = pd.read_csv(DATA_DIR / "synthetic_incidents.csv")
        incidents = []
        for _, row in df.iterrows():
            incidents.append({
                "incident_id": row["id"], "description": row["description"],
                "pred_mitre": row.get("mitre_technique", "T1078"),
                "severity": random.choice(["CRITICAL","HIGH","MEDIUM"]),
                "confidence": round(random.uniform(0.55, 0.95), 2),
                "affected_resource": row.get("affected_resource", "EC2:i-unknown"),
            })
        print(f"\n[1/4] Generated {len(incidents)} test incidents.")

    incidents = incidents[:L4["num_test_incidents"]]

    audit_log = RESULTS_DIR / "layer4_audit_log.jsonl"
    soar = SOARPlaybook(audit_log_path=str(audit_log))

    # ── Exp: Compare autonomy levels ──
    print("\n[2/4] Running response simulation across autonomy levels...")
    all_level_results = []

    for level in [0, 1, 2, 3, 4]:
        agent = AutonomousResponseAgent(soar, autonomy_level=level)
        level_results = []
        for inc in incidents:
            result = agent.respond(inc)
            level_results.append(result)

        auto_rate = np.mean([r["decision"] == "AUTONOMOUS" for r in level_results])
        avg_mttr  = np.mean([r["mttr_seconds"] for r in level_results])
        success   = np.mean([r["success"] for r in level_results])

        level_summary = {
            "label":          f"Level {level} ({['Manual','Human-Approved','Conditional','Supervised','Fully-Auto'][level]})",
            "autonomy_level": level,
            "auto_rate":      round(auto_rate, 3),
            "avg_mttr_s":     round(avg_mttr, 3),
            "avg_mttr_min":   round(avg_mttr / 60, 4),
            "success_rate":   round(success, 3),
        }
        all_level_results.append(level_summary)
        print(f"  Level {level}: auto_rate={auto_rate:.1%} | MTTR={avg_mttr:.2f}s | success={success:.1%}")

    # ── MTTR vs manual baseline ──
    manual_mttr_s = L4["simulated_manual_mttr_mins"] * 60
    best_auto = min(all_level_results, key=lambda x: x["avg_mttr_s"])
    mttr_reduction_pct = (manual_mttr_s - best_auto["avg_mttr_s"]) / manual_mttr_s * 100

    print(f"\n  Manual baseline MTTR:  {L4['simulated_manual_mttr_mins']} minutes")
    print(f"  Best automated MTTR:   {best_auto['avg_mttr_min']:.3f} minutes")
    print(f"  MTTR reduction:        {mttr_reduction_pct:.1f}%")

    # ── Save results ──
    print("\n[3/4] Saving results...")
    save_results_csv(all_level_results, "layer4_autonomy_comparison.csv")

    # MTTR comparison chart
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels   = [f"L{r['autonomy_level']}" for r in all_level_results]
    mttr_min = [r["avg_mttr_min"] for r in all_level_results]
    auto_r   = [r["auto_rate"] * 100 for r in all_level_results]

    ax = axes[0]
    bars = ax.bar(labels, mttr_min, color="#2563EB")
    ax.axhline(y=L4["simulated_manual_mttr_mins"], color="red", linestyle="--",
               label=f"Manual baseline: {L4['simulated_manual_mttr_mins']} min")
    ax.set_title("Layer 4: MTTR by Autonomy Level", fontsize=13)
    ax.set_ylabel("MTTR (minutes)")
    ax.set_yscale("log")
    ax.legend()
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)

    axes[1].bar(labels, auto_r, color="#16A34A")
    axes[1].set_title("Layer 4: Autonomous Action Rate by Level", fontsize=13)
    axes[1].set_ylabel("Autonomous Action Rate (%)")
    axes[1].set_ylim(0, 110)
    for bar, val in zip(axes[1].patches, auto_r):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.0f}%", ha="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "layer4_response_analysis.png", dpi=150)
    plt.close()
    print(f"  Saved chart → {RESULTS_DIR / 'layer4_response_analysis.png'}")
    print(f"  Audit log → {audit_log}")

    # MLflow
    print("\n[4/4] Logging to MLflow...")
    with mlflow.start_run(run_name="layer4_autonomous_response"):
        mlflow.log_metrics({
            "mttr_reduction_pct": mttr_reduction_pct,
            "best_auto_mttr_min": best_auto["avg_mttr_min"],
            "manual_baseline_min": L4["simulated_manual_mttr_mins"],
            "level2_auto_rate": all_level_results[2]["auto_rate"],
            "level4_auto_rate": all_level_results[4]["auto_rate"],
        })

    print("\n✅ Layer 4 complete.")
    print(f"   MTTR reduction vs manual: {mttr_reduction_pct:.1f}%")
    return {"results": all_level_results, "mttr_reduction_pct": mttr_reduction_pct}


if __name__ == "__main__":
    run_layer4_experiment()
