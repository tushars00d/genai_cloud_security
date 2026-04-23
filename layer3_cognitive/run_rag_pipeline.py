"""
layer3_cognitive/run_rag_pipeline.py
Layer 3: LLM + RAG Cognitive Analysis and Forensic Intelligence.

Builds a RAG pipeline that:
1. Ingests incident logs + MITRE ATT&CK knowledge base
2. On alert: retrieves relevant context
3. LLM generates root cause analysis + MITRE mapping + remediation
4. Evaluates factual accuracy against ground-truth labels
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow

from config import DATA_DIR, RESULTS_DIR, MLFLOW_TRACKING_URI, EXPERIMENT_NAME, L3, LLM_PROVIDER, LLM_MODEL, GROQ_API_KEY
from utils.metrics import save_results_csv, Timer

_LLM_DISABLED_REASON = None


def _coerce_float(value, default=0.0):
    """Best-effort conversion for LLM-emitted numeric fields."""
    if value is None:
        return float(default)
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                pass
    return float(default)


# ── MITRE ATT&CK Knowledge Base ────────────────────────────────────────────────

MITRE_KB = {
    "T1041": {"name": "Exfiltration Over C2 Channel", "tactic": "Exfiltration",
              "mitigation": "Monitor for large outbound transfers. Block unknown external IPs. Enable VPC flow logs."},
    "T1110": {"name": "Brute Force", "tactic": "Credential Access",
              "mitigation": "Enable MFA. Rate-limit SSH. Block IP after 5 failed attempts. Use AWS GuardDuty."},
    "T1078": {"name": "Valid Accounts", "tactic": "Initial Access",
              "mitigation": "Rotate IAM credentials. Enable CloudTrail. Review unusual IAM activity. Enforce least privilege."},
    "T1530": {"name": "Data from Cloud Storage Object", "tactic": "Collection",
              "mitigation": "Enable S3 Block Public Access. Set bucket policies. Enable S3 server access logging."},
    "T1496": {"name": "Resource Hijacking", "tactic": "Impact",
              "mitigation": "Monitor Lambda execution time. Set cost alerts. Use AWS Budgets. Scan code dependencies."},
    "T1562": {"name": "Impair Defenses", "tactic": "Defense Evasion",
              "mitigation": "Enable CloudTrail log file validation. Alert on logging changes. Use AWS Config rules."},
    "T1133": {"name": "External Remote Services", "tactic": "Initial Access",
              "mitigation": "Remove public RDP/SSH. Use Systems Manager Session Manager. Enforce security group review."},
    "T1552": {"name": "Unsecured Credentials", "tactic": "Credential Access",
              "mitigation": "Block IMDS v1. Enforce IMDSv2. Use IAM roles instead of credentials in containers."},
    "T1190": {"name": "Exploit Public-Facing Application", "tactic": "Initial Access",
              "mitigation": "Deploy WAF. Parameterise SQL queries. Use RDS IAM authentication. Enable enhanced monitoring."},
    "T1071": {"name": "Application Layer Protocol", "tactic": "Command and Control",
              "mitigation": "Monitor DNS anomalies. Use Route53 Resolver DNS Firewall. Block unknown domains."},
}


# ── LLM Client (supports Groq free tier + Ollama) ────────────────────────────

def get_llm_response(prompt: str, system_prompt: str = "") -> str:
    """
    Calls the configured LLM provider. Falls back to rule-based
    template if no API key is set (for demo / offline mode).
    """
    global _LLM_DISABLED_REASON
    if _LLM_DISABLED_REASON:
        return _rule_based_fallback(prompt)

    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        return _call_groq(prompt, system_prompt)
    elif LLM_PROVIDER == "ollama":
        return _call_ollama(prompt, system_prompt)
    else:
        return _rule_based_fallback(prompt)


def _call_groq(prompt, system_prompt):
    global _LLM_DISABLED_REASON
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(model=LLM_MODEL, messages=msgs, max_tokens=800)
        return resp.choices[0].message.content
    except Exception as e:
        _LLM_DISABLED_REASON = str(e)
        print(f"  Groq error: {e}. Falling back to rule-based for the rest of this run.")
        return _rule_based_fallback(prompt)


def _call_ollama(prompt, system_prompt):
    try:
        import requests
        payload = {"model": LLM_MODEL, "prompt": prompt,
                   "system": system_prompt, "stream": False}
        resp = requests.post("http://localhost:11434/api/generate",
                             json=payload, timeout=60)
        return resp.json().get("response", "")
    except Exception as e:
        print(f"  Ollama error: {e}. Falling back to rule-based.")
        return _rule_based_fallback(prompt)


def _rule_based_fallback(prompt: str) -> str:
    """
    Template-based analysis when no LLM is available.
    Simulates LLM output for demo purposes.
    """
    keywords = {
        "SSH": ("Brute force credential attack", "T1110"),
        "IAM": ("Privilege escalation via IAM role abuse", "T1078"),
        "S3":  ("Data exposure via misconfigured storage", "T1530"),
        "exfiltration": ("Data exfiltration over covert channel", "T1041"),
        "Lambda": ("Resource hijacking / cryptomining", "T1496"),
        "CloudTrail": ("Defense evasion via audit tampering", "T1562"),
        "RDP": ("External remote service exposure", "T1133"),
        "metadata": ("Credential theft from cloud metadata service", "T1552"),
        "SQL": ("Exploitation of public-facing database", "T1190"),
        "DNS": ("DNS tunneling for C2 communication", "T1071"),
    }
    for kw, (cause, technique) in keywords.items():
        if kw.lower() in prompt.lower():
            info = MITRE_KB.get(technique, {})
            return json.dumps({
                "root_cause": cause,
                "mitre_technique": technique,
                "mitre_name": info.get("name", "Unknown"),
                "tactic": info.get("tactic", "Unknown"),
                "severity_assessment": "HIGH",
                "immediate_actions": info.get("mitigation", "Investigate and isolate affected resource."),
                "confidence": 0.82
            })
    return json.dumps({
        "root_cause": "Suspicious activity detected — requires investigation",
        "mitre_technique": "T1059",
        "mitre_name": "Command and Scripting Interpreter",
        "tactic": "Execution",
        "severity_assessment": "MEDIUM",
        "immediate_actions": "Isolate affected resource. Review logs. Escalate to security team.",
        "confidence": 0.55
    })


# ── FAISS Vector Store ─────────────────────────────────────────────────────────

class SimpleVectorStore:
    """Lightweight FAISS-based vector store for RAG retrieval."""

    def __init__(self):
        self.documents = []
        self.embeddings = None
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(L3["embedding_model"])
        return self._model

    def add_documents(self, docs: list[dict]):
        """Add documents with 'text' and 'metadata' keys."""
        texts = [d["text"] for d in docs]
        embs  = self._get_model().encode(texts, show_progress_bar=False)
        self.documents.extend(docs)
        if self.embeddings is None:
            self.embeddings = embs
        else:
            self.embeddings = np.vstack([self.embeddings, embs])

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return top-k most relevant documents."""
        if self.embeddings is None or len(self.documents) == 0:
            return []
        q_emb = self._get_model().encode([query])[0]
        scores = np.dot(self.embeddings, q_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8
        )
        top_k = np.argsort(scores)[::-1][:k]
        return [{"doc": self.documents[i], "score": float(scores[i])} for i in top_k]


# ── Knowledge Base Builder ─────────────────────────────────────────────────────

def build_knowledge_base(vector_store: SimpleVectorStore):
    """Populate the vector store with MITRE ATT&CK entries and past incidents."""
    docs = []

    # MITRE ATT&CK entries
    for tid, info in MITRE_KB.items():
        docs.append({
            "text": f"MITRE {tid} — {info['name']}. Tactic: {info['tactic']}. "
                    f"Mitigation: {info['mitigation']}",
            "metadata": {"type": "mitre", "technique_id": tid, "tactic": info["tactic"]}
        })

    # Past incident patterns
    past_incidents = [
        "Credential brute force attacks typically originate from botnets. "
        "Key indicators: high login failure rate, sequential IP scanning, off-hours activity.",
        "Data exfiltration via reverse shell uses common ports like 4444, 8080, 443 to blend with traffic. "
        "High outbound byte counts to unfamiliar IPs are the main signal.",
        "IAM privilege escalation often occurs by abusing cross-account roles or misconfigured trust policies. "
        "Monitor for AssumeRole API calls from unexpected principals.",
        "Cryptomining in cloud compute shows CPU utilization spikes, increased network traffic, "
        "and connection to known mining pool domains.",
        "Container escape attacks target the metadata endpoint 169.254.169.254 to steal instance credentials. "
        "IMDSv2 enforcement prevents this attack vector.",
        "SQL injection attacks are identified by UNION SELECT patterns in WAF logs, "
        "unusually high query counts, and error responses in application logs.",
        "DNS tunneling uses high-frequency TXT record queries to encode data. "
        "Detection relies on query entropy analysis and domain reputation feeds.",
    ]
    for text in past_incidents:
        docs.append({"text": text, "metadata": {"type": "incident_pattern"}})

    vector_store.add_documents(docs)
    print(f"  Knowledge base: {len(docs)} documents indexed.")
    return vector_store


# ── RAG Analysis Pipeline ──────────────────────────────────────────────────────

def analyse_incident(incident: dict, vector_store: SimpleVectorStore) -> dict:
    """
    Full RAG pipeline for a single incident:
    1. Retrieve relevant context
    2. Build augmented prompt
    3. LLM generates structured analysis
    4. Parse and return result
    """
    query = incident["description"]

    # Retrieve context
    retrieved = vector_store.search(query, k=L3["top_k_retrieval"])
    context = "\n".join([r["doc"]["text"] for r in retrieved])

    system_prompt = """You are a cloud security analyst. Analyse the incident and respond ONLY
with a JSON object containing these fields:
- root_cause: string
- mitre_technique: string (e.g. "T1078")
- mitre_name: string
- tactic: string
- severity_assessment: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
- immediate_actions: string
- confidence: float between 0 and 1

No other text. Only valid JSON."""

    user_prompt = f"""INCIDENT ALERT:
{incident['description']}

AFFECTED RESOURCE: {incident.get('affected_resource', 'Unknown')}
REGION: {incident.get('region', 'us-east-1')}

RELEVANT CONTEXT FROM KNOWLEDGE BASE:
{context}

Analyse this incident and provide structured root cause analysis."""

    start_time = __import__("time").time()
    raw_response = get_llm_response(user_prompt, system_prompt)
    response_time = __import__("time").time() - start_time

    # Parse LLM response
    try:
        if raw_response.strip().startswith("{"):
            analysis = json.loads(raw_response)
        else:
            # Extract JSON block if wrapped in markdown
            import re
            match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            analysis = json.loads(match.group()) if match else {}
    except Exception:
        analysis = {"root_cause": raw_response[:200], "mitre_technique": "PARSE_ERROR"}

    # Evaluate correctness
    gt_technique = incident.get("mitre_technique", "")
    pred_technique = analysis.get("mitre_technique", "")
    technique_correct = gt_technique == pred_technique

    # Semantic root cause match (simplified: check keyword overlap)
    gt_cause = incident.get("ground_truth_root_cause", "").lower()
    pred_cause = analysis.get("root_cause", "").lower()
    gt_words = set(gt_cause.split())
    pred_words = set(pred_cause.split())
    semantic_overlap = len(gt_words & pred_words) / (len(gt_words) + 1e-8)
    cause_correct = semantic_overlap > 0.3

    return {
        "incident_id":          incident["id"],
        "description":          incident["description"][:100] + "...",
        "gt_mitre":             gt_technique,
        "pred_mitre":           pred_technique,
        "technique_correct":    int(technique_correct),
        "gt_root_cause":        incident.get("ground_truth_root_cause", ""),
        "pred_root_cause":      analysis.get("root_cause", ""),
        "cause_correct":        int(cause_correct),
        "severity":             str(analysis.get("severity_assessment", "UNKNOWN")).upper(),
        "confidence":           round(_coerce_float(analysis.get("confidence", 0.0), 0.0), 4),
        "response_time_s":      round(_coerce_float(response_time, 0.0), 3),
        "immediate_actions":    analysis.get("immediate_actions", ""),
        "retrieved_docs":       len(retrieved),
    }


# ── Multi-Agent Supervisor ─────────────────────────────────────────────────────

def supervisor_synthesise(incident_results: list[dict]) -> dict:
    """
    Layer 3 supervisor agent: synthesises findings from multiple
    analysis agents into a unified incident report.
    """
    critical = [r for r in incident_results if r["severity"] == "CRITICAL"]
    high     = [r for r in incident_results if r["severity"] == "HIGH"]

    tactics_seen = set()
    techniques_seen = set()
    for r in incident_results:
        if r["pred_mitre"] in MITRE_KB:
            tactic = MITRE_KB[r["pred_mitre"]]["tactic"]
            tactics_seen.add(tactic)
            techniques_seen.add(r["pred_mitre"])

    # Check for multi-stage attack pattern
    multi_stage = len(tactics_seen) >= 3

    confidences = [_coerce_float(r.get("confidence", 0.0), 0.0) for r in incident_results]
    response_times = [_coerce_float(r.get("response_time_s", 0.0), 0.0) for r in incident_results]

    summary = {
        "total_incidents_analysed": len(incident_results),
        "critical_count": len(critical),
        "high_count":     len(high),
        "unique_tactics": list(tactics_seen),
        "unique_techniques": list(techniques_seen),
        "multi_stage_attack_detected": multi_stage,
        "avg_confidence": round(float(np.mean(confidences)) if confidences else 0.0, 3),
        "avg_response_time_s": round(float(np.mean(response_times)) if response_times else 0.0, 3),
        "recommendation": (
            "ESCALATE: Multi-stage attack campaign detected across multiple tactics. "
            "Immediate isolation of affected resources recommended."
            if multi_stage else
            "Isolated incidents detected. Standard incident response procedures apply."
        )
    }
    return summary


# ── Main Experiment ────────────────────────────────────────────────────────────

def run_layer3_experiment():
    print(f"\n{'='*60}")
    print("LAYER 3: LLM + RAG Cognitive Analysis")
    print(f"LLM Provider: {LLM_PROVIDER} | Model: {LLM_MODEL}")
    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        print("  ⚠ No GROQ_API_KEY set. Using rule-based fallback (offline mode).")
        print("  Set env var: export GROQ_API_KEY=your_key")
    print(f"{'='*60}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load incidents
    incidents_path = DATA_DIR / "synthetic_incidents.csv"
    if not incidents_path.exists():
        print("Incidents not found. Run: python utils/download_datasets.py")
        sys.exit(1)
    df = pd.read_csv(incidents_path)
    incidents = df.to_dict("records")
    print(f"\n[1/4] Loaded {len(incidents)} synthetic incidents.")

    # Build knowledge base
    print("\n[2/4] Building RAG knowledge base...")
    vector_store = SimpleVectorStore()
    with Timer("Knowledge base indexing"):
        build_knowledge_base(vector_store)

    # Run analysis
    print(f"\n[3/4] Analysing incidents with {LLM_PROVIDER} + RAG...")
    results = []
    for i, incident in enumerate(incidents):
        r = analyse_incident(incident, vector_store)
        results.append(r)
        if (i + 1) % 10 == 0:
            running_acc = np.mean([x["technique_correct"] for x in results])
            print(f"  Processed {i+1}/{len(incidents)} | MITRE accuracy: {running_acc:.2f}")

    # Supervisor synthesis
    summary = supervisor_synthesise(results)

    # Metrics
    mitre_acc   = np.mean([r["technique_correct"] for r in results])
    cause_acc   = np.mean([r["cause_correct"]     for r in results])
    avg_conf    = float(np.mean([_coerce_float(r.get("confidence", 0.0), 0.0) for r in results]))
    avg_time    = float(np.mean([_coerce_float(r.get("response_time_s", 0.0), 0.0) for r in results]))

    print(f"\n[4/4] Results:")
    print(f"  MITRE ATT&CK mapping accuracy: {mitre_acc:.2%}")
    print(f"  Root cause semantic accuracy:  {cause_acc:.2%}")
    print(f"  Average confidence score:      {avg_conf:.3f}")
    print(f"  Average response time:         {avg_time:.2f}s")
    print(f"\n  Supervisor summary:")
    for k, v in summary.items():
        print(f"    {k}: {v}")

    # Save
    results_df = pd.DataFrame(results)
    save_results_csv(results, "layer3_rag_analysis_results.csv")

    summary_path = RESULTS_DIR / "layer3_supervisor_summary.json"
    with open(summary_path, "w") as f:
        json.dump({**summary, "mitre_accuracy": mitre_acc, "cause_accuracy": cause_acc}, f, indent=2)
    print(f"  Saved supervisor summary → {summary_path}")

    # Visualise MITRE accuracy
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = ["MITRE Accuracy", "Cause Accuracy", "Avg Confidence"]
    values  = [mitre_acc, cause_acc, avg_conf]
    axes[0].bar(metrics, values, color=["#2563EB","#16A34A","#D97706"])
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Layer 3: RAG Pipeline Metrics", fontsize=13)
    axes[0].bar_label(axes[0].containers[0], fmt="%.2f")

    tactic_counts = pd.Series([r["pred_mitre"] for r in results]).value_counts().head(8)
    axes[1].barh(tactic_counts.index, tactic_counts.values, color="#7C3AED")
    axes[1].set_title("Predicted MITRE Techniques Distribution", fontsize=13)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "layer3_analysis_metrics.png", dpi=150)
    plt.close()
    print(f"  Saved chart → {RESULTS_DIR / 'layer3_analysis_metrics.png'}")

    # MLflow
    with mlflow.start_run(run_name="layer3_rag_analysis"):
        mlflow.log_metrics({
            "mitre_accuracy": mitre_acc,
            "cause_accuracy": cause_acc,
            "avg_confidence": avg_conf,
            "avg_response_time_s": avg_time,
        })
        mlflow.log_param("llm_provider", LLM_PROVIDER)
        mlflow.log_param("llm_model", LLM_MODEL)

    print("\n✅ Layer 3 complete.")
    return {"results": results, "summary": summary, "vector_store": vector_store}


if __name__ == "__main__":
    run_layer3_experiment()
