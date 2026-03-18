"""
utils/download_datasets.py
Downloads NSL-KDD and CICIDS-style data automatically.
CICIDS 2017 full dataset is large — we use a curated sample here.
"""

import os, requests, zipfile, io
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

NSL_KDD_TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
NSL_KDD_TEST_URL  = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"

NSL_KDD_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

LABEL_MAP = {
    "normal": "normal",
    "neptune": "DoS", "back": "DoS", "land": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "httptunnel": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}


def download_nsl_kdd():
    out_dir = DATA_DIR / "nsl_kdd"
    out_dir.mkdir(exist_ok=True)

    for split, url in [("train", NSL_KDD_TRAIN_URL), ("test", NSL_KDD_TEST_URL)]:
        out_path = out_dir / f"{split}.csv"
        if out_path.exists():
            print(f"  NSL-KDD {split} already exists, skipping.")
            continue
        print(f"  Downloading NSL-KDD {split}...")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), header=None, names=NSL_KDD_COLUMNS)
        df["attack_category"] = df["label"].map(
            lambda x: LABEL_MAP.get(x.strip("."), "other")
        )
        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} rows → {out_path}")


def generate_cicids_sample():
    """
    CICIDS 2017 full dataset is 50GB+. We generate a realistic synthetic
    stand-in with the same feature space for Colab compatibility.
    Real dataset: https://www.unb.ca/cic/datasets/ids-2017.html
    """
    out_path = DATA_DIR / "cicids_sample.csv"
    if out_path.exists():
        print("  CICIDS sample already exists, skipping.")
        return

    print("  Generating CICIDS-format sample dataset...")
    np.random.seed(42)
    n = 20000

    attack_types = [
        "BENIGN", "DoS Hulk", "PortScan", "DDoS",
        "DoS GoldenEye", "FTP-Patator", "SSH-Patator",
        "DoS slowloris", "Bot", "Web Attack – Brute Force",
        "Web Attack – XSS", "Infiltration", "Heartbleed", "Web Attack – Sql Injection"
    ]
    weights = np.array([0.50, 0.12, 0.10, 0.08, 0.05, 0.04, 0.04, 0.03, 0.02, 0.005,
                        0.005, 0.002, 0.001, 0.001])
    weights = weights / weights.sum()

    labels = np.random.choice(attack_types, size=n, p=weights)

    features = {
        "Flow Duration":            np.abs(np.random.normal(50000, 100000, n)).astype(int),
        "Total Fwd Packets":        np.random.randint(1, 200, n),
        "Total Backward Packets":   np.random.randint(0, 150, n),
        "Total Length of Fwd Packets": np.abs(np.random.normal(1000, 2000, n)),
        "Total Length of Bwd Packets": np.abs(np.random.normal(800, 1500, n)),
        "Fwd Packet Length Max":    np.abs(np.random.normal(500, 400, n)),
        "Fwd Packet Length Min":    np.abs(np.random.normal(20, 30, n)),
        "Fwd Packet Length Mean":   np.abs(np.random.normal(200, 150, n)),
        "Bwd Packet Length Max":    np.abs(np.random.normal(400, 350, n)),
        "Bwd Packet Length Mean":   np.abs(np.random.normal(150, 120, n)),
        "Flow Bytes/s":             np.abs(np.random.exponential(50000, n)),
        "Flow Packets/s":           np.abs(np.random.exponential(500, n)),
        "Flow IAT Mean":            np.abs(np.random.normal(5000, 10000, n)),
        "Flow IAT Std":             np.abs(np.random.normal(3000, 6000, n)),
        "Fwd IAT Total":            np.abs(np.random.normal(20000, 40000, n)),
        "Bwd IAT Total":            np.abs(np.random.normal(15000, 30000, n)),
        "SYN Flag Count":           np.random.randint(0, 5, n),
        "RST Flag Count":           np.random.randint(0, 3, n),
        "PSH Flag Count":           np.random.randint(0, 10, n),
        "ACK Flag Count":           np.random.randint(0, 20, n),
        "Average Packet Size":      np.abs(np.random.normal(200, 150, n)),
        "Avg Fwd Segment Size":     np.abs(np.random.normal(180, 130, n)),
        "Init_Win_bytes_forward":   np.random.randint(0, 65535, n),
        "Init_Win_bytes_backward":  np.random.randint(0, 65535, n),
        "Label": labels,
    }

    # Add attack-specific signal so classifiers can actually learn
    dos_mask = np.isin(labels, ["DoS Hulk","DoS GoldenEye","DoS slowloris","DDoS"])
    features["Flow Packets/s"][dos_mask] *= 8
    features["SYN Flag Count"][dos_mask] += np.random.randint(5, 20, dos_mask.sum())

    scan_mask = labels == "PortScan"
    features["Total Fwd Packets"][scan_mask] = np.random.randint(1, 3, scan_mask.sum())
    features["RST Flag Count"][scan_mask] += np.random.randint(1, 5, scan_mask.sum())

    df = pd.DataFrame(features)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")


def generate_synthetic_incidents():
    """Generate labelled incident dataset for Layer 3/4 LLM evaluation."""
    out_path = DATA_DIR / "synthetic_incidents.csv"
    if out_path.exists():
        print("  Synthetic incidents already exist, skipping.")
        return

    print("  Generating synthetic incident dataset...")
    incidents = [
        {"id": 1, "description": "Unusual outbound traffic spike from EC2 instance i-0abc123 to external IP 203.0.113.45 on port 4444. 50GB transferred in 2 hours.", "ground_truth_root_cause": "Data exfiltration via reverse shell", "mitre_technique": "T1041", "severity": "critical", "affected_resource": "EC2:i-0abc123"},
        {"id": 2, "description": "Multiple failed SSH login attempts (847 in 10 min) against bastion host from IP 198.51.100.23.", "ground_truth_root_cause": "Brute force credential attack", "mitre_technique": "T1110", "severity": "high", "affected_resource": "EC2:bastion-host"},
        {"id": 3, "description": "IAM user 'svc-account-dev' assumed admin role at 3:47 AM and created new IAM user with AdministratorAccess policy.", "ground_truth_root_cause": "Privilege escalation via IAM role abuse", "mitre_technique": "T1078", "severity": "critical", "affected_resource": "IAM:svc-account-dev"},
        {"id": 4, "description": "S3 bucket 'prod-customer-data' made public. Bucket contains 2.3M customer PII records.", "ground_truth_root_cause": "Misconfiguration exposing sensitive data", "mitre_technique": "T1530", "severity": "critical", "affected_resource": "S3:prod-customer-data"},
        {"id": 5, "description": "Lambda function execution time spiked to 15 minutes. CPU at 100%. Function processing payment transactions.", "ground_truth_root_cause": "Cryptomining code injection in Lambda", "mitre_technique": "T1496", "severity": "high", "affected_resource": "Lambda:payment-processor"},
        {"id": 6, "description": "CloudTrail logging disabled in us-east-1 region by API call from root account at 02:15 UTC.", "ground_truth_root_cause": "Defense evasion via audit log tampering", "mitre_technique": "T1562", "severity": "critical", "affected_resource": "CloudTrail:us-east-1"},
        {"id": 7, "description": "New security group rule added allowing inbound traffic from 0.0.0.0/0 on port 3389 (RDP) to production VPC.", "ground_truth_root_cause": "Backdoor access via misconfigured security group", "mitre_technique": "T1133", "severity": "high", "affected_resource": "SecurityGroup:sg-prod"},
        {"id": 8, "description": "Kubernetes pod in namespace 'default' attempting to access AWS metadata endpoint 169.254.169.254 repeatedly.", "ground_truth_root_cause": "Container escape attempt to steal cloud credentials", "mitre_technique": "T1552", "severity": "critical", "affected_resource": "K8s:default/suspicious-pod"},
        {"id": 9, "description": "RDS database receiving 10,000 queries/second with UNION SELECT statements. WAF triggered 450 alerts.", "ground_truth_root_cause": "SQL injection attack against database", "mitre_technique": "T1190", "severity": "high", "affected_resource": "RDS:prod-database"},
        {"id": 10, "description": "DNS queries from internal host to domain 'update.microsoft-security.xyz' — 500 queries in 1 hour with TXT record responses containing encoded data.", "ground_truth_root_cause": "DNS tunneling for C2 communication", "mitre_technique": "T1071", "severity": "high", "affected_resource": "Host:192.168.1.45"},
    ]

    # Expand to 100 incidents by varying the 10 templates
    expanded = []
    regions = ["us-east-1","eu-west-1","ap-southeast-1","us-west-2"]
    for i in range(100):
        base = incidents[i % 10].copy()
        base["id"] = i + 1
        base["region"] = regions[i % 4]
        base["timestamp"] = f"2025-{(i%12)+1:02d}-{(i%28)+1:02d}T{(i%24):02d}:00:00Z"
        expanded.append(base)

    df = pd.DataFrame(expanded)
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} incidents → {out_path}")


if __name__ == "__main__":
    print("=== Downloading / generating datasets ===")
    download_nsl_kdd()
    generate_cicids_sample()
    generate_synthetic_incidents()
    print("\nAll datasets ready.")
