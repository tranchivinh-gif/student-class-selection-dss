import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
packages = ["numpy", "pandas", "scikit-learn", "catboost", "xgboost", "matplotlib", "joblib"]
versions = {}
for package in packages:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
except (subprocess.CalledProcessError, FileNotFoundError):
    commit = None
payload = {"python": sys.version, "python_executable": sys.executable, "packages": versions,
           "os": platform.platform(), "cpu_logical": os.cpu_count(),
           "run_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit}
out = ROOT / "experiments" / "results" / "env.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(out)
