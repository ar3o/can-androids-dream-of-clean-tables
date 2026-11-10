"""Clean tables: robot policies from unlabelled phone videos."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# All data, caches and runs live under one folder (several hundred GB once everything is built).
CT_DATA = os.environ.get("CLEANTABLES_DATA", str(ROOT / "workdir"))
