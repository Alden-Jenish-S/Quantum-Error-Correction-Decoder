"""Preserve the complete exploratory 250k stage before the allowed scale-up."""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
target = HERE / "stage_250k"
target.mkdir(exist_ok=False)
names = ["candidate_d.py", "reference_iid.py", "geometry.py", "run_track_d.py",
         "PROTOCOL.md", "training.json", "training_counts.npz", "frozen_tables.py",
         "geometry_audit.json", "results_pilot_250k.jsonl", "results_pilot_250k.summary.json"]
for name in names:
    shutil.copyfile(HERE / name, target / name)
print("Preserved stage_250k")
