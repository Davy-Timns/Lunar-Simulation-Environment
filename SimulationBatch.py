"""Sequential, isolated simulation workers and an incrementally saved manifest."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

WORKER_CONFIG_ENV = "SERVAL_BATCH_CONFIG"


def write_json(path, value):
    """Replace a JSON document atomically; never emit nonstandard NaN/Infinity."""
    import math
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if hasattr(item, "tolist"):
            return clean(item.tolist())
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, Path):
            return str(item)
        return item
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_seeds(base_seed, run_index):
    import numpy as np
    names = ("fault_schedule", "fault_runtime", "imu", "star_tracker", "xy", "range")
    sequence = np.random.SeedSequence([base_seed, run_index])
    return {name: int(child.generate_state(1)[0]) for name, child in zip(names, sequence.spawn(len(names)))}


def source_fingerprint(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*.py")):
        if any(part in {".venv", ".git", "work", "results", "__pycache__"} for part in path.relative_to(root).parts):
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_batch(script, amount, base_seed, output_root, options=None):
    if isinstance(amount, bool) or not isinstance(amount, int) or amount < 1:
        raise ValueError("RUN_SIM_AMOUNT must be a positive integer")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("BATCH_BASE_SEED must be a nonnegative integer")
    script = Path(script).resolve()
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = Path(output_root).resolve() / batch_id
    directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, batch_id=batch_id, requested_runs=amount,
                    base_seed=base_seed, source_sha256=source_fingerprint(script.parent),
                    status="running", runs=[])
    manifest_path = directory / "manifest.json"
    write_json(manifest_path, manifest)
    print(f"Batch output: {directory}", flush=True)
    try:
        for index in range(amount):
            run_id = f"run_{index + 1:06d}"
            run_dir = directory / run_id
            run_dir.mkdir()
            config = dict(options or {}, run_id=run_id, output_dir=str(run_dir),
                          seeds=run_seeds(base_seed, index), source_sha256=manifest["source_sha256"])
            config_path = run_dir / "config.json"
            write_json(config_path, config)
            entry = dict(run_id=run_id, directory=run_id, execution_status="running",
                         landing_outcome="failure", landing_success=False,
                         landed=False, seeds=config["seeds"])
            manifest["runs"].append(entry)
            write_json(manifest_path, manifest)
            environment = dict(os.environ, **{WORKER_CONFIG_ENV: str(config_path), "MPLBACKEND": "Agg"})
            started = time.monotonic()
            print(f"[{index + 1}/{amount}] {run_id}; log: {run_dir / 'run.log'}", flush=True)
            try:
                with (run_dir / "run.log").open("w", encoding="utf-8") as log:
                    child = subprocess.run([sys.executable, "-B", str(script)], cwd=script.parent,
                                           env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
                entry["return_code"] = child.returncode
                outcome_path = run_dir / "outcome.json"
                outcome = json.loads(outcome_path.read_text(encoding="utf-8")) if outcome_path.exists() else {}
                if child.returncode or not outcome:
                    outcome.update(execution_status="error", landing_outcome="failure", landing_success=False,
                                   failure_reason=outcome.get("failure_reason", "worker_failed; see run.log"))
                    outcome.setdefault("landed", False)
                    outcome.setdefault("landing_diagnostics", None)
                    write_json(outcome_path, outcome)
                entry.update({key: outcome.get(key) for key in
                              ("execution_status", "landing_outcome", "landing_success", "landed", "failure_reason")})
                entry["outcome_file"] = f"{run_id}/outcome.json"
            except Exception:
                entry.update(execution_status="error", failure_reason=traceback.format_exc())
                write_json(run_dir / "outcome.json", entry)
            entry["wall_time_s"] = time.monotonic() - started
            entry["has_dataset"] = (run_dir / "data.npz").is_file()
            entry["has_vizard_recording"] = (run_dir / "vizard_UnityViz.bin").is_file()
            write_json(manifest_path, manifest)
            print(f"  {entry['execution_status']}; landing: {entry['landing_outcome']}", flush=True)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        if manifest["runs"] and manifest["runs"][-1]["execution_status"] == "running":
            manifest["runs"][-1].update(execution_status="interrupted", failure_reason="batch_interrupted")
            write_json(directory / manifest["runs"][-1]["run_id"] / "outcome.json", manifest["runs"][-1])
        write_json(manifest_path, manifest)
        raise
    manifest["status"] = "completed"
    manifest["successful_landings"] = sum(bool(run["landing_success"]) for run in manifest["runs"])
    manifest["execution_errors"] = sum(run["execution_status"] != "completed" for run in manifest["runs"])
    write_json(manifest_path, manifest)
    return directory
