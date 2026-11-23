#!/usr/bin/env python3
"""Continue final evaluation only after recorded user approval; never train or publish."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import traceback

GPU0 = "GPU-cfbe689d-b422-47e1-f640-5940af92cb57"
GPU1 = "GPU-f8bdb15e-74dd-4dd9-8d6b-a54083638d92"
FRACTION = 0.25
COMMIT = "ab78ea34b49edc000e8dc05128857ce6b036c721"
TOKENS, STEPS = 2_999_910_400, 18_310
TASKS = ["hellaswag", "piqa", "arc_easy", "arc_challenge"]
VALIDATION_SHA = "cd9b0e5cbf20e45922905d249a1dfe397d66abe886aa622d2632eaeb604e81a8"
TOKENIZER_SHA = "78025606dd6a505cd74f4b187866bf5540ee83239131aa2f1ddefdfd2e104272"
REVISIONS = {"Rowan/hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
             "baber/piqa": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
             "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453"}
WRAPPER = "import sys,torch,runpy; target=sys.argv.pop(1); sys.argv[0]=target; torch.cuda.set_per_process_memory_fraction(0.25,0); runpy.run_path(target,run_name='__main__')"
CACHE_KEYS = {"HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE",
              "XDG_CACHE_HOME", "MPLCONFIGDIR", "CUDA_CACHE_PATH", "TMPDIR", "TEMP", "TMP",
              "TORCH_EXTENSIONS_DIR", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    require(path.is_file() and not path.is_symlink(), f"Missing or symlinked evidence: {path}")
    def invalid(value):
        raise ValueError(f"Non-finite evidence value: {value}")
    return json.loads(path.read_text(), parse_constant=invalid)


def validate_transition(before, after, approval, root):
    require(approval.get("approved") is True and approval.get("approved_by") == "user"
            and approval.get("action") == "shared_gpu1_evaluation"
            and approval.get("gpu_uuid") == GPU1 and approval.get("memory_fraction") == FRACTION
            and approval.get("root") == str(root) and bool(approval.get("user_instruction")),
            "Explicit user approval for this shared GPU1 evaluation is required")
    approved_at = datetime.fromisoformat(approval["approved_at"])
    require(approved_at.tzinfo is not None, "Approval timestamp must include its timezone")
    require(before.get("state") == "running" and before.get("stage") == "wait_gpu_loss",
            "Before-stop evidence must show the original supervisor waiting for final loss evaluation")
    require(after.get("state") == "failed" and after.get("stage") == "wait_gpu_loss"
            and after.get("error") == "Supervisor received signal 15",
            "Only deliberate SIGTERM while waiting for final evaluation can be continued")
    for status in (before, after):
        require(status.get("source_commit") == COMMIT and status.get("gpu_uuid") == GPU0
                and status.get("root") == str(root), "Unexpected training source, GPU, or root")
    require(before.get("supervisor_pid") == after.get("supervisor_pid"), "Supervisor PID changed")
    stages = after.get("stages", [])
    require(stages and before.get("stages", [])[:-1] == stages[:-1], "Completed stages changed during stop")
    require(all(s.get("state") == "complete" and s.get("exit_code", 0) == 0 for s in stages[:-1]),
            "All previous stages must have genuinely completed")
    for name in ("prepare", "training", "plots", "checkpoint_check"):
        require(sum(s.get("name") == name for s in stages[:-1]) == 1, f"Missing completed {name}")
    require(not any(s.get("name") in ("validation", "benchmarks", "evaluation_packages") for s in stages),
            "Final evaluation already started; manual review is required")
    prior_wait, wait = before["stages"][-1], stages[-1]
    require(prior_wait.get("name") == wait.get("name") == "wait_gpu_loss"
            and prior_wait.get("state") == "running" and wait.get("state") == "failed"
            and prior_wait.get("started_at") == wait.get("started_at")
            and wait.get("exit_code") is None, "Unexpected waiting-stage transition")
    ended = datetime.fromisoformat(after["failed_at"])
    started = datetime.fromisoformat(wait["started_at"])
    require(started <= ended and approved_at <= ended <= datetime.now(timezone.utc),
            "Approval/stop timestamps are inconsistent")
    return (ended - started).total_seconds()


def validate_checkpoint(report, metrics):
    for key, value in {"completed_steps": STEPS, "training_tokens": TOKENS, "parameters": 146_496_016,
                       "seed": 1337, "scratch": True, "strict_model_load": True,
                       "finite_model_parameters": True}.items():
        require(report.get(key) == value, f"Checkpoint report failed: {key}")
    starts = [r for r in metrics if r.get("event") == "start"]
    ends = [r for r in metrics if r.get("event") == "complete"]
    require(len(starts) == len(ends) == 1 and starts[0].get("step") == 0
            and starts[0].get("resume") is None and ends[0].get("step") == STEPS
            and ends[0].get("tokens") == TOKENS, "Training is not one completed scratch run")


def capture_gpu(supervisor, base, label, require_capacity=False):
    raw = supervisor.query(["nvidia-smi", "-i", "1", "--query-gpu=uuid,name,memory.total,memory.free,utilization.gpu",
                            "--format=csv,noheader,nounits"])
    fields = [field.strip() for field in raw.split(",")]
    require(len(fields) == 5 and fields[0] == GPU1 and "RTX 3090" in fields[1], "Evaluation GPU identity changed")
    total, free = float(fields[2]), float(fields[3])
    if require_capacity:
        require(free >= total * FRACTION + 1024, "Insufficient free GPU1 memory for the approved cap plus context allowance")
    snapshot = {"captured_at": base.utc_now(), "gpu_index": 1, "gpu_uuid": fields[0], "name": fields[1],
                "memory_total_mib": total, "memory_free_mib": free, "utilization_percent": float(fields[4]),
                "compute_processes": supervisor.query(["nvidia-smi", "-i", "1", "--query-compute-apps=pid,used_memory",
                                                        "--format=csv,noheader,nounits"])}
    base.write_json(supervisor.artifacts / "evaluation_continuation" / (label + "_gpu.json"), snapshot)


def validate_results(loss, benchmarks, checkpoint_hash):
    require(loss.get("training_tokens") == TOKENS and loss.get("step") == STEPS
            and loss.get("checkpoint_sha256") == checkpoint_hash and loss.get("batch_size") == 20
            and loss.get("full_vocabulary") is True and loss.get("validation_tokens") == 262144
            and loss.get("validation_sha256") == VALIDATION_SHA and loss.get("tokenizer_sha256") == TOKENIZER_SHA
            and loss.get("sequences") == 128 and loss.get("sequence_length") == 2048
            and loss.get("dtype") == "torch.bfloat16" and math.isfinite(loss["loss"]), "Invalid frozen loss report")
    require(benchmarks.get("step") == STEPS and benchmarks.get("zero_shot") is True
            and "limit" in benchmarks and benchmarks["limit"] is None
            and benchmarks.get("truncated_requests") == 0
            and benchmarks.get("dataset_revisions") == REVISIONS
            and set(benchmarks.get("results", {})) == set(TASKS), "Incomplete benchmark protocol")
    for task in TASKS:
        for metric in ("acc,none", "acc_norm,none", "acc_stderr,none", "acc_norm_stderr,none"):
            value = benchmarks["results"][task][metric]
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                    f"Invalid {task}/{metric}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    artifacts = root / "artifacts"
    continuation = artifacts / "evaluation_continuation"
    approval = read(continuation / "approval.json")
    before = read(continuation / "status_before_stop.json")
    after = read(continuation / "status_after_stop.json")
    waited_seconds = validate_transition(before, after, approval, root)
    lock = (artifacts / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(read(artifacts / "status.json") == after, "Current status differs from archived termination evidence")
    original_plan = read(artifacts / "launch_plan.json")
    module_path = artifacts / "supervise_run.py"
    require(sha(module_path) == original_plan["supervisor_sha256"], "Original supervisor source changed")
    spec = importlib.util.spec_from_file_location("original_supervisor", module_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    supervisor = base.Supervisor.__new__(base.Supervisor)
    supervisor.root, supervisor.artifacts, supervisor.lock = root, artifacts, lock
    supervisor.status_path = artifacts / "status.json"
    supervisor.python = str(root / ".venv/bin/python")
    supervisor.data = root / "data/ultrafineweb_3b"
    supervisor.run_dir = root / "runs" / base.RUN_NAME
    supervisor.child = None
    supervisor.env = base.clean_environment()
    caches = original_plan["runtime_cache_environment"]
    require(set(caches) == CACHE_KEYS and all(Path(value).resolve().is_relative_to(artifacts / "runtime_caches")
                                            for value in caches.values()), "Unexpected runtime cache settings")
    supervisor.env.update(caches)
    checkpoint = supervisor.run_dir / "final_l3_mix.pt"
    require(checkpoint.is_file() and not checkpoint.is_symlink(), "Missing final checkpoint")
    metrics = [json.loads(line) for line in (supervisor.run_dir / "metrics.jsonl").read_text().splitlines()]
    validate_checkpoint(read(artifacts / "checkpoint_report.json"), metrics)
    for path in (artifacts / "evaluation_plan.json", artifacts / "validation.log", artifacts / "benchmarks.log",
                 artifacts / "evaluation_packages.log", supervisor.run_dir / "validation.json", supervisor.run_dir / "evaluation.json",
                 continuation / "finish_evaluation.py"):
        require(not path.exists(), f"Continuation output already exists; manual review required: {path}")
    checkpoint_hash, timing_hash, launch_hash = sha(checkpoint), sha(artifacts / "timing.json"), sha(artifacts / "launch_plan.json")
    with (continuation / "finish_evaluation.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    plan = {"started_at": base.utc_now(), "training_gpu_uuid": GPU0, "evaluation_gpu_index": 1,
            "evaluation_gpu_uuid": GPU1, "shared": True, "memory_fraction": FRACTION,
            "memory_limit_scope": "PyTorch allocator; CUDA context and external allocations are additional",
            "checkpoint_sha256": checkpoint_hash, "training_timing_sha256": timing_hash,
            "launch_plan_sha256": launch_hash, "helper_sha256": sha(Path(__file__)),
            "approval_sha256": sha(continuation / "approval.json"),
            "status_before_stop_sha256": sha(continuation / "status_before_stop.json"),
            "status_after_stop_sha256": sha(continuation / "status_after_stop.json"),
            "reason": "User-authorized final evaluation on shared GPU1; original GPU0 wait ended without observing it idle"}
    base.write_json(artifacts / "evaluation_plan.json", plan)
    supervisor.status = copy.deepcopy(after)
    for key in ("error", "failed_at", "recovery"):
        supervisor.status.pop(key, None)
    wait = supervisor.status["stages"][-1]
    wait.update(state="complete", outcome="ended_by_user_authorized_reallocation", gpu_idle_observed=False,
                original_state="failed", original_supervisor_error=after["error"], finished_at=after["failed_at"],
                wall_seconds=waited_seconds, duration_source="UTC started_at to manual SIGTERM failed_at")
    wait.pop("exit_code", None)
    supervisor.status.update(state="running", stage="evaluation_continuation", supervisor_pid=os.getpid(),
                             supervisor_parent_pid=os.getppid(), evaluation_gpu_uuid=GPU1,
                             original_supervisor_pid=after["supervisor_pid"])
    supervisor.persist()
    (artifacts / "supervisor.pid").write_text(str(os.getpid()) + "\n")
    def stop(signum, frame):
        raise RuntimeError(f"Evaluation continuation received signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        gpu_env = supervisor.env | {"CUDA_VISIBLE_DEVICES": GPU1}
        loss_path, benchmark_path = supervisor.run_dir / "validation.json", supervisor.run_dir / "evaluation.json"
        capture_gpu(supervisor, base, "before_validation", require_capacity=True)
        supervisor.command("validation", [supervisor.python, "-c", WRAPPER, "scripts/evaluate_loss.py",
                           "--checkpoint", str(checkpoint), "--data-dir", str(supervisor.data), "--out", str(loss_path),
                           "--batch-size", "20", "--device", "cuda", "--dtype", "bf16"], gpu_env)
        capture_gpu(supervisor, base, "after_validation")
        require(Path(base.EVAL_OVERLAY).is_dir(), "Missing historical benchmark dependencies")
        eval_env = gpu_env | {"PYTHONPATH": base.EVAL_OVERLAY, "OMP_NUM_THREADS": "2",
                             "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
        supervisor.command("evaluation_packages", [supervisor.python, "-c", base.PACKAGES_PROBE], eval_env)
        base.write_json(artifacts / "evaluation_packages.json", json.loads((artifacts / "evaluation_packages.log").read_text()))
        capture_gpu(supervisor, base, "before_benchmarks", require_capacity=True)
        supervisor.command("benchmarks", [supervisor.python, "-c", WRAPPER, "scripts/evaluate.py",
                           "--checkpoint", str(checkpoint), "--data-dir", str(supervisor.data), "--out", str(benchmark_path),
                           "--batch-size", "8", "--tasks", *TASKS], eval_env)
        capture_gpu(supervisor, base, "after_benchmarks")
        loss, benchmarks = read(loss_path), read(benchmark_path)
        validate_results(loss, benchmarks, checkpoint_hash)
        require(sha(checkpoint) == checkpoint_hash and sha(artifacts / "timing.json") == timing_hash
                and sha(artifacts / "launch_plan.json") == launch_hash, "Training evidence changed during evaluation")
        for filename, key in (("approval.json", "approval_sha256"), ("status_before_stop.json", "status_before_stop_sha256"),
                              ("status_after_stop.json", "status_after_stop_sha256"), ("finish_evaluation.py", "helper_sha256")):
            require(sha(continuation / filename) == plan[key], f"Continuation evidence changed: {filename}")
        supervisor.manifest("evaluation_continuation_manifest", [artifacts / "evaluation_plan.json",
                             *[p for p in continuation.iterdir() if p.is_file()]], artifacts)
        supervisor.manifest("results_manifest", [p for p in supervisor.run_dir.rglob("*") if p.is_file()], supervisor.run_dir)
        timing = read(artifacts / "timing.json")
        supervisor.status.update(state="complete", stage="complete", finished_at=base.utc_now(),
                                 training_launcher_wall_seconds=timing["training_launcher_wall_seconds"],
                                 validation_loss=loss["loss"], publication="pending_local_collection")
        supervisor.persist()
    except BaseException as error:
        traceback.print_exc()
        supervisor.fail(error)
        raise


if __name__ == "__main__":
    main()
