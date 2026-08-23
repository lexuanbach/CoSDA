#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import os
import queue
import shlex
import subprocess
import threading
import time
from pathlib import Path
import shutil


def load_commands(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Command file not found: {path}")
    commands = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#!") or line.startswith("set ") or line.startswith("COSDA=") or line.startswith("if "):
            continue
        commands.append(line)
    return commands


def visible_gpu_ids() -> list[str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def parse_gpu_ids(raw: str) -> list[str]:
    value = raw.strip().lower()
    if value in {"auto", "all"}:
        return visible_gpu_ids()
    return [gpu.strip() for gpu in raw.split(",") if gpu.strip()]


def query_gpu_stats() -> dict[str, tuple[int, int]]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi not found; cannot use wait-free GPU scheduling") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {proc.stderr.strip()}")
    stats: dict[str, tuple[int, int]] = {}
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            stats[parts[0]] = (int(parts[1]), int(parts[2]))
    return stats


def try_lock_gpu(lock_dir: Path, gpu: str):
    lock_dir.mkdir(parents=True, exist_ok=True)
    handle = (lock_dir / f"gpu_{gpu}.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.write(f"pid={os.getpid()} gpu={gpu}\n")
    handle.flush()
    return handle


@contextmanager
def reserve_gpu(
    gpus: list[str],
    policy: str,
    static_gpu: str | None,
    max_memory_used_mib: int,
    max_utilization: int,
    poll_seconds: float,
    lock_dir: Path,
    worker_name: str,
    idle_confirmations: int = 1,
):
    if not gpus:
        yield None
        return
    if policy == "static":
        yield static_gpu
        return

    last_status = 0.0
    idle_counts = {gpu: 0 for gpu in gpus}
    while True:
        stats = query_gpu_stats()
        snapshots = []
        for gpu in gpus:
            memory_used, utilization = stats.get(gpu, (10**9, 100))
            snapshots.append(f"{gpu}:mem={memory_used}MiB,util={utilization}%")
            if memory_used > max_memory_used_mib or utilization > max_utilization:
                idle_counts[gpu] = 0
                continue
            idle_counts[gpu] = idle_counts.get(gpu, 0) + 1
            if idle_counts[gpu] < max(1, idle_confirmations):
                continue
            handle = try_lock_gpu(lock_dir, gpu)
            if handle is None:
                continue
            try:
                yield gpu
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
            return
        now = time.monotonic()
        if now - last_status >= max(30.0, poll_seconds):
            print(f"WAIT [{worker_name}] no free GPU under thresholds; " + " | ".join(snapshots), flush=True)
            last_status = now
        time.sleep(poll_seconds)


def worker(
    name: str,
    static_gpu: str | None,
    gpus: list[str],
    gpu_policy: str,
    max_gpu_memory_used_mib: int,
    max_gpu_utilization: int,
    gpu_poll_seconds: float,
    gpu_lock_dir: Path,
    idle_confirmations: int,
    work: "queue.Queue[str]",
    log_dir: Path,
    stop_on_failure: bool,
) -> None:
    while True:
        try:
            cmd = work.get_nowait()
        except queue.Empty:
            return
        safe_name = f"{name}_{work.qsize()}".replace("/", "_")
        log_path = log_dir / f"{safe_name}.log"
        with reserve_gpu(
            gpus,
            gpu_policy,
            static_gpu,
            max_gpu_memory_used_mib,
            max_gpu_utilization,
            gpu_poll_seconds,
            gpu_lock_dir,
            name,
            idle_confirmations,
        ) as gpu:
            env = os.environ.copy()
            if "COSDA" not in env:
                local_cosda = Path(".venv/bin/cosda")
                env["COSDA"] = str(local_cosda) if local_cosda.exists() else (shutil.which("cosda") or "cosda")
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu
            if "OMP_NUM_THREADS" not in env:
                env["OMP_NUM_THREADS"] = "2"
            if "MKL_NUM_THREADS" not in env:
                env["MKL_NUM_THREADS"] = "2"
            if "TOKENIZERS_PARALLELISM" not in env:
                env["TOKENIZERS_PARALLELISM"] = "false"
            with log_path.open("w", encoding="utf-8") as log:
                log.write(f"$ {cmd}\n")
                if gpu is not None:
                    log.write(f"CUDA_VISIBLE_DEVICES={gpu}\n")
                log.write(f"gpu_policy={gpu_policy}\n")
                log.flush()
                proc = subprocess.run(cmd, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            print(f"FAILED [{name}] rc={proc.returncode}: {cmd}\n  log={log_path}")
            if stop_on_failure:
                os._exit(proc.returncode)
        else:
            print(f"DONE [{name}]: {cmd}")
        work.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shell commands in a simple GPU-aware queue.")
    parser.add_argument("command_file", type=Path)
    parser.add_argument("--gpus", default="", help="Comma-separated GPU ids, e.g. 0,1,2,3. Empty means CPU/no assignment.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers. Defaults to number of GPUs if --gpus is set.")
    parser.add_argument(
        "--gpu-policy",
        choices=["static", "wait-free"],
        default="static",
        help="static assigns workers round-robin; wait-free waits for a GPU below memory/util thresholds and locks it.",
    )
    parser.add_argument("--max-gpu-memory-used-mib", type=int, default=1024)
    parser.add_argument("--max-gpu-utilization", type=int, default=10)
    parser.add_argument("--gpu-poll-seconds", type=float, default=20.0)
    parser.add_argument("--gpu-lock-dir", type=Path, default=Path("/tmp/cosda_gpu_locks"))
    parser.add_argument(
        "--idle-confirmations",
        type=int,
        default=1,
        help="Require this many consecutive idle GPU polls before reserving a GPU.",
    )
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    commands = load_commands(args.command_file)
    log_dir = args.log_dir or (Path(os.environ.get("COSDA_RUNS", "runs")) / "queue_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    gpus = parse_gpu_ids(args.gpus)
    jobs = args.jobs if args.jobs > 0 else max(1, len(gpus))
    if gpus and args.gpu_policy == "static" and jobs > len(gpus):
        print(f"jobs={jobs} exceeds GPU count={len(gpus)}; multiple workers will share GPUs round-robin.")
    if args.gpus.strip().lower() in {"auto", "all"} and not gpus:
        raise SystemExit("GPU_IDS=auto requested but no GPUs were detected.")

    work: "queue.Queue[str]" = queue.Queue()
    for cmd in commands:
        work.put(cmd)

    threads = []
    for idx in range(jobs):
        gpu = gpus[idx % len(gpus)] if gpus else None
        thread = threading.Thread(
            target=worker,
            args=(
                f"w{idx}",
                gpu,
                gpus,
                args.gpu_policy,
                args.max_gpu_memory_used_mib,
                args.max_gpu_utilization,
                args.gpu_poll_seconds,
                args.gpu_lock_dir,
                args.idle_confirmations,
                work,
                log_dir,
                args.stop_on_failure,
            ),
            daemon=True,
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
