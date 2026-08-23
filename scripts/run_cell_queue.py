#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import queue
import re
import shlex
import shutil
import subprocess
import threading
from pathlib import Path

from run_command_queue import parse_gpu_ids, reserve_gpu


def load_commands(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Command file not found: {path}")
    commands: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("#!", "#", "set ")):
            continue
        if line.startswith(("COSDA=", "if ")):
            continue
        commands.append(line)
    return commands


def split_cells(commands: list[str]) -> list[list[str]]:
    cells: list[list[str]] = []
    current: list[str] = []
    current_key: str | None = None
    for command in commands:
        key = command_cell_key(command)
        if (" make-seeds " in command and current) or (
            current and key is not None and current_key is not None and key != current_key
        ):
            cells.append(current)
            current = []
            current_key = None
        current.append(command)
        if key is not None:
            current_key = key
    if current:
        cells.append(current)
    return cells


def command_cell_key(command: str) -> str | None:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    for part in parts:
        match = re.search(r"(.+?/b\d+_m\d+_s\d+)(?:/|$)", part)
        if match:
            return match.group(1)
    return None


def cell_name(commands: list[str], fallback: str) -> str:
    try:
        parts = shlex.split(commands[0])
        if "--out" in parts:
            out_path = Path(parts[parts.index("--out") + 1])
            if out_path.name == "gold.jsonl":
                out_path = out_path.parent
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(out_path))
    except (ValueError, IndexError):
        pass
    return fallback


def build_env(gpu: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if "COSDA" not in env:
        local_cosda = Path(".venv/bin/cosda")
        env["COSDA"] = str(local_cosda) if local_cosda.exists() else (shutil.which("cosda") or "cosda")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


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
    work: "queue.Queue[tuple[int, list[str]]]",
    log_dir: Path,
    stop_event: threading.Event,
    stop_on_failure: bool,
) -> None:
    while not stop_event.is_set():
        try:
            index, commands = work.get_nowait()
        except queue.Empty:
            return
        label = cell_name(commands, f"cell_{index:04d}")
        log_path = log_dir / f"{index:04d}_{label}.log"
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
            env = build_env(gpu)
            ok = True
            with log_path.open("w", encoding="utf-8") as log:
                if gpu is not None:
                    log.write(f"CUDA_VISIBLE_DEVICES={gpu}\n")
                log.write(f"gpu_policy={gpu_policy}\n")
                for step, command in enumerate(commands, start=1):
                    if stop_event.is_set():
                        ok = False
                        break
                    log.write(f"\n[{step}/{len(commands)}] $ {command}\n")
                    log.flush()
                    proc = subprocess.run(command, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
                    if proc.returncode != 0:
                        print(f"FAILED [{name}] cell={index} rc={proc.returncode}: {command}\n  log={log_path}")
                        ok = False
                        if stop_on_failure:
                            stop_event.set()
                        break
        if ok:
            print(f"DONE [{name}] cell={index}: {label}")
        work.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a cosda plan-scenario shell file in a cell-aware parallel queue."
    )
    parser.add_argument("command_file", type=Path)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel cells to run.")
    parser.add_argument("--gpus", default="", help="Comma-separated GPU ids for CUDA_VISIBLE_DEVICES.")
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
    parser.add_argument("--dry-run", action="store_true", help="Print detected cells and exit.")
    args = parser.parse_args()

    cells = split_cells(load_commands(args.command_file))
    if not cells:
        raise SystemExit(f"No commands found in {args.command_file}")
    if args.dry_run:
        for index, commands in enumerate(cells):
            print(f"{index:04d}\t{len(commands)} commands\t{cell_name(commands, f'cell_{index:04d}')}")
        return

    log_dir = args.log_dir or (Path(os.environ.get("COSDA_RUNS", "runs")) / "cell_queue_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    gpus = parse_gpu_ids(args.gpus)
    jobs = max(1, args.jobs)
    if gpus and args.gpu_policy == "static" and jobs > len(gpus):
        print(f"jobs={jobs} exceeds GPU count={len(gpus)}; workers will share GPUs round-robin.")
    if args.gpus.strip().lower() in {"auto", "all"} and not gpus:
        raise SystemExit("GPU_IDS=auto requested but no GPUs were detected.")

    work: "queue.Queue[tuple[int, list[str]]]" = queue.Queue()
    for index, commands in enumerate(cells):
        work.put((index, commands))

    stop_event = threading.Event()
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
                stop_event,
                args.stop_on_failure,
            ),
            daemon=True,
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    if stop_event.is_set():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
