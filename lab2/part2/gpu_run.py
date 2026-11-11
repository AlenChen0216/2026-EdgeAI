#!/usr/bin/env python3
"""GPU queue and logs, using only the standard library (Linux/NVIDIA).

Kept identical in Part2 and Part3 so each submission directory is standalone.
The entrypoint present beside this file determines which experiments to run.
"""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


MIN_FREE_MIB = 12 * 1024
MAX_PARALLEL = 2
LOCK_ROOT = Path("/tmp") / f"edgeai-gpu-{os.getuid()}"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def query_gpus():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"nvidia-smi failed ({exc.returncode}): {exc.stderr.strip()}") from exc
    gpus = []
    try:
        for row in csv.reader(io.StringIO(result.stdout)):
            index, uuid, total, used, free = (value.strip() for value in row)
            gpu = dict(index=int(index), uuid=uuid, total_mib=int(total),
                       used_mib=int(used), free_mib=int(free))
            if (gpu["index"] < 0 or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid)
                    or gpu["total_mib"] <= 0
                    or not 0 <= gpu["used_mib"] <= gpu["total_mib"]
                    or not 0 <= gpu["free_mib"] <= gpu["total_mib"]):
                raise ValueError(f"invalid GPU record: {row}")
            gpus.append(gpu)
        if (len({gpu["index"] for gpu in gpus}) != len(gpus)
                or len({gpu["uuid"] for gpu in gpus}) != len(gpus)):
            raise ValueError("duplicate GPU index or UUID")
    except ValueError as exc:
        raise RuntimeError(f"Cannot parse nvidia-smi memory data: {exc}") from exc
    if not gpus:
        raise RuntimeError("nvidia-smi reported no GPUs")
    # Preserve an existing visibility restriction; bind children using physical UUIDs.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        allowed = set()
        for token in visible.split(","):
            token = token.strip()
            matches = [gpu for gpu in gpus if token == str(gpu["index"])
                       or (token.startswith("GPU-") and gpu["uuid"].startswith(token))]
            if len(matches) != 1:
                raise RuntimeError(f"CUDA_VISIBLE_DEVICES has no unique GPU match: {token!r}")
            allowed.add(matches[0]["uuid"])
        gpus = [gpu for gpu in gpus if gpu["uuid"] in allowed]
    return gpus


def claim_gpu(busy):
    """Take a nonblocking lock, then query again before admitting a job."""
    candidates = sorted(query_gpus(), key=lambda gpu: (-gpu["free_mib"], gpu["index"]))
    LOCK_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    for gpu in candidates:
        if gpu["uuid"] in busy or gpu["free_mib"] <= MIN_FREE_MIB:
            continue
        lock = (LOCK_ROOT / f'{gpu["uuid"]}.lock').open("a")
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            current = next((row for row in query_gpus() if row["uuid"] == gpu["uuid"]), None)
            if current is None:
                raise RuntimeError(f'GPU disappeared: {gpu["uuid"]}')
            if current["free_mib"] > MIN_FREE_MIB:
                claimed = lock
                lock = None
                return current, claimed
        finally:
            if lock is not None:
                lock.close()
    return None


class Interrupted(Exception):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"Interrupted by signal {signum}")


class Runner:
    def __init__(self, root, arguments, poll_seconds=30):
        self.root = Path(root).resolve()
        self.arguments = arguments
        self.poll_seconds = poll_seconds
        self.python = os.environ.get("PYTHON") or sys.executable
        self.part2 = (self.root / "314552017_run.sh").exists()
        self.entrypoint = "314552017_run.sh" if self.part2 else "run.sh"
        self.directory = self.root / "results" / "gpu_runs" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + f"_{os.getpid()}")
        self.directory.mkdir(parents=True)
        names = [f"case{i}" for i in range(1, 5)] if self.part2 else ["experiment"]
        self.summary = dict(started_at=timestamp(), status="running", min_free_mib=MIN_FREE_MIB,
                            poll_seconds=poll_seconds, jobs=[dict(name=name, status="pending")
                                                            for name in names])
        self.active = {}
        self.setup_process = None

    def save(self):
        temporary = self.directory / "summary.json.tmp"
        temporary.write_text(json.dumps(self.summary, indent=2) + "\n")
        temporary.replace(self.directory / "summary.json")

    def event(self, message):
        with (self.directory / "runner.log").open("a") as output:
            output.write(f"{timestamp()} {message}\n")

    def setup(self):
        with (self.directory / "setup.log").open("w") as output:
            if os.environ.get("SKIP_INSTALL") == "1":
                output.write("SKIP_INSTALL=1; dependency installation skipped.\n")
                return
            command = [self.python, "-m", "pip", "install", "-r", "requirements.txt"]
            self.summary["setup_command"] = command
            self.setup_process = subprocess.Popen(
                command, cwd=self.root, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            code = self.setup_process.wait()
            self.summary["setup_exit_code"] = code
            self.setup_process = None
            if code:
                raise RuntimeError(f"Dependency installation failed ({code}); see setup.log")

    def launch(self, job, gpu, lock):
        name = job["name"]
        metrics = self.directory / f"{name}.json"
        command = ["bash", self.entrypoint, *self.arguments]
        env = {**os.environ, "PYTHON": self.python, "SKIP_INSTALL": "1",
               "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": gpu["uuid"]}
        if self.part2:
            env.update(CASE_IDS=name.removeprefix("case"), RESULTS_DIR=str(self.directory))
        else:
            command += ["--metrics-json", str(metrics)]
        job.update(status="running", command=command, cwd=str(self.root), gpu=gpu,
                   environment={key: env[key] for key in (
                       "PYTHON", "SKIP_INSTALL", "CUDA_VISIBLE_DEVICES", "CASE_IDS", "RESULTS_DIR",
                       "CASE1_QUANTIZER", "CASE2_QUANTIZER") if key in env},
                   started_at=timestamp(), log_file=f"{name}.log", metrics_file=metrics.name)
        output = None
        try:
            output = (self.directory / job["log_file"]).open("w")
            process = subprocess.Popen(
                command, cwd=self.root, env=env, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(lock.fileno(),),
            )
        except OSError as exc:
            if output is not None:
                output.write(f"Cannot launch experiment: {exc}\n")
                output.close()
            lock.close()
            job.update(status="failed", error=str(exc), exit_code=None,
                       ended_at=timestamp(), duration_seconds=0, metrics_status="missing")
        else:
            job["pid"] = process.pid
            self.active[name] = (process, lock, output, time.monotonic())
        self.event(f'{name}: {job["status"]} on {gpu["uuid"]}')
        self.save()

    def finish(self, job):
        process, lock, output, started = self.active.pop(job["name"])
        output.close()
        lock.close()
        job.update(ended_at=timestamp(), duration_seconds=round(time.monotonic() - started, 3),
                   exit_code=process.returncode)
        try:
            job["metrics"] = json.loads((self.directory / job["metrics_file"]).read_text())
            if not isinstance(job["metrics"], dict):
                raise ValueError("metrics JSON must be an object")
            job["metrics_status"] = "present"
        except (OSError, ValueError) as exc:
            job["metrics_status"] = "missing" if isinstance(exc, FileNotFoundError) else "invalid"
            job["metrics_error"] = str(exc)
        job["status"] = ("succeeded" if process.returncode == 0
                         and job["metrics_status"] == "present" else "failed")
        self.event(f'{job["name"]}: {job["status"]}, exit code {process.returncode}')
        self.save()

    @staticmethod
    def stop(process, signum):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    def cleanup(self):
        processes = [item[0] for item in self.active.values()]
        if self.setup_process is not None:
            processes.append(self.setup_process)
        for process in processes:
            self.stop(process, signal.SIGTERM)
        deadline = time.monotonic() + 5
        for process in processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
            # Also stop descendants if their shell exited before them.
            self.stop(process, signal.SIGKILL)
            process.wait()
        self.setup_process = None
        for job in self.summary["jobs"]:
            if job["name"] in self.active:
                self.finish(job)
                job["status"] = "cancelled"
            elif job["status"] == "pending":
                job["status"] = "cancelled"

    def run(self):
        print(f"Logs: {self.directory}", flush=True)
        code = 1
        self.save()
        try:
            self.summary["initial_gpus"] = query_gpus()
            self.event("Initial GPUs: " + json.dumps(self.summary["initial_gpus"]))
            self.save()
            self.setup()
            pending = list(self.summary["jobs"])
            next_probe = 0
            while pending or self.active:
                for job in self.summary["jobs"]:
                    item = self.active.get(job["name"])
                    if item is not None and item[0].poll() is not None:
                        self.finish(job)
                        next_probe = 0
                if pending and time.monotonic() >= next_probe:
                    while pending and len(self.active) < MAX_PARALLEL:
                        busy = {job["gpu"]["uuid"] for job in self.summary["jobs"]
                                if job["name"] in self.active}
                        claimed = claim_gpu(busy)
                        if claimed is None:
                            break
                        self.launch(pending.pop(0), *claimed)
                    next_probe = time.monotonic() + self.poll_seconds
                if self.active:
                    time.sleep(0.1)
                elif pending:
                    # No event/save here: waiting must not generate log entries.
                    time.sleep(max(0, next_probe - time.monotonic()))
            code = int(any(job["status"] != "succeeded" for job in self.summary["jobs"]))
            self.summary["status"] = "failed" if code else "succeeded"
        except Interrupted as exc:
            code = 128 + exc.signum
            self.summary.update(status="interrupted", error=str(exc))
            self.event(str(exc))
        except Exception as exc:
            self.summary.update(status="failed", error=str(exc))
            self.event(f"Error: {exc}")
            print(f"Error: {exc}", file=sys.stderr)
        finally:
            # Repeated signals must not interrupt child termination and lock release.
            previous = {sig: signal.signal(sig, signal.SIG_IGN)
                        for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                if self.active or self.setup_process is not None or code:
                    self.cleanup()
                self.summary.update(ended_at=timestamp(), exit_code=code)
                self.save()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=30,
                        help="memory/lock retry interval (default: 30)")
    parser.add_argument("arguments", nargs=argparse.REMAINDER,
                        help="arguments for run_eval.py, following --")
    args = parser.parse_args()
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("--poll-seconds must be a positive finite number")
    forwarded = args.arguments
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    root = Path(__file__).resolve().parent
    reserved = ("--metrics-json", "--case") if (root / "314552017_run.sh").exists() else ("--metrics-json",)
    # Disallow argparse abbreviations too: they could override the scheduled case/path.
    for argument in forwarded:
        option = argument.split("=", 1)[0]
        if option.startswith("--") and any(flag.startswith(option) for flag in reserved):
            parser.error(f"{option} is managed by the GPU runner")

    def interrupt(signum, frame):
        raise Interrupted(signum)

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        return Runner(root, forwarded, args.poll_seconds).run()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(main())
