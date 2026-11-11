"""GPU scheduler integration tests; no torch, NVIDIA GPU, or downloads needed.

Run from the repository root: python -m unittest discover -s lab2 -p test_gpu_runner.py -v
"""

from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid


ROOT = Path(__file__).resolve().parent


def load_runner(part):
    spec = importlib.util.spec_from_file_location(f"gpu_runner_{part}", ROOT / part / "gpu_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PART2, PART3 = load_runner("part2"), load_runner("part3")

FAKE_EVAL = '''import argparse, fcntl, json, os, subprocess, sys, time
p = argparse.ArgumentParser()
p.add_argument('--case', type=int)
p.add_argument('--metrics-json', required=True)
p.add_argument('--seed', type=int, default=0)
a, _ = p.parse_known_args()
gpu = os.environ['CUDA_VISIBLE_DEVICES']
child = None
if os.environ.get('FAKE_SPAWN_CHILD') == '1':
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
def event(kind):
    with open(os.environ['FAKE_EVENTS'], 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(dict(kind=kind, case=a.case, gpu=gpu, pid=os.getpid(),
                              child_pid=child.pid if child else None, time=time.monotonic())) + '\\n')
event('start')
print('experiment stdout', flush=True)
print('experiment stderr', file=sys.stderr, flush=True)
time.sleep(float(os.environ.get('FAKE_DURATION', '0.2')))
event('end')
if str(a.case) == os.environ.get('FAKE_FAIL_CASE'):
    sys.exit(7)
if os.environ.get('FAKE_METRICS_MODE') == 'missing':
    sys.exit(0)
if os.environ.get('FAKE_METRICS_MODE') == 'invalid':
    open(a.metrics_json, 'w').write('{invalid json')
    sys.exit(0)
metrics = dict(ppl=12.5, seed=a.seed,
               storage=dict(svd=dict(compression_ratio=0.21), pruning=None),
               bit_rate=dict(weight_bits=4.5, activation_bits=5.0,
                             realized_effective_bit_rate=4.75, passes_limit=True),
               format_counts={'weight_tiles': {'4': 10}}, case3_details={'rank': 8})
with open(a.metrics_json, 'w') as f:
    json.dump(metrics, f)
'''


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.state = self.base / "gpus.json"
        self.events = self.base / "events.jsonl"
        self.gpus = [dict(index=i, uuid=f"GPU-{uuid.uuid4()}", total_mib=16000,
                          used_mib=1000, free_mib=15000) for i in range(2)]
        self.write_gpus(self.gpus)
        self.write_executable(self.bin / "nvidia-smi", '''import json, os, sys
data = json.load(open(os.environ['FAKE_GPU_DATA']))
if data == 'FAIL':
    print('GPU query unavailable', file=sys.stderr)
    sys.exit(9)
if isinstance(data, str):
    print(data)
else:
    for g in data:
        print(', '.join(str(g[k]) for k in ('index', 'uuid', 'total_mib', 'used_mib', 'free_mib')))
''')
        self.python = self.bin / "experiment_python"
        self.write_executable(self.python, '''import os, sys, time
if sys.argv[1:3] == ['-m', 'pip']:
    print('fake dependency install', flush=True)
    if os.environ.get('FAKE_SETUP_PID'):
        open(os.environ['FAKE_SETUP_PID'], 'w').write(str(os.getpid()))
        time.sleep(60)
    sys.exit(int(os.environ.get('FAKE_SETUP_EXIT', '0')))
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
''')
        env = {key: value for key, value in os.environ.items()
               if key not in ("CUDA_VISIBLE_DEVICES", "CASE_IDS", "RESULTS_DIR",
                              "SKIP_INSTALL", "FAKE_FAIL_CASE")}
        env.update(PATH=str(self.bin) + os.pathsep + env["PATH"], PYTHON=str(self.python),
                   SKIP_INSTALL="1", FAKE_GPU_DATA=str(self.state), FAKE_EVENTS=str(self.events))
        self.environment = patch.dict(os.environ, env, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        for module in (PART2, PART3):
            patcher = patch.object(module, "LOCK_ROOT", self.base / "locks")
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_executable(self, path, body):
        path.write_text(f"#!{sys.executable}\n" + body)
        path.chmod(0o755)

    def write_gpus(self, gpus):
        self.state.write_text(json.dumps(gpus))

    def make_root(self, part, name=None):
        directory = self.base / (name or part)
        directory.mkdir()
        entry = "314552017_run.sh" if part == "part2" else "run.sh"
        for filename in (entry, "gpu_run.py"):
            (directory / filename).write_text((ROOT / part / filename).read_text())
        (directory / "run_eval.py").write_text(FAKE_EVAL)
        return directory

    def run_queue(self, part="part2", arguments=None, poll=0.02):
        runner = (PART2 if part == "part2" else PART3).Runner(
            self.make_root(part), arguments or [], poll)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = runner.run()
        summary = json.loads((runner.directory / "summary.json").read_text())
        return runner, code, summary

    def read_events(self):
        return [json.loads(line) for line in self.events.read_text().splitlines()]

    def assert_parallel_limit(self, expected_peak):
        active, peak = set(), 0
        for event in self.read_events():
            if event["kind"] == "start":
                self.assertNotIn(event["gpu"], active)
                active.add(event["gpu"])
                peak = max(peak, len(active))
            else:
                active.remove(event["gpu"])
        self.assertFalse(active)
        self.assertEqual(peak, expected_peak)

    def spawn_runner(self, directory):
        # Override only the lock location to keep subprocess tests isolated.
        command = [sys.executable, "-c", (
            "import sys; from pathlib import Path; import gpu_run; "
            "gpu_run.LOCK_ROOT=Path(sys.argv.pop(1)); sys.exit(gpu_run.main())"),
            str(self.base / "locks"), "--poll-seconds", "0.02"]
        if not (directory / "314552017_run.sh").exists():
            command += ["--", "--thresholds", "thresholds.json"]
        process = subprocess.Popen(command, cwd=directory, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        def cleanup():
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=10)
        self.addCleanup(cleanup)
        return process

    def wait_for(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        self.fail("timed out waiting for fake experiment")

    def assert_dead(self, pid):
        status = Path(f"/proc/{pid}/stat")
        if status.exists():
            self.assertEqual(status.read_text().split()[2], "Z", f"process {pid} is still running")

    def test_standalone_copies_match(self):
        self.assertEqual((ROOT / "part2/gpu_run.py").read_bytes(),
                         (ROOT / "part3/gpu_run.py").read_bytes())

    def test_strict_threshold_and_shared_locks(self):
        for free in (12000, 12288):
            self.write_gpus([dict(self.gpus[0], free_mib=free)])
            self.assertIsNone(PART2.claim_gpu(set()))
        self.write_gpus([dict(self.gpus[0], free_mib=12289)])
        gpu, lock = PART2.claim_gpu(set())
        try:
            self.assertEqual(gpu["free_mib"], 12289)
            self.assertIsNone(PART3.claim_gpu(set()))
        finally:
            lock.close()
        _, lock = PART3.claim_gpu(set())
        lock.close()

    def test_recheck_after_lock_rejects_changed_memory(self):
        high = [self.gpus[0]]
        low = [dict(self.gpus[0], free_mib=12288)]
        with patch.object(PART2, "query_gpus", side_effect=[high, low]):
            self.assertIsNone(PART2.claim_gpu(set()))
        # The rejected claim released its lock.
        self.write_gpus(high)
        _, lock = PART3.claim_gpu(set())
        lock.close()

    def test_visibility_and_most_free_gpu(self):
        self.write_gpus([dict(self.gpus[0], free_mib=13000), self.gpus[1]])
        gpu, lock = PART2.claim_gpu(set())
        self.assertEqual(gpu["index"], 1)
        lock.close()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        gpu, lock = PART2.claim_gpu(set())
        self.assertEqual(gpu["index"], 0)
        lock.close()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpus[1]["uuid"][:16]
        self.assertEqual(PART3.query_gpus()[0]["index"], 1)

    def test_parallel_cases_continue_after_failure_and_preserve_metrics(self):
        os.environ["FAKE_FAIL_CASE"] = "2"
        os.environ["SKIP_INSTALL"] = "0"
        with patch.object(PART2, "query_gpus", wraps=PART2.query_gpus) as query:
            runner, code, summary = self.run_queue(arguments=["--seed", "42"])
        self.assertEqual(code, 1)
        self.assertEqual(summary["status"], "failed")
        self.assertGreaterEqual(query.call_count, 9)  # initial + two checks per launch
        self.assert_parallel_limit(2)
        self.assertEqual([job["status"] for job in summary["jobs"]],
                         ["succeeded", "failed", "succeeded", "succeeded"])
        self.assertEqual(summary["jobs"][1]["exit_code"], 7)
        self.assertEqual(summary["jobs"][1]["metrics_status"], "missing")
        self.assertEqual((runner.directory / "setup.log").read_text().count("fake dependency install"), 1)
        for job in summary["jobs"]:
            log = (runner.directory / job["log_file"]).read_text()
            self.assertIn("experiment stdout", log)
            self.assertIn("experiment stderr", log)
            self.assertGreaterEqual(job["duration_seconds"], 0)
            self.assertGreater(job["gpu"]["free_mib"], 12288)
            self.assertEqual(job["environment"]["SKIP_INSTALL"], "1")
            if job["status"] == "succeeded":
                self.assertEqual(job["metrics"]["seed"], 42)
                self.assertEqual(job["metrics"], json.loads(
                    (runner.directory / job["metrics_file"]).read_text()))

    def test_one_gpu_runs_sequentially(self):
        self.write_gpus([self.gpus[0]])
        _, code, _ = self.run_queue()
        self.assertEqual(code, 0)
        self.assert_parallel_limit(1)

    def test_waits_thirty_seconds_without_log_entries(self):
        self.write_gpus([dict(self.gpus[0], free_mib=12288)])
        runner = PART3.Runner(self.make_root("part3"), ["--thresholds", "thresholds.json"])
        real_sleep = time.sleep
        real_monotonic = time.monotonic
        offset = 0
        waits = []
        def sleep(seconds):
            nonlocal offset
            if seconds > 29:
                waits.append(seconds)
                lines = (runner.directory / "runner.log").read_text().splitlines()
                self.assertEqual(len(lines), 1)  # initial snapshot only
                self.write_gpus([self.gpus[0]])
                offset += seconds
            else:
                real_sleep(seconds)
        with patch.object(PART3.time, "sleep", side_effect=sleep), \
                patch.object(PART3.time, "monotonic", side_effect=lambda: real_monotonic() + offset), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run(), 0)
        self.assertEqual(len(waits), 1)
        self.assertLessEqual(waits[0], 30)
        self.assertEqual(len((runner.directory / "runner.log").read_text().splitlines()), 3)

    def test_part3_preserves_bit_rate_and_uses_original_entry(self):
        runner, code, summary = self.run_queue("part3", ["--thresholds", "thresholds.json"])
        self.assertEqual(code, 0)
        job = summary["jobs"][0]
        self.assertEqual(job["command"][:2], ["bash", "run.sh"])
        self.assertEqual(job["metrics"]["bit_rate"]["weight_bits"], 4.5)
        self.assertTrue(job["metrics"]["bit_rate"]["passes_limit"])
        self.assertEqual(job["metrics"]["case3_details"], {"rank": 8})
        self.assertTrue((runner.directory / "experiment.json").exists())

    def test_two_runner_processes_share_gpu_locks(self):
        first = self.spawn_runner(self.make_root("part2"))
        second = self.spawn_runner(self.make_root("part3"))
        for process in (first, second):
            stdout, stderr = process.communicate(timeout=15)
            self.assertEqual(process.returncode, 0, (stdout, stderr))
        self.assertEqual(len(self.read_events()), 10)
        self.assert_parallel_limit(2)

    def test_sigterm_stops_experiments_and_descendants_then_releases_locks(self):
        os.environ.update(FAKE_DURATION="60", FAKE_SPAWN_CHILD="1")
        directory = self.make_root("part2")
        process = self.spawn_runner(directory)
        self.wait_for(lambda: self.events.exists() and len(self.read_events()) == 2)
        children = self.read_events()
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 143, (stdout, stderr))
        summary = json.loads(next(directory.glob("results/gpu_runs/*/summary.json")).read_text())
        self.assertEqual(summary["status"], "interrupted")
        self.assertTrue(all(job["status"] == "cancelled" for job in summary["jobs"]))
        for event in children:
            self.assert_dead(event["pid"])
            self.assert_dead(event["child_pid"])
        for gpu in self.gpus:
            with patch.object(PART2, "query_gpus", return_value=[gpu]):
                _, lock = PART2.claim_gpu(set())
                lock.close()

    def test_sigterm_during_setup_stops_installer(self):
        marker = self.base / "setup.pid"
        os.environ.update(SKIP_INSTALL="0", FAKE_SETUP_PID=str(marker))
        directory = self.make_root("part3")
        process = self.spawn_runner(directory)
        self.wait_for(marker.exists)
        process.terminate()
        stdout, stderr = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 143, (stdout, stderr))
        self.assert_dead(int(marker.read_text()))
        self.assertFalse(self.events.exists())

    def test_query_errors_and_no_gpus_are_logged(self):
        for index, data in enumerate(([], "0, GPU-bad, N/A, 0, 0", "bad csv", "FAIL")):
            with self.subTest(data=data):
                self.write_gpus(data)
                runner = PART3.Runner(self.make_root("part3", f"query{index}"), [])
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(runner.run(), 1)
                summary = json.loads((runner.directory / "summary.json").read_text())
                self.assertEqual(summary["status"], "failed")
                self.assertIn("error", summary)
                self.assertIn("Error:", (runner.directory / "runner.log").read_text())
                self.assertFalse(self.events.exists())

    def test_missing_and_invalid_metrics_fail_even_with_zero_exit_code(self):
        for mode in ("missing", "invalid"):
            with self.subTest(mode=mode):
                os.environ["FAKE_METRICS_MODE"] = mode
                runner = PART3.Runner(self.make_root("part3", mode), [], 0.02)
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.run(), 1)
                job = json.loads((runner.directory / "summary.json").read_text())["jobs"][0]
                self.assertEqual(job["exit_code"], 0)
                self.assertEqual(job["metrics_status"], mode)
                self.assertEqual(job["status"], "failed")

    def test_query_failure_during_execution_cleans_up_running_jobs(self):
        self.write_gpus([self.gpus[0]])
        os.environ["FAKE_DURATION"] = "60"
        directory = self.make_root("part2")
        process = self.spawn_runner(directory)
        self.wait_for(lambda: self.events.exists() and len(self.read_events()) == 1)
        pid = self.read_events()[0]["pid"]
        self.write_gpus("FAIL")
        stdout, stderr = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 1, (stdout, stderr))
        summary = json.loads(next(directory.glob("results/gpu_runs/*/summary.json")).read_text())
        self.assertIn("GPU query unavailable", summary["error"])
        self.assertTrue(all(job["status"] == "cancelled" for job in summary["jobs"]))
        self.assert_dead(pid)

    def test_sigint_interrupts_idle_memory_wait(self):
        self.write_gpus([dict(self.gpus[0], free_mib=12288)])
        directory = self.make_root("part3")
        process = self.spawn_runner(directory)
        self.wait_for(lambda: bool(list(directory.glob("results/gpu_runs/*/setup.log"))))
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 130, (stdout, stderr))
        summary = json.loads(next(directory.glob("results/gpu_runs/*/summary.json")).read_text())
        self.assertEqual(summary["status"], "interrupted")
        self.assertFalse(self.events.exists())

    def test_install_failure_stops_before_dispatch(self):
        os.environ.update(SKIP_INSTALL="0", FAKE_SETUP_EXIT="3")
        runner, code, summary = self.run_queue("part3")
        self.assertEqual(code, 1)
        self.assertEqual(summary["setup_exit_code"], 3)
        self.assertTrue((runner.directory / "setup.log").exists())
        self.assertFalse(self.events.exists())

    def test_original_shell_defaults_case_selection_and_paths(self):
        directory = self.make_root("part2")
        fake_python = self.bin / "print_args"
        fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$*"\n')
        fake_python.chmod(0o755)
        env = {**os.environ, "PYTHON": str(fake_python)}
        result = subprocess.run(["bash", "314552017_run.sh"], cwd=directory,
                                env=env, text=True, capture_output=True, check=True)
        self.assertEqual(len(result.stdout.splitlines()), 4)
        self.assertEqual(result.stderr, "")
        for case, line in enumerate(result.stdout.splitlines(), 1):
            self.assertIn(f"--case {case} ", line)
            self.assertIn(f"--metrics-json results/case{case}.json", line)
            self.assertIn("--submission-output", line)
        env.update(CASE_IDS="3", RESULTS_DIR=str(self.base / "custom results"))
        result = subprocess.run(["bash", "314552017_run.sh", "--seed", "42"],
                                cwd=directory, env=env, text=True, capture_output=True, check=True)
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertIn("--case 3 ", result.stdout)
        self.assertIn("custom results/case3.json", result.stdout)
        self.assertIn("--seed 42", result.stdout)
        for invalid in ("", "0", "5", "1 nope"):
            env["CASE_IDS"] = invalid
            result = subprocess.run(["bash", "314552017_run.sh"], cwd=directory,
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
