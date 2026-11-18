import importlib.util
import json
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = module("run_matrix")
summary = module("summarize")


class BenchmarkEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "bench" / "matrix.json").read_text())

    def test_plan_controls_lengths_rate_and_repeat_seed(self):
        plan = runner.build_plan(self.config, "http://localhost:8000", "vllm", Path("raw"))
        self.assertEqual(len(plan), 36)
        self.assertEqual(len({p["id"] for p in plan}), 36)
        for item in plan:
            cmd = item["command"]
            self.assertNotIn("--max-concurrency", cmd)
            self.assertIn("--save-detailed", cmd)
            self.assertEqual(cmd[cmd.index("--random-range-ratio") + 1], "0.0")
            self.assertEqual(cmd[cmd.index("--seed") + 1], str(41 + item["repetition"]))

    def test_rejects_invalid_experiments(self):
        for value in (0, -1, float("nan"), float("inf"), True):
            self.config["rates"] = [value]
            with self.assertRaises(ValueError):
                runner.build_plan(self.config, "http://localhost", "vllm", Path("raw"))
        self.config["rates"] = [1]
        self.config["workloads"][0]["input_tokens"] = 4096
        with self.assertRaises(ValueError):
            runner.build_plan(self.config, "http://localhost", "vllm", Path("raw"))

    def test_placeholder_environment_is_not_evidence(self):
        env = json.loads((ROOT / "bench" / "environment.example.json").read_text())
        with self.assertRaises(ValueError):
            runner.validate_environment(env)

    def test_026_rejects_one_as_fixed_length_ratio(self):
        self.config["random_range_ratio"] = 1
        with self.assertRaises(ValueError):
            runner.build_plan(self.config, "http://localhost", "vllm", Path("raw"))

    def test_failure_without_result_is_not_dropped_from_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "command.json").write_text(json.dumps({"workload": {"name": "smoke"}, "rate": 1, "repetition": 1}))
            (root / "status.json").write_text('{"state":"failed"}')
            row = summary.collect(root)[0]
            self.assertEqual(row["status"], "failed")
            self.assertFalse(row["result_available"])
            self.assertEqual(row["completed"], "")

    def test_missing_metrics_remain_missing_and_failures_remain_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "result.json").write_text(json.dumps({"completed": 7, "duration": 10}))
            (p / "command.json").write_text(json.dumps({"workload": {"name": "smoke"}, "rate": 1, "repetition": 1}))
            (p / "status.json").write_text('{"state":"failed"}')
            row = summary.collect(p)[0]
            self.assertEqual(row["completed"], 7)
            self.assertEqual(row["p99_ttft_ms"], "")
            self.assertEqual(row["status"], "failed")

    def test_empty_results_cannot_create_fake_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                summary.collect(Path(tmp))

    def test_compact_help_expands_before_rejecting_real_flags(self):
        replies = [subprocess.CompletedProcess([], 0, "use --help=all"),
                   subprocess.CompletedProcess([], 0, "--endpoint --save-detailed")]
        with patch.object(runner.subprocess, "run", side_effect=replies) as mocked:
            text = runner.cli_help("vllm", {"--endpoint", "--save-detailed"})
        self.assertIn("--endpoint", text)
        self.assertEqual(mocked.call_args_list[1].args[0][-1], "--help=all")

    def test_unsupported_cli_flag_is_rejected(self):
        reply = subprocess.CompletedProcess([], 0, "no matching options")
        with patch.object(runner.subprocess, "run", return_value=reply):
            with self.assertRaises(ValueError):
                runner.cli_help("vllm", {"--unsupported"})

    def test_collector_includes_completion_snapshot(self):
        class StopAfterFirstInterval:
            stopped = False

            def is_set(self):
                return self.stopped

            def wait(self, seconds):
                self.stopped = True

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            with patch.object(runner, "fetch", side_effect=["counter 0", "counter 8"]):
                runner.capture_metrics("http://localhost", path, StopAfterFirstInterval())
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["prometheus_text"] for row in rows], ["counter 0", "counter 8"])

    def exercise_execution(self, failure):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = json.loads((ROOT / "bench" / "environment.example.json").read_text())
            env = {key: "test fixture only" for key in env}
            env_path = root / "environment.json"
            env_path.write_text(json.dumps(env))
            plan = runner.build_plan(self.config, "http://localhost", "vllm", root)
            help_text = " ".join(plan[0]["command"]) + " --tokenizer"

            def fake_run(cmd, **kwargs):
                if cmd[-1] == "--version":
                    return subprocess.CompletedProcess(cmd, 0, "0.26.0", "")
                if cmd[-1] == "--help":
                    return subprocess.CompletedProcess(cmd, 0, help_text)
                if failure == "timeout":
                    raise subprocess.TimeoutExpired(cmd, 1)
                folder = Path(cmd[cmd.index("--result-dir") + 1])
                (folder / "result.json").write_text(json.dumps({
                    "completed": 7 if failure == "partial" else 8, "duration": 1}))
                return subprocess.CompletedProcess(cmd, 0)

            argv = ["run_matrix.py", "--config", str(ROOT / "bench" / "smoke.json"),
                    "--environment", str(env_path), "--output", str(root / "out"),
                    "--tokenizer", str(root), "--execute"]
            with patch.object(sys, "argv", argv), patch.object(runner.subprocess, "run", side_effect=fake_run), \
                 patch.object(runner, "fetch", return_value="{}"), patch.object(runner, "capture_metrics"), \
                 patch.object(runner.time, "sleep"):
                if failure:
                    with self.assertRaises((RuntimeError, subprocess.TimeoutExpired)):
                        runner.main()
                else:
                    runner.main()
            statuses = list((root / "out").rglob("status.json"))
            self.assertEqual(len(statuses), 1)
            status = json.loads(statuses[0].read_text())
            self.assertEqual(status["state"], "failed" if failure else "completed")
            if failure:
                self.assertIn("error", status)
            self.assertTrue(statuses[0].with_name("command.json").exists())

    def test_partial_http_success_stops_and_preserves_failure(self):
        self.exercise_execution("partial")

    def test_timeout_preserves_failure(self):
        self.exercise_execution("timeout")

    def test_success_writes_complete_evidence_status(self):
        self.exercise_execution(None)


if __name__ == "__main__":
    unittest.main()
