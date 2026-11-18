"""Run the installed vLLM benchmark and retain evidence; stdlib orchestrator.

Default is plan-only. This wrapper does not deploy or tune the server.
The vLLM CLI and matching tokenizer dependencies must exist in the client env.
"""
import argparse
import itertools
import json
import math
import re
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def positive(value, label, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(value) or value <= 0 or (integer and int(value) != value):
        raise ValueError(f"invalid {label}: {value}")


def build_plan(config, base_url, binary, output):
    for key in ("max_model_len", "num_prompts", "repetitions"):
        positive(config[key], key, integer=True)
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ValueError("seed must be an integer")
    if not isinstance(config["model"], str) or not config["model"].strip():
        raise ValueError("model must be a nonempty string")
    if not config["rates"] or not config["workloads"]:
        raise ValueError("rates and workloads must be nonempty")
    names = set()
    for w in config["workloads"]:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", w["name"]) or w["name"] in names:
            raise ValueError("workload names must be unique safe path components")
        names.add(w["name"])
        for k in ("input_tokens", "output_tokens"):
            positive(w[k], k, integer=True)
        if w["input_tokens"] + w["output_tokens"] > config["max_model_len"]:
            raise ValueError(f"context limit exceeded: {w['name']}")
    for rate in config["rates"]:
        positive(rate, "request rate")
    plan = []
    for w, rate, rep in itertools.product(
        config["workloads"], config["rates"], range(config["repetitions"])
    ):
        run_id = f"{w['name']}-r{rate}-rep{rep + 1}"
        folder = output / run_id
        cmd = [binary, "bench", "serve", "--backend", "vllm",
               "--base-url", base_url, "--endpoint", "/v1/completions",
               "--model", config["model"], "--dataset-name", "random",
               "--random-input-len", str(w["input_tokens"]),
               "--random-output-len", str(w["output_tokens"]),
               "--random-range-ratio", "1.0", "--num-prompts", str(config["num_prompts"]),
               "--request-rate", str(rate), "--seed", str(config["seed"] + rep),
               "--ignore-eos", "--save-result", "--save-detailed",
               "--result-dir", str(folder), "--result-filename", "result.json"]
        plan.append({"id": run_id, "workload": w, "rate": rate,
                     "repetition": rep + 1, "command": cmd})
    if len({x["id"] for x in plan}) != len(plan):
        raise ValueError("duplicate rates create duplicate runs")
    return plan


def fetch(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8")


def capture_metrics(url, target, stop):
    start = time.monotonic()
    with target.open("w", encoding="utf-8") as f:
        while True:
            row = {"utc": datetime.now(timezone.utc).isoformat(),
                   "elapsed_s": time.monotonic() - start}
            try:
                row["prometheus_text"] = fetch(url + "/metrics")
            except Exception as exc:
                row["scrape_error"] = str(exc)  # Missing is never zero.
            f.write(json.dumps(row) + "\n")
            f.flush()
            if stop.wait(1):
                break


def validate_environment(env):
    required = ("gpu", "driver", "runtime", "server_image_digest", "server_vllm_version",
                "model_revision", "server_args", "prefix_cache", "client_path")
    for key in required:
        value = env.get(key)
        if not isinstance(value, str) or not value.strip() or "REPLACE" in value:
            raise ValueError(f"record actual environment field: {key}")


def cli_help(binary, flags):
    result = subprocess.run([binary, "bench", "serve", "--help"],
                            text=True, capture_output=True, check=True, timeout=60)
    # Recent vLLM versions show a compact argument-group index by default.
    if any(flag not in result.stdout for flag in flags):
        result = subprocess.run([binary, "bench", "serve", "--help=all"],
                                text=True, capture_output=True, check=True, timeout=60)
    for flag in flags:
        if flag not in result.stdout:
            raise ValueError(f"installed benchmark CLI does not advertise {flag}; adapt to its version")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("smoke.json"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--tokenizer", help="Local tokenizer snapshot matching the server revision")
    parser.add_argument("--output", type=Path, default=Path("bench/raw"))
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--cooldown", type=float, default=15)
    args = parser.parse_args()
    positive(args.timeout, "timeout")
    positive(args.cooldown, "cooldown")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    batch = args.output.resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    url = args.base_url.rstrip("/")
    plan = build_plan(config, url, args.vllm_bin, batch)
    if args.tokenizer:
        for item in plan:
            item["command"].extend(["--tokenizer", args.tokenizer])
    if not args.execute:
        print(json.dumps({"mode": "plan_only", "runs": plan}, indent=2))
        return
    if args.environment is None:
        parser.error("--execute requires --environment with actual server metadata")
    env = json.loads(args.environment.read_text(encoding="utf-8"))
    validate_environment(env)
    if not args.tokenizer:
        parser.error("--execute requires --tokenizer pointing to the matching local snapshot")
    if not Path(args.tokenizer).is_dir():
        parser.error("--tokenizer must be an existing local snapshot directory")
    version_run = subprocess.run([args.vllm_bin, "--version"], text=True,
                                 capture_output=True, check=True, timeout=60)
    help_text = cli_help(args.vllm_bin, {x for item in plan for x in item["command"] if x.startswith("--")})
    fetch(url + "/health")
    models = fetch(url + "/v1/models")
    batch.mkdir(parents=True, exist_ok=False)
    (batch / "benchmark-help.txt").write_text(help_text, encoding="utf-8")
    (batch / "client-version.txt").write_text(version_run.stdout + version_run.stderr, encoding="utf-8")
    (batch / "models.json").write_text(models, encoding="utf-8")
    (batch / "manifest.json").write_text(json.dumps({"config": config, "environment": env,
        "plan": plan, "started_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    for item in plan:
        folder = batch / item["id"]
        folder.mkdir()
        (folder / "command.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
        stop = threading.Event()
        collector = threading.Thread(target=capture_metrics, args=(url, folder / "metrics.jsonl", stop))
        collector.start()
        start = time.monotonic()
        status = {"state": "failed"}
        try:
            with (folder / "client.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(item["command"], stdout=log, stderr=subprocess.STDOUT,
                                        timeout=args.timeout, check=False)
            status["returncode"] = result.returncode
            if result.returncode != 0:
                raise RuntimeError(f"benchmark failed: {folder / 'client.log'}")
            raw = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            status["completed"] = raw.get("completed")
            # A zero exit code does not mean all HTTP requests succeeded.
            if raw.get("completed") != config["num_prompts"]:
                raise RuntimeError("incomplete/unknown completed count; preserve evidence and stop sweep")
            status["state"] = "completed"
        except BaseException as exc:
            status["error"] = str(exc)
            raise
        finally:
            stop.set()
            collector.join(timeout=7)
            status["wall_seconds"] = time.monotonic() - start
            (folder / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"completed {item['id']}: {folder}", flush=True)
        time.sleep(args.cooldown)
    print(f"Evidence saved in {batch}")


if __name__ == "__main__":
    main()
