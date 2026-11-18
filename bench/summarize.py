"""Export official benchmark aggregates without inventing missing metrics."""
import argparse
import csv
import json
from pathlib import Path

FIELDS = ["completed", "duration", "request_throughput", "output_throughput",
          "total_output_tokens", "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
          "mean_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "p99_itl_ms"]


def collect(root):
    rows = []
    for path in root.rglob("result.json"):
        if not path.with_name("command.json").exists():
            raise ValueError(f"missing run provenance: {path}")
    for command_path in sorted(root.rglob("command.json")):
        path = command_path.with_name("result.json")
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if path.exists() and (not isinstance(data.get("completed"), int) or "duration" not in data):
            raise ValueError(f"unrecognized benchmark result schema: {path}")
        run = json.loads(command_path.read_text(encoding="utf-8"))
        row = {"run": str(path.parent), "workload": run["workload"]["name"],
               "offered_rate": run["rate"], "repetition": run["repetition"]}
        row.update({name: data.get(name, "") for name in FIELDS})
        status_path = path.with_name("status.json")
        row["status"] = json.loads(status_path.read_text(encoding="utf-8")).get("state", "unknown") if status_path.exists() else "unknown"
        row["result_available"] = path.exists()
        rows.append(row)
    if not rows:
        raise ValueError("no run provenance found; execute a benchmark first")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.root)
    with args.output.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} runs; empty cells mean unavailable, not zero.")


if __name__ == "__main__":
    main()
