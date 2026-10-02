"""Run the extracted pipecat smart-turn v3 baseline over a manifest.

Usage:
    python scripts/run_smart_turn_baseline.py --manifest data/test.jsonl
    python scripts/run_smart_turn_baseline.py --manifest data/test.jsonl --limit 200

Computes overall and per-language accuracy, false-interruption rate
(model says "complete" while the user was still mid-turn) and false-hold
rate (model says "incomplete" although the turn had ended).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from turn_detector.audio import load_audio
from turn_detector.baselines.smart_turn import SmartTurnBaseline
from turn_detector.data.records import AudioRecord, read_manifest


def evaluate(
    records: list[AudioRecord],
    model: SmartTurnBaseline,
    manifest_path: str | Path,
) -> dict:
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0}
    )

    for record in records:
        waveform, sample_rate = load_audio(record.resolved_audio_path(manifest_path))
        if sample_rate != 16_000:
            from turn_detector.audio import resample_audio

            waveform = resample_audio(waveform, sample_rate, 16_000)

        result = model.predict_endpoint(waveform)
        predicted = int(result["prediction"])
        actual = record.label  # 1 = endpoint (turn ended), 0 = hold

        key = record.language
        bucket = counts[key]
        bucket["n"] += 1
        if predicted == 1 and actual == 1:
            bucket["tp"] += 1
        elif predicted == 0 and actual == 0:
            bucket["tn"] += 1
        elif predicted == 1 and actual == 0:
            bucket["fp"] += 1  # false interruption: cut user off mid-turn
        else:
            bucket["fn"] += 1  # false hold: model failed to detect the endpoint

    totals = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0}
    for bucket in counts.values():
        for key in totals:
            totals[key] += bucket[key]

    report: dict = {"overall": _slice_metrics("overall", totals), "slices": []}
    for language in sorted(counts):
        report["slices"].append(_slice_metrics(language, counts[language]))
    return report


def _slice_metrics(name: str, bucket: dict[str, int]) -> dict:
    n = max(bucket["n"], 1)
    accuracy = (bucket["tp"] + bucket["tn"]) / n
    false_interruption = bucket["fp"] / max(bucket["fp"] + bucket["tn"], 1)
    false_hold = bucket["fn"] / max(bucket["fn"] + bucket["tp"], 1)
    return {
        "slice": name,
        "n": bucket["n"],
        "accuracy": round(accuracy, 4),
        "false_interruption_rate": round(false_interruption, 4),
        "false_hold_rate": round(false_hold, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a .jsonl (or .parquet) AudioRecord manifest",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N records")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local ONNX path; defaults to downloading pipecat-ai/smart-turn-v3 from HF",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON report",
    )
    args = parser.parse_args()

    records = list(read_manifest(args.manifest))
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        print(f"No records found in {args.manifest}", file=sys.stderr)
        return 1

    model = SmartTurnBaseline(model_path=args.model_path)
    report = evaluate(records, model, args.manifest)

    print(f"Smart-turn v3.2 CPU baseline over {args.manifest} ({len(records)} clips)")
    for metrics in [report["overall"], *report["slices"]]:
        print(
            f"  {metrics['slice']:<10} n={metrics['n']:<6} "
            f"acc={metrics['accuracy']:.4f}  "
            f"false_interruption={metrics['false_interruption_rate']:.4f}  "
            f"false_hold={metrics['false_hold_rate']:.4f}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
