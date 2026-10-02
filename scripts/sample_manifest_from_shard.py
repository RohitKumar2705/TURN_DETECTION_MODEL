"""Build a stratified AudioRecord manifest from a smart-turn v3.2 test shard.

Usage:
    python scripts/sample_manifest_from_shard.py \
        --shard <path/to/train-00000-of-00010.parquet> \
        --manifest data/test_sample.jsonl \
        --per-bucket 132

Samples up to ``--per-bucket`` clips per (language, endpoint_bool) bucket,
writes wav files next to the manifest, and emits an AudioRecord JSONL
compatible with scripts/run_smart_turn_baseline.py.
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from turn_detector.audio import (
    analyze_quality,
    load_audio,
    resample_audio,
    save_audio,
)
from turn_detector.data.duplicates import (
    acoustic_fingerprint,
    duplicate_group,
    waveform_hash,
)
from turn_detector.data.records import AudioRecord, write_manifest

AUDIO_COLUMNS = ["audio", "id", "language", "endpoint_bool", "synthetic", "dataset"]


def decode_audio(audio_struct: dict) -> tuple[bytes, str]:
    payload = audio_struct.get("bytes")
    if payload is None:
        raise ValueError("audio struct has no bytes")
    return bytes(payload), str(audio_struct.get("path") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per-bucket", type=int, default=132)
    parser.add_argument("--languages", nargs="*", default=["hin", "eng"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    table = pq.ParquetFile(args.shard).read(columns=AUDIO_COLUMNS)
    rows = table.to_pylist()

    wanted = set(args.languages)
    buckets: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for row in rows:
        language = row["language"]
        if language in wanted:
            buckets[(language, bool(row["endpoint_bool"]))].append(row)

    sampled: list[dict] = []
    for key in sorted(buckets, key=str):
        pool = buckets[key]
        random.shuffle(pool)
        sampled.extend(pool[: args.per_bucket])

    audio_dir = args.manifest.parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    records: list[AudioRecord] = []
    for row in sampled:
        payload, suffix = decode_audio(row["audio"])
        # libsndfile on Windows fails to read FLAC from BytesIO, so stage to a
        # temporary file and load from disk instead.
        descriptor, temp_name = tempfile.mkstemp(suffix=suffix or ".flac")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
            waveform, sample_rate = load_audio(temp_name)
        finally:
            os.unlink(temp_name)
        if sample_rate != 16_000:
            waveform = resample_audio(waveform, sample_rate, 16_000)
            sample_rate = 16_000

        quality = analyze_quality(waveform, sample_rate)
        audio_hash = waveform_hash(waveform)
        fingerprint = acoustic_fingerprint(waveform, sample_rate)

        clip_id = str(row["id"])
        audio_path = audio_dir / f"{clip_id}.wav"
        save_audio(audio_path, waveform, sample_rate)
        # Store paths relative to the manifest so resolved_audio_path() works.
        stored_audio_path = os.path.relpath(audio_path, args.manifest.parent)

        records.append(
            AudioRecord(
                id=clip_id,
                parent_id=clip_id,
                audio_path=stored_audio_path,
                source_repo="pipecat-ai/smart-turn-data-v3.2-test",
                source_dataset=str(row["dataset"] or "unknown"),
                language=language_map(row["language"]),
                endpoint_bool=bool(row["endpoint_bool"]),
                synthetic=bool(row["synthetic"]) if row["synthetic"] is not None else None,
                split="test",
                duration_seconds=quality.duration_seconds,
                valid_samples=int(waveform.size),
                speech_seconds=quality.speech_seconds,
                speech_ratio=quality.speech_seconds / max(quality.duration_seconds, 1e-6),
                peak_dbfs=quality.peak_dbfs,
                rms_dbfs=quality.rms_dbfs,
                clipping_ratio=quality.clipping_ratio,
                silence_ratio=quality.silence_ratio,
                audio_hash=audio_hash,
                acoustic_fingerprint=fingerprint,
                duplicate_group=duplicate_group(audio_hash, fingerprint),
                quality_status=quality.reason or "ok",
            )
        )

    write_manifest(records, args.manifest)
    counts: dict[tuple[str, bool], int] = defaultdict(int)
    for record in records:
        counts[(record.language, record.endpoint_bool)] += 1
    for key in sorted(counts, key=str):
        print(f"  {key[0]:<4} endpoint={str(key[1]):<5} -> {counts[key]}")
    print(f"Wrote {len(records)} records to {args.manifest}")
    return 0


def language_map(code: str) -> str:
    """Map dataset language codes to the project's hin/eng vocabulary."""
    mapping = {"hin": "hin", "eng": "eng", "hin_de": "hin", "hi": "hin", "en": "eng"}
    if code not in mapping:
        raise ValueError(f"Unexpected language code {code!r}; extend language_map()")
    return mapping[code]


if __name__ == "__main__":
    raise SystemExit(main())
