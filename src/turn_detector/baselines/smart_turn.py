"""Baseline turn detector: pipecat-ai smart-turn v3 (extracted and adapted).

Source: https://github.com/pipecat-ai/smart-turn (BSD-2-Clause)
Weights: https://huggingface.co/pipecat-ai/smart-turn-v3

This module is a self-contained adaptation of the upstream ``inference.py`` +
``audio_utils.py``. It exposes :func:`predict_endpoint`, which takes 16 kHz mono
float32 PCM and returns ``{"prediction": 0|1, "probability": float}`` exactly
like upstream, plus a manifest-friendly batch API for the evaluation harness.

Input contract (matches upstream):
- 16 kHz mono float32 in [-1, 1].
- Audio shorter than 8 s is zero-padded at the *front*; longer audio keeps the
  *last* 8 s (the end of the turn is what matters for endpointing).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DEFAULT_MODEL_REPO = "pipecat-ai/smart-turn-v3"
DEFAULT_MODEL_FILE = "smart-turn-v3.2-cpu.onnx"

_ONNX_INPUT_NAME = "input_features"


def truncate_audio_to_last_n_seconds(
    audio_array: np.ndarray,
    n_seconds: float = 8.0,
    sample_rate: int = 16_000,
) -> np.ndarray:
    """Truncate audio to its last ``n_seconds`` or front-pad with zeros to that length.

    Identical to upstream smart-turn ``audio_utils.truncate_audio_to_last_n_seconds``.
    """
    max_samples = int(n_seconds * sample_rate)
    if audio_array.size > max_samples:
        return audio_array[-max_samples:]
    if audio_array.size < max_samples:
        padding = max_samples - audio_array.size
        return np.pad(audio_array, (padding, 0), mode="constant", constant_values=0)
    return audio_array


class SmartTurnBaseline:
    """ONNX wrapper around pipecat smart-turn v3 with lazy session loading."""

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        repo_id: str = DEFAULT_MODEL_REPO,
        filename: str = DEFAULT_MODEL_FILE,
        intra_op_num_threads: int | None = None,
    ) -> None:
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        self._ort = ort
        self._repo_id = repo_id
        self._filename = filename
        self._model_path: Path | None = Path(model_path) if model_path else None
        self._session = None
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=8)
        self._intra_op_num_threads = intra_op_num_threads

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _resolve_model_path(self) -> Path:
        if self._model_path is not None and self._model_path.is_file():
            return self._model_path
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(repo_id=self._repo_id, filename=self._filename)
        self._model_path = Path(downloaded)
        return self._model_path

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        session_options = self._ort.SessionOptions()
        session_options.execution_mode = self._ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.inter_op_num_threads = 1
        if self._intra_op_num_threads is not None:
            session_options.intra_op_num_threads = self._intra_op_num_threads
        session_options.graph_optimization_level = (
            self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self._session = self._ort.InferenceSession(
            str(self._resolve_model_path()), sess_options=session_options
        )
        return self._session

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict_endpoint(self, audio_array: np.ndarray) -> dict[str, int | float]:
        """Predict whether a single utterance ended (1) or continues (0).

        ``audio_array`` must be mono float32 PCM at 16 kHz. Upstream-compatible
        result: ``{"prediction": 0|1, "probability": float}``.
        """
        waveform = np.asarray(audio_array, dtype=np.float32).reshape(-1)
        audio_array = truncate_audio_to_last_n_seconds(waveform, n_seconds=8.0)

        inputs = self._feature_extractor(
            audio_array,
            sampling_rate=16_000,
            return_tensors="np",
            padding="max_length",
            max_length=8 * 16_000,
            truncation=True,
            do_normalize=True,
        )
        input_features = np.asarray(inputs.input_features, dtype=np.float32).squeeze(0)
        input_features = np.expand_dims(input_features, axis=0)

        session = self._ensure_session()
        outputs = session.run(None, {_ONNX_INPUT_NAME: input_features})
        probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        prediction = 1 if probability > 0.5 else 0
        return {"prediction": prediction, "probability": probability}

    def predict_batch(
        self, audio_arrays: list[np.ndarray]
    ) -> list[dict[str, int | float]]:
        """Sequential inference over a list of utterances (ONNX session reuse)."""
        return [self.predict_endpoint(audio) for audio in audio_arrays]


def predict_endpoint(audio_array: np.ndarray) -> dict[str, int | float]:
    """Convenience upstream-style function using a module-level lazy session."""
    global _DEFAULT_INSTANCE
    if "_DEFAULT_INSTANCE" not in globals() or _DEFAULT_INSTANCE is None:
        _DEFAULT_INSTANCE = SmartTurnBaseline()
    return _DEFAULT_INSTANCE.predict_endpoint(audio_array)


_DEFAULT_INSTANCE: SmartTurnBaseline | None = None
