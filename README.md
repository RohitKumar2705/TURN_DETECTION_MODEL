# Hinglish Turn Detector

Audio-native end-of-turn detection for Hindi, Hinglish, and Indian English.

This project answers a practical question for voice applications:

> Has the speaker finished their turn, or are they likely to continue?

It accepts microphone input or an audio file and returns a completion
probability, a `complete` or `hold` decision, and a recommended wait time.
The runtime model is exported to ONNX and runs with ONNX Runtime.

## How it works

```text
Audio
  -> mono conversion and resampling to 16 kHz
  -> Whisper Tiny audio features
  -> dual-scale turn classifier
  -> calibrated completion probability
  -> COMPLETE or HOLD decision
```

The model is designed for conversational systems that need to know when to
stop waiting for more speech and start responding.

## Requirements

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/)
- Internet access for the default Hugging Face model and datasets

The project requires Python `>=3.11,<3.13`.

## Installation

From the repository root:

```powershell
cd D:\TURN_DETECTION_MODEL
uv sync
```

For the Gradio demo and ONNX runtime:

```powershell
uv sync --extra demo --extra runtime --extra export
```

For training:

```powershell
uv sync --extra train
```

For most local workflows:

```powershell
uv sync `
  --extra data `
  --extra demo `
  --extra eval `
  --extra export `
  --extra runtime `
  --extra train
```

Always use `uv run` so commands use the project environment.

## Run the Gradio application

The model is selected with `HINGLISH_TURN_MODEL` in [`.env`](.env). The
default repository is:

```env
HINGLISH_TURN_MODEL=Mayank022/hinglish-turn-detector-whisper-tiny-dual-scale
```

Start the application:

```powershell
uv run python app.py
```

Open <http://127.0.0.1:7860> in a browser.

To use a local exported model:

```powershell
$env:HINGLISH_TURN_MODEL = "artifacts\export\hinglish-turn.onnx"
uv run python app.py
```

## Command-line interface

Show all commands:

```powershell
uv run turn-detector --help
```

Validate the default configuration:

```powershell
uv run turn-detector validate-config
```

Validate the smoke configuration:

```powershell
uv run turn-detector validate-config --config configs\smoke.yaml
```

The CLI can also be run as a module:

```powershell
uv run python -m turn_detector.cli --help
```

## Data preparation

Prepare train, validation, and test manifests:

```powershell
uv run turn-detector data prepare --config configs\default.yaml
```

Run a small bounded preparation:

```powershell
uv run turn-detector data prepare `
  --config configs\smoke.yaml `
  --limit 64
```

Manifests are written below the configured output directory, normally:

```text
artifacts/data/train.jsonl
artifacts/data/validation.jsonl
artifacts/data/test.jsonl
```

## Training

Install the training dependencies:

```powershell
uv sync --extra train
```

Start training:

```powershell
uv run turn-detector train --config configs\default.yaml
```

Training checkpoints and reports are written to:

```text
artifacts/checkpoints/
```

For a small pipeline check:

```powershell
uv run turn-detector train --config configs\smoke.yaml
```

The smoke configuration is intentionally small and is not intended to produce
a final-quality model.

## Prediction, evaluation, and export

Predict from an audio file using a local ONNX model:

```powershell
uv run turn-detector predict `
  --model-path artifacts\export\hinglish-turn.onnx `
  data\audio\example.wav
```

Evaluate a model:

```powershell
uv run turn-detector evaluate `
  --model-path artifacts\export\hinglish-turn.onnx `
  --config configs\default.yaml
```

Export a training checkpoint:

```powershell
uv sync --extra export --extra train
uv run turn-detector export `
  --checkpoint artifacts\checkpoints\best `
  --output artifacts\export\hinglish-turn.onnx `
  --config configs\default.yaml
```

Calibrate an exported model:

```powershell
uv run turn-detector calibrate `
  --model-path artifacts\export\hinglish-turn.onnx `
  --config configs\default.yaml
```

Example prediction output:

```json
{
  "probability": 0.86,
  "decision": "complete",
  "recommended_wait_ms": 200,
  "inference_ms": 12.4
}
```

## Configuration

The main configuration is [`configs/default.yaml`](configs/default.yaml).
Important policy settings include:

```yaml
policy:
  threshold: 0.80
  min_silence_ms: 200
  timeout_ms: 1600
  speech_rms_threshold: 0.015
```

- `threshold`: probability required for a `complete` decision.
- `min_silence_ms`: recommended wait after a complete decision.
- `timeout_ms`: recommended maximum wait for a hold decision.
- `speech_rms_threshold`: threshold used to distinguish speech from silence.

Experiment configurations are in [`configs/experiment`](configs/experiment).

## Project structure

```text
app.py                         Gradio application entry point
configs/                       YAML data, model, and runtime configurations
infra/model_app.py             Modal cloud jobs
src/turn_detector/
  audio.py                     Audio loading and preprocessing
  cli.py                       Command-line interface
  config.py                    Configuration validation
  demo.py                      Gradio UI and prediction callbacks
  environment.py               .env loading
  features.py                  Whisper-based feature extraction
  inference.py                 ONNX model loading and inference
  modeling.py                  Training model definition
  training/                    Dataset and training implementation
  evaluation/                  Metrics and evaluation reports
test/                          Automated tests
```

## Modal cloud jobs

Install the cloud dependency:

```powershell
uv sync --extra cloud
```

Authenticate with Modal:

```powershell
uv run modal token new
```

Create the persistent volume once:

```powershell
uv run modal volume create hinglish-turn-data
```

Run data preparation:

```powershell
uv run modal run infra\model_app.py::prepare
```

Run training:

```powershell
uv run modal run infra\model_app.py::train_model
```

Run an experiment:

```powershell
uv run modal run infra\model_app.py::train_experiment `
  --experiment e5_causal_filler
```

## Testing

Run the full test suite:

```powershell
uv run pytest
```

Run configuration tests:

```powershell
uv run pytest test\test_config.py
```

Check the main entry points for syntax errors:

```powershell
uv run python -m py_compile app.py infra\model_app.py
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'gradio'`

```powershell
uv sync --extra demo
```

### ONNX inference dependency error

```powershell
uv sync --extra runtime --extra export
```

### Hugging Face reports `404 Repository Not Found`

Check `HINGLISH_TURN_MODEL` in [`.env`](.env). It must be a valid local model
path or an accessible Hugging Face model repository.

### Modal reports `Token missing`

```powershell
uv run modal token new
```

### Commands cannot find project modules

Run from the repository root and use `uv run`:

```powershell
cd D:\TURN_DETECTION_MODEL
uv run turn-detector --help
```

## License

See [LICENSE](LICENSE).
