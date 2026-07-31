# Universal ML Deployment Pipeline

Hardware-aware optimization, quantization, validation, and packaging for machine
learning models targeting edge deployment.

## Project Overview

Universal ML Deployment Pipeline converts supported model formats to ONNX,
selects an optimization strategy from model and hardware characteristics, applies
the selected optimization pipeline, validates the output, benchmarks latency, and
emits deployment-ready artifacts.

The project is designed for reproducible release workflows: source code,
configuration, documentation, and small placeholders are tracked; generated
models, reports, logs, benchmark outputs, datasets, and downloaded model weights
are recreated locally or by GitHub Actions.

## Features

- Automatic input format detection for common ML model artifacts.
- PyTorch-to-ONNX export with validation.
- Hardware-aware strategy selection through a decision engine.
- Graph optimization for ONNX models.
- FP16 optimization for compatible deployment targets.
- Dynamic INT8 quantization for MatMul/Linear-heavy models.
- Static INT8 quantization with representative image calibration.
- Optional structured pruning for supported PyTorch sources.
- Accuracy validation and latency benchmarking with ONNX Runtime.
- Final execution reports and GitHub Actions artifact uploads.
- Placeholder artifact folders for clean public releases.

## Architecture

The pipeline is split into small, focused modules:

- `main.py` orchestrates model loading, conversion, strategy selection,
  optimization, validation, and final report generation.
- `convert/` detects source formats and converts models to ONNX.
- `decision_engine/` analyzes model structure and selects feasible strategies for
  the requested hardware.
- `optimize/` contains graph, FP16, INT8, pruning, and size-analysis backends.
- `calibration/` loads representative images for static INT8 quantization.
- `benchmark/` validates optimized artifacts and measures ONNX Runtime latency.
- `reporting/` creates final release/run summaries.
- `configs/` stores hardware and calibration profiles.

## Folder Structure

```text
.github/workflows/        GitHub Actions pipelines
benchmark/                Validation and latency benchmarking
calibration/              Calibration dataset loading and preprocessing
comparison/               Placeholder for local comparison artifacts
configs/                  Pipeline, hardware, and calibration configuration
convert/                  Model format detection and ONNX conversion
converted/                Placeholder for generated converted models
core/                     Logging, configuration, and artifact helpers
decision_engine/          Hardware-aware optimization strategy selection
deploy/                   Deployment packaging utilities
input/                    Placeholder for local source models
logs/                     Placeholder for runtime logs
models/                   Model registry metadata
optimize/                 Optimization backends
output/                   Placeholder for optimized runtime artifacts
pipeline/                 Legacy report helpers
reporting/                Final report generation
reports/                  Placeholder for generated JSON/Markdown reports
scripts/                  Utility scripts
tests/                    Quality gate checks
```

## Installation

Use Python 3.11. Creating a virtual environment is recommended:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

## Requirements

The direct Python dependencies are pinned in `requirements.txt` for v1.0
reproducibility. The pipeline uses:

- ONNX and ONNX Runtime for graph representation, optimization, quantization,
  validation, and benchmarking.
- PyTorch and Torchvision for PyTorch model loading/export and sample model
  generation.
- TensorFlow, `onnx2tf`, and `tf2onnx` for TensorFlow/TFLite conversion paths.
- Pillow and NumPy for calibration image preprocessing.
- PyYAML for configuration loading.
- Ultralytics for YOLO model support.

Large models, datasets, calibration images, generated reports, and runtime logs
are intentionally ignored by Git.

## Usage

For a quick reproducible demo that does not require calibration images:

```bash
python scripts/download_model.py --model mobilenetv2
python main.py --model input/mobilenetv2.pt --hardware cpu
```

The `--optimization` flag is optional. By default, `auto` delegates to the
decision engine.

For Raspberry Pi or explicit static INT8 runs, place representative calibration
images in `input/calibration/` before running the pipeline. The GitHub Actions
workflow creates temporary sample images for validation runs, but production
quantization should use representative images from your deployment domain.

## Example Commands

Generate a sample model artifact:

```bash
python scripts/download_model.py --model mobilenetv2
```

Run the verified CPU demo:

```bash
python main.py --model input/mobilenetv2.pt --hardware cpu
```

Run automatic optimization for Raspberry Pi after adding calibration images:

```bash
python main.py --model input/mobilenetv2.pt --hardware raspberry_pi
```

Force FP16 optimization for Jetson:

```bash
python main.py --model input/mobilenetv2.pt --hardware jetson --optimization fp16
```

Run static INT8 explicitly:

```bash
python main.py --model input/resnet50.pt --hardware raspberry_pi --optimization static_int8
```

## Example Output

Runtime outputs are generated into ignored folders:

```text
output/
  optimization_stats.json
  final_execution_report.md
  <optimized-model-artifact>

reports/
  optimization_recommendation.json
  optimization_summary.json
  benchmark_statistics.json
  accuracy_validation.json
  validation_summary.json

logs/
  pipeline.log
```

## Benchmark Results

Benchmark results depend on the model, target hardware profile, runtime, and
host machine. Generated benchmark summaries are written to
`reports/benchmark_statistics.json` and are uploaded by GitHub Actions as part
of the `optimization-reports` artifact.

For public documentation, add curated benchmark tables here after running the
v1.0 workflow on representative models and hardware.

## Architecture Flow

```text
Input Model -> Format Detection -> ONNX Conversion -> Model Analysis
  -> Hardware Profile -> Strategy Selection -> Optimization Pipeline
  -> Validation -> Benchmarking -> Reports + Deployment Artifacts
```

## GitHub Actions

Two workflows are included:

- `.github/workflows/optimize-model.yml` provides a manual `workflow_dispatch`
  release/demo pipeline. It creates a Python 3.11 environment, installs
  `requirements.txt`, generates the selected sample model, creates calibration
  samples, runs optimization, and uploads output, report, and log artifacts.
- `.github/workflows/pipeline.yml` provides the existing model-upload style
  pipeline for optimizing models committed or supplied under `input/`.

## Supported Hardware

- Raspberry Pi
- NVIDIA Jetson
- Android
- CPU
- ARM edge devices through extensible hardware profiles

Hardware behavior is configured in `configs/hardware/` and supplemented by the
decision engine.

## Supported Model Formats

- PyTorch: `.pt`, `.pth`
- ONNX: `.onnx`
- TensorFlow: `.pb`
- Keras: `.h5`, `.keras`
- Python model scripts: `.py`

## Future Work

- Publish curated benchmark results for common edge devices.
- Add example screenshots and a rendered architecture diagram.
- Add optional Docker or devcontainer setup for fully isolated environments.
- Expand hardware profiles for additional accelerators.
- Add release assets or GitHub Pages documentation for generated examples.
- Consider optional Git LFS guidance for users who want to version private model
  artifacts outside the public source repository.

## Citation

If you use this project in research or published benchmarking, cite it as:

```bibtex
@software{universal_ml_deployment_pipeline,
  title = {Universal ML Deployment Pipeline},
  version = {1.0.0},
  year = {2026},
  url = {https://github.com/vishal00656/ml-deploy-pipeline-production_gpt}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.

## Contributing

Contributions are welcome. Please read `CONTRIBUTING.md` before opening issues
or pull requests, and keep optimization behavior changes well documented and
covered by validation evidence.
