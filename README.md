
# Universal ML Deployment Pipeline

Hardware-aware automated ML deployment pipeline.

## Supported Input Formats

The pipeline now supports:

- PyTorch (.pt / .pth)
- TensorFlow (.pb)
- Keras (.h5 / .keras)
- Python model scripts (.py)
- ONNX (.onnx)

## Pipeline Flow

1. User uploads model
2. GitHub Actions triggers
3. Framework detection runs
4. Model converts to ONNX
5. Optimization + quantization execute
6. Validation + benchmarking run
7. Deployment artifacts generated

## Supported Targets

- Raspberry Pi
- NVIDIA Jetson
- Android
- ARM edge devices
- Extendable hardware profiles

## Quick Test

Put a model inside:

input/model.pt

Then push to GitHub.

The workflow automatically:
- detects framework
- converts to ONNX
- optimizes model
- packages deployment artifacts

## Static INT8 Calibration

Conv-heavy CNN targets such as Raspberry Pi now use ONNX Runtime static INT8
quantization with representative image calibration.

Place calibration images in:

input/calibration

Tune preprocessing and calibration behavior in:

configs/calibration.yaml

## Running Optimization from GitHub Actions

You can run the optimization pipeline directly from GitHub without using a
local terminal.

1. Open the repository on GitHub.
2. Go to the **Actions** tab.
3. Select **Optimize ML Model**.
4. Click **Run workflow**.
5. Choose:
   - **Model**: `resnet50` or `mobilenetv2`
   - **Hardware**: `raspberry_pi`, `jetson`, or `cpu`
   - **Optimization**: `auto`, `static_int8`, `fp16`, or `dynamic_int8`
6. Click **Run workflow**.

The workflow will create a Python 3.11 environment, install dependencies,
generate the selected model if it is not already present, run the adaptive
optimization pipeline, and upload artifacts.

### Workflow Inputs

| Input | Options | Description |
|---|---|---|
| Model | `resnet50`, `mobilenetv2` | Model artifact generated into `input/`. |
| Hardware | `raspberry_pi`, `jetson`, `cpu` | Target deployment profile. |
| Optimization | `auto`, `static_int8`, `fp16`, `dynamic_int8` | `auto` uses the decision engine; other values override it. |

### Downloading Artifacts

After the workflow completes, open the finished workflow run and download:

- `optimized-output`: optimized model artifacts, `optimization_stats.json`, and `final_execution_report.md`
- `optimization-reports`: recommendation, benchmark, validation, and detailed JSON reports
- `execution-logs`: `logs/pipeline.log` and workflow execution logs

### Placeholder Screenshots

Add screenshots here when publishing project documentation:

- Screenshot: Actions tab with **Optimize ML Model** selected.
- Screenshot: **Run workflow** input form.
- Screenshot: completed run artifacts panel.
