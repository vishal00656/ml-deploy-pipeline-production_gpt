
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
