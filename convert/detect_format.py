
from pathlib import Path

SUPPORTED = {
    ".onnx": "onnx",
    ".pt": "pytorch",
    ".pth": "pytorch",
    ".h5": "keras",
    ".keras": "keras",
    ".pb": "tensorflow",
    ".py": "python_model"
}

def detect_model_format(model_path):
    suffix = Path(model_path).suffix.lower()

    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported model format: {suffix}")

    return SUPPORTED[suffix]
