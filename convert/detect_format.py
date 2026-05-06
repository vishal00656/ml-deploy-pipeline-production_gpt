import os
from pathlib import Path

def detect_format(file_path):
    """Detect model format from file extension and content"""
    if not os.path.exists(file_path):
        return 'unknown'
    
    ext = Path(file_path).suffix.lower()
    
    if ext in ['.pt', '.pth']:
        return 'pytorch'
    elif ext == '.onnx':
        return 'onnx'
    elif ext == '.tflite':
        return 'tflite'
    elif ext in ['.h5', '.keras']:
        return 'keras'
    elif ext == '.pb':
        return 'tensorflow'
    elif os.path.isdir(file_path):
        # Check for saved_model.pb
        if os.path.exists(os.path.join(file_path, 'saved_model.pb')):
            return 'tensorflow_savedmodel'
        # Check for config.json (HuggingFace)
        if os.path.exists(os.path.join(file_path, 'config.json')):
            return 'huggingface_local'
    
    return 'unknown'

def get_model_info(format_type):
    """Get conversion strategy for format"""
    strategies = {
        'pytorch': {
            'converter': 'torch.onnx.export',
            'needs_architecture': True,
            'description': 'PyTorch model - needs architecture info or full model'
        },
        'onnx': {
            'converter': 'copy/validate',
            'needs_architecture': False,
            'description': 'Already ONNX - validate and optimize'
        },
        'tflite': {
            'converter': 'tflite2onnx',
            'needs_architecture': False,
            'description': 'TensorFlow Lite - convert to ONNX'
        },
        'tensorflow': {
            'converter': 'tf2onnx',
            'needs_architecture': False,
            'description': 'TensorFlow - convert to ONNX'
        },
        'huggingface_local': {
            'converter': 'transformers.onnx',
            'needs_architecture': False,
            'description': 'HuggingFace model - export to ONNX'
        }
    }
    return strategies.get(format_type, {'converter': 'unknown', 'description': 'Unknown format'})