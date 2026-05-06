import os
import sys
import torch
import onnx
import torchvision.models as models

MODEL_SOURCE = os.environ.get('MODEL_SOURCE', 'huggingface')
MODEL_ID = os.environ.get('MODEL_ID', 'resnet50')

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def merge_external_data():
    """Force ONNX model into single file (no external .data)"""
    model = onnx.load('input/model.onnx')
    onnx.save(model, 'input/model.onnx', save_as_external_data=False)
    print("✅ Merged external data into single file")

def convert_rfdetr():
    """Convert RF-DETR model to ONNX using built-in export() on CPU"""
    import torch
    from rfdetr import RFDETRBase, RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
    
    print(f"Loading RF-DETR model: {MODEL_ID}")
    
    rfdetr_models = {
        'rfdetr-nano': RFDETRNano,
        'rfdetr-small': RFDETRSmall,
        'rfdetr-medium': RFDETRMedium,
        'rfdetr-base': RFDETRBase,
        'rfdetr-large': RFDETRLarge,
        'Roboflow/rf-detr-base': RFDETRBase,
        'Roboflow/rf-detr-nano': RFDETRNano,
    }
    
    model_class = rfdetr_models.get(MODEL_ID.lower(), RFDETRBase)
    print(f"Using model class: {model_class.__name__}")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_default_device('cpu')
    
    model = model_class()
    
    ensure_dir('input')
    model.export(output_dir='input', simplify=True, device='cpu')
    
    import shutil
    shutil.move('input/inference_model.onnx', 'input/model.onnx')
    
    merge_external_data()
    print("✅ RF-DETR exported to ONNX on CPU")
    return True

def convert_huggingface():
    """Convert HuggingFace model to ONNX"""
    from transformers import AutoModel, AutoModelForObjectDetection, AutoImageProcessor
    
    print(f"Loading HuggingFace model: {MODEL_ID}")
    
    try:
        model = AutoModelForObjectDetection.from_pretrained(MODEL_ID)
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model.eval()
        
        size = getattr(processor, 'size', {'height': 560, 'width': 560})
        if isinstance(size, dict):
            h = size.get('height', 560)
            w = size.get('width', 560)
        else:
            h = w = 560
        
        dummy_input = torch.randn(1, 3, h, w)
        
        ensure_dir('input')
        torch.onnx.export(
            model,
            dummy_input,
            'input/model.onnx',
            input_names=['pixel_values'],
            output_names=['logits', 'pred_boxes'],
            dynamic_axes={
                'pixel_values': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
                'pred_boxes': {0: 'batch_size'}
            },
            opset_version=14
        )
        merge_external_data()
        print(f"✅ HF Object Detection model exported: {h}x{w}")
        return True
        
    except Exception as e:
        print(f"Not an object detection model or error: {e}")
    
    try:
        model = AutoModel.from_pretrained(MODEL_ID)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        
        ensure_dir('input')
        torch.onnx.export(
            model,
            dummy_input,
            'input/model.onnx',
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=14
        )
        merge_external_data()
        print("✅ HF Generic model exported")
        return True
        
    except Exception as e:
        print(f"Failed to load as generic model: {e}")
        return False

def convert_torchvision():
    """Convert torchvision model to ONNX"""
    print(f"Loading torchvision model: {MODEL_ID}")
    
    model_fn = getattr(models, MODEL_ID.replace('-', '_'), None)
    if model_fn is None:
        print(f"❌ Unknown torchvision model: {MODEL_ID}")
        print(f"Available: resnet18, resnet50, mobilenet_v2, efficientnet_b0, etc.")
        return False
    
    model = model_fn(weights='DEFAULT')
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    ensure_dir('input')
    torch.onnx.export(
        model,
        dummy_input,
        'input/model.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=14
    )
    merge_external_data()
    print("✅ Torchvision model exported")
    return True

def convert_from_url():
    """Download model from URL and convert"""
    import requests
    url = MODEL_ID
    
    print(f"Downloading model from: {url}")
    r = requests.get(url)
    
    ensure_dir('input')
    temp_path = 'input/downloaded_model'
    with open(temp_path, 'wb') as f:
        f.write(r.content)
    
    from convert.detect_format import detect_format, get_model_info
    fmt = detect_format(temp_path)
    info = get_model_info(fmt)
    
    print(f"Detected format: {fmt} - {info['description']}")
    
    if fmt == 'onnx':
        os.rename(temp_path, 'input/model.onnx')
        merge_external_data()
        return True
    else:
        print(f"❌ URL format '{fmt}' not yet supported for auto-convert")
        return False

def main():
    ensure_dir('input')
    
    if 'rf-detr' in MODEL_ID.lower() or 'rfdetr' in MODEL_ID.lower():
        success = convert_rfdetr()
    elif MODEL_SOURCE == 'huggingface':
        success = convert_huggingface()
    elif MODEL_SOURCE == 'torchvision':
        success = convert_torchvision()
    elif MODEL_SOURCE == 'url':
        success = convert_from_url()
    else:
        print(f"❌ Unknown source: {MODEL_SOURCE}")
        success = False
    
    if not success:
        sys.exit(1)
    
    model = onnx.load('input/model.onnx')
    onnx.checker.check_model(model)
    print("✅ ONNX validation passed")

if __name__ == '__main__':
    main()
