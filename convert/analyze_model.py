import onnx
import json
import os
from onnx import numpy_helper
import numpy as np

def analyze_model(model_path='input/model.onnx'):
    """Analyze ONNX model and extract metadata"""
    model = onnx.load(model_path)
    
    # Basic info
    opset = model.opset_import[0].version if model.opset_import else 'unknown'
    
    # Inputs
    inputs = []
    for inp in model.graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value:
                shape.append(dim.dim_value)
            else:
                shape.append('dynamic')
        inputs.append({
            'name': inp.name,
            'shape': shape,
            'dtype': onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
        })
    
    # Outputs
    outputs = []
    for out in model.graph.output:
        shape = []
        for dim in out.type.tensor_type.shape.dim:
            if dim.dim_value:
                shape.append(dim.dim_value)
            else:
                shape.append('dynamic')
        outputs.append({
            'name': out.name,
            'shape': shape
        })
    
    # Count parameters
    total_params = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        total_params += arr.size
    
    # Detect model type from ops
    ops = [node.op_type for node in model.graph.node]
    unique_ops = list(set(ops))
    
    model_type = 'unknown'
    if any(op in ops for op in ['NonMaxSuppression', 'RoiAlign', 'TopK']):
        model_type = 'object_detection'
    elif any(op in ops for op in ['Resize', 'Upsample']) and len([o for o in ops if o == 'Conv']) > 10:
        model_type = 'segmentation'
    elif len([o for o in ops if o == 'Conv']) > 5:
        model_type = 'classification'
    elif any(op in ops for op in ['MatMul', 'Attention', 'LayerNormalization']):
        model_type = 'nlp_transformer'
    
    # Count nodes
    node_count = len(model.graph.node)
    
    # File size
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    analysis = {
        'model_type': model_type,
        'opset_version': opset,
        'inputs': inputs,
        'outputs': outputs,
        'parameters': int(total_params),
        'parameters_human': f'{total_params/1e6:.1f}M' if total_params > 1e6 else f'{total_params/1e3:.1f}K',
        'node_count': node_count,
        'unique_ops': unique_ops[:20],  # Top 20 unique ops
        'file_size_mb': round(size_mb, 2),
        'recommended_batch': 1  # Default
    }
    
    # Save analysis
    os.makedirs('output', exist_ok=True)
    with open('output/analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n📊 Model Analysis:")
    print(f"  Type: {model_type}")
    print(f"  Parameters: {analysis['parameters_human']}")
    print(f"  Nodes: {node_count}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Inputs: {inputs}")
    print(f"  Outputs: {outputs}")
    
    return analysis

if __name__ == '__main__':
    analyze_model()