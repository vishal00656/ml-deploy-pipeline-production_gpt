# Repository Architecture

This repository is a universal ML deployment optimization pipeline.

Main focus:
- model optimization
- quantization
- pruning
- compression
- hardware-aware deployment

Priority is optimization engineering, not workflow automation.

Current architecture:

- convert/ -> framework conversion
- optimize/ -> optimization backends
- calibration/ -> representative image loading and ONNX Runtime calibration readers
- decision_engine/ -> model-aware and hardware-aware strategy selection
- benchmark/ -> validation and latency testing
- deploy/ -> deployment packaging
- core/ -> configs, logging, orchestration
- configs/ -> hardware and calibration profiles

Design goals:
- modular architecture
- scalable optimization system
- hardware-aware deployment
- model-aware strategy selection
- minimal accuracy loss
- deployable edge AI artifacts

Preferred optimization stack:
- ONNX Runtime
- FP16 conversion
- static INT8 quantization for Conv-heavy CNNs
- dynamic INT8 quantization for MatMul/Linear-heavy models
- graph optimization
- pruning
- TensorRT hooks
- OpenVINO hooks
- TensorFlow Lite support

Adaptive routing:
- YOLO/CNN + Raspberry Pi -> pruning, ONNX export, graph optimization, static INT8 calibration, benchmark, validation
- Transformer + CPU -> graph optimization and dynamic INT8
- YOLO/CNN + Jetson -> graph optimization, FP16, TensorRT recommendation
- TinyML-class hardware -> accept only models that fit microcontroller memory heuristics

Important:
- Do not simplify repository architecture.
- Extend modules professionally.
- Keep implementation modular.
