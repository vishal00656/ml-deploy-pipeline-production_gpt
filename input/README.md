# Input Artifacts

Place local model files here when running the pipeline manually.

Supported source formats include `.pt`, `.pth`, `.onnx`, `.pb`, `.h5`, `.keras`,
and Python model scripts.

Large model artifacts are intentionally not committed. To recreate the sample
workflow inputs, install dependencies and run:

```bash
python scripts/download_model.py --model resnet50
python scripts/download_model.py --model mobilenetv2
```

The GitHub Actions workflow also generates the selected model automatically.
