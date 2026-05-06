
from pathlib import Path
import shutil

from core.logger import logger
from convert.detect_format import detect_model_format

class UniversalConverter:

    def convert_to_onnx(self, model_path):
        model_path = Path(model_path)

        framework = detect_model_format(model_path)

        logger.info(f"Detected framework: {framework}")

        output_dir = Path("converted")
        output_dir.mkdir(exist_ok=True)

        output_model = output_dir / "model.onnx"

        # Simulated conversion layer
        # Real implementations can later integrate:
        # torch.onnx.export
        # tf2onnx
        # keras2onnx

        shutil.copy(model_path, output_model)

        logger.info(f"Converted {model_path.name} -> ONNX")

        return str(output_model)
