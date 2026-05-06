
import argparse
from pathlib import Path

from core.logger import logger
from core.config_manager import ConfigManager
from convert.auto_convert import UniversalConverter
from optimize.smart_optimizer import SmartOptimizer
from benchmark.validator import generate_validation_report

parser = argparse.ArgumentParser()

parser.add_argument("--model", required=True)
parser.add_argument("--hardware", required=True)

args = parser.parse_args()

model_path = Path(args.model)

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

logger.info("Starting universal ML deployment pipeline")

config = ConfigManager()

hardware_profile = config.load_hardware(args.hardware)

# Step 1 — Convert to ONNX
converter = UniversalConverter()

onnx_model = converter.convert_to_onnx(model_path)

# Step 2 — Optimize
optimizer = SmartOptimizer()

optimized_model = optimizer.optimize(
    onnx_model,
    hardware_profile
)

# Step 3 — Validate
report = generate_validation_report(optimized_model)

logger.info(f"Validation report: {report}")

logger.info("Pipeline completed successfully")
