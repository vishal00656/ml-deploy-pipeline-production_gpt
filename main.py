
import argparse

from core.config_manager import ConfigManager
from core.logger import logger
from optimize.smart_optimizer import SmartOptimizer
from benchmark.validator import generate_validation_report

parser = argparse.ArgumentParser()

parser.add_argument("--model", required=True)
parser.add_argument("--hardware", required=True)
parser.add_argument("--optimize", default="balanced")

args = parser.parse_args()

config = ConfigManager()
hardware = config.load_hardware(args.hardware)

optimizer = SmartOptimizer()

optimized_model = optimizer.optimize(args.model, hardware)

report = generate_validation_report(optimized_model)

logger.info(f"Validation complete: {report}")
