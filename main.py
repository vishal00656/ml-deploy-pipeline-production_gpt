import argparse
from pathlib import Path

from benchmark.validator import generate_validation_report
from convert.auto_convert import UniversalConverter
from core.config_manager import ConfigManager
from core.logger import logger
from decision_engine.hardware_capabilities import HardwareCapabilities
from decision_engine.model_analyzer import ModelAnalyzer
from decision_engine.recommendation_report import RecommendationReporter
from decision_engine.strategy_selector import StrategySelector
from optimize.smart_optimizer import SmartOptimizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--hardware", required=True)
    return parser.parse_args()


def load_legacy_hardware_config(config, target):
    try:
        return config.load_hardware(target)
    except FileNotFoundError:
        logger.info(
            "No legacy hardware YAML found for %s; using decision-engine profile",
            target,
        )
        return {}


def load_calibration_config(config):
    try:
        return config.load_calibration()
    except FileNotFoundError:
        logger.info("No calibration YAML found; using static INT8 defaults")
        return {}


def runtime_profile(hardware_profile, legacy_profile, strategy, calibration_config=None):
    profile = hardware_profile.to_legacy_profile()
    profile.update(legacy_profile or {})
    profile.update(strategy.to_hardware_overrides())
    profile["calibration"] = calibration_config or {}
    profile["calibration_config_path"] = "configs/calibration.yaml"
    return profile


def main():
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info("Starting adaptive ML deployment optimization pipeline")

    config = ConfigManager()
    analyzer = ModelAnalyzer()
    hardware_capabilities = HardwareCapabilities()
    selector = StrategySelector()
    reporter = RecommendationReporter()

    legacy_profile = load_legacy_hardware_config(config, args.hardware)
    calibration_config = load_calibration_config(config)
    hardware_profile = hardware_capabilities.get_profile(
        args.hardware,
        legacy_profile=legacy_profile,
    )

    # Source analysis enables conversion-time decisions such as PyTorch pruning.
    source_analysis = analyzer.analyze(model_path)
    preliminary_strategy = selector.select(source_analysis, hardware_profile)
    conversion_profile = runtime_profile(
        hardware_profile,
        legacy_profile,
        preliminary_strategy,
        calibration_config,
    )

    if not preliminary_strategy.feasible:
        reporter.write(source_analysis, hardware_profile, preliminary_strategy)
        raise RuntimeError(
            "Deployment rejected before conversion by decision engine. "
            "See reports/optimization_recommendation.json"
        )

    converter = UniversalConverter()
    onnx_model = converter.convert_to_onnx(model_path, conversion_profile)

    # ONNX graph analysis is authoritative for final optimization selection.
    model_analysis = analyzer.analyze(onnx_model)
    strategy = selector.select(model_analysis, hardware_profile)
    reporter.write(model_analysis, hardware_profile, strategy)

    if not strategy.feasible:
        raise RuntimeError(
            "Deployment rejected by decision engine. "
            "See reports/optimization_recommendation.json"
        )

    optimizer = SmartOptimizer()
    optimized_model = optimizer.optimize(
        onnx_model,
        runtime_profile(hardware_profile, legacy_profile, strategy, calibration_config),
    )

    report = generate_validation_report(optimized_model)
    logger.info(f"Validation report: {report}")
    logger.info("Adaptive pipeline completed successfully")


if __name__ == "__main__":
    main()
