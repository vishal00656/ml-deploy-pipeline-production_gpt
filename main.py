import argparse
from pathlib import Path

from benchmark.validator import generate_validation_report
from convert.auto_convert import UniversalConverter
from core.artifact_manager import cleanup_output_directory
from core.config_manager import ConfigManager
from core.logger import logger
from decision_engine.hardware_capabilities import HardwareCapabilities
from decision_engine.model_analyzer import ModelAnalyzer
from decision_engine.recommendation_report import RecommendationReporter
from decision_engine.strategy_selector import StrategySelector
from optimize.smart_optimizer import SmartOptimizer
from reporting.final_report_generator import generate_final_report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument(
        "--optimization",
        choices=["auto", "static_int8", "fp16", "dynamic_int8"],
        default="auto",
        help="Optional optimization override. Defaults to decision-engine auto mode.",
    )
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


def runtime_profile(
    hardware_profile,
    legacy_profile,
    strategy,
    calibration_config=None,
    source_model_stem=None,
):
    profile = hardware_profile.to_legacy_profile()
    profile.update(legacy_profile or {})
    profile.update(strategy.to_hardware_overrides())
    profile["calibration"] = calibration_config or {}
    profile["calibration_config_path"] = "configs/calibration.yaml"
    if source_model_stem:
        profile["source_model_stem"] = source_model_stem
    return profile


def apply_user_override(strategy, optimization):
    if not optimization or optimization == "auto":
        logger.info("Optimization selection: auto")
        return strategy

    logger.info("Optimization override requested: %s", optimization)
    strategy.primary_optimization = optimization
    strategy.executable_optimization = optimization
    strategy.optimization_pipeline = [optimization, "graph"]
    if optimization == "static_int8":
        strategy.deployment_recommendations = [
            item
            for item in strategy.deployment_recommendations
            if item != "dynamic_int8"
        ]
    strategy.decision_reasons = [
        *strategy.decision_reasons,
        f"User override selected {optimization}; decision-engine recommendation was overridden.",
    ]
    strategy.rule_id = f"user_override_{optimization}"
    strategy.feasible = True
    if strategy.feasibility_score <= 0:
        strategy.feasibility_score = 50.0
    return strategy


def main():
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info("Starting adaptive ML deployment optimization pipeline")
    logger.info("Model selection: %s", model_path)
    logger.info("Hardware selection: %s", args.hardware)
    logger.info("Optimization selection: %s", args.optimization)
    cleanup_output_directory(preserve_paths=[model_path])

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
        model_path.stem,
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
    strategy = apply_user_override(strategy, args.optimization)
    reporter.write(model_analysis, hardware_profile, strategy)

    if not strategy.feasible:
        raise RuntimeError(
            "Deployment rejected by decision engine. "
            "See reports/optimization_recommendation.json"
        )

    optimizer = SmartOptimizer()
    optimizer_profile = runtime_profile(
        hardware_profile,
        legacy_profile,
        strategy,
        calibration_config,
        model_path.stem,
    )
    if conversion_profile.get("pruned_model_path"):
        optimizer_profile["pruned_model_path"] = conversion_profile["pruned_model_path"]

    optimized_model = optimizer.optimize(
        onnx_model,
        optimizer_profile,
    )

    report = generate_validation_report(optimized_model)
    logger.info(f"Validation report: {report}")
    final_report = generate_final_report(output_path="output/final_execution_report.md")
    logger.info("Final execution report: %s", final_report)
    logger.info("Adaptive pipeline completed successfully")


if __name__ == "__main__":
    main()
