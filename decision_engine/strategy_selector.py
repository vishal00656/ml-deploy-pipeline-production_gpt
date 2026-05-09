from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import yaml

from core.logger import logger


EXECUTABLE_OPTIMIZATIONS = {"static_int8", "dynamic_int8", "fp16", "graph"}


@dataclass
class OptimizationStrategy:
    hardware_target: str
    primary_optimization: str
    optimization_pipeline: List[str]
    executable_optimization: str
    rejected_optimizations: List[str] = field(default_factory=list)
    deployment_recommendations: List[str] = field(default_factory=list)
    decision_reasons: List[str] = field(default_factory=list)
    feasibility_score: float = 0.0
    feasible: bool = True
    rule_id: str = "unknown"

    def to_dict(self):
        return asdict(self)

    def to_hardware_overrides(self):
        quantization = self.executable_optimization
        if quantization == "dynamic_int8":
            quantization = "int8"
        if quantization == "static_int8":
            quantization = "static_int8"
        if quantization == "graph":
            quantization = "graph"

        return {
            "decision_rule_id": self.rule_id,
            "decision_primary_optimization": self.primary_optimization,
            "decision_pipeline": self.optimization_pipeline,
            "decision_rejected_optimizations": self.rejected_optimizations,
            "decision_reasons": self.decision_reasons,
            "quantization": quantization,
            "graph_optimization": "graph" in self.optimization_pipeline
            or self.executable_optimization == "graph",
            "pruning_enabled": "pruning" in self.optimization_pipeline,
            "pruning_ratio": 0.2 if "pruning" in self.optimization_pipeline else 0.0,
            "deployment_recommendations": self.deployment_recommendations,
            "deployment_feasible": self.feasible,
            "deployment_feasibility_score": self.feasibility_score,
        }


class StrategySelector:
    def __init__(self, rules_path=None):
        self.rules_path = Path(rules_path or Path(__file__).with_name("optimization_rules.yaml"))
        self.rules = self._load_rules(self.rules_path)

    def select(self, model_analysis, hardware_profile):
        rule = self._best_rule(model_analysis, hardware_profile)
        primary = rule.get("recommendation", "graph")
        pipeline = list(rule.get("optimizations", []))
        rejected = list(rule.get("rejected_optimizations", []))
        reasons = [rule.get("reason", "No rule reason supplied.")]
        deployment_recommendations = self._deployment_recommendations(
            primary,
            pipeline,
            hardware_profile,
            rule,
        )

        feasible = primary != "reject"
        executable = self._executable_optimization(primary, pipeline)
        if primary == "reject":
            executable = "graph"
        if executable != primary and primary not in {"reject", "graph"}:
            reasons.append(
                f"{primary} is recommended, but this repository currently executes "
                f"{executable} for ONNX artifacts."
            )
        if "dynamic_int8" in rejected:
            reasons.append(self._dynamic_rejection_reason(model_analysis))
        if hardware_profile.tensorrt_support and "tensorrt" in pipeline:
            reasons.append("TensorRT export is recommended for deployment packaging.")
        if hardware_profile.tflite_preference and (
            "tflite" in pipeline or "tflite_micro" in pipeline
        ):
            reasons.append("TFLite-family export is preferred for this hardware target.")

        feasibility = self._feasibility_score(model_analysis, hardware_profile, feasible)

        strategy = OptimizationStrategy(
            hardware_target=hardware_profile.target,
            primary_optimization=primary,
            optimization_pipeline=pipeline,
            executable_optimization=executable,
            rejected_optimizations=rejected,
            deployment_recommendations=deployment_recommendations,
            decision_reasons=reasons,
            feasibility_score=feasibility,
            feasible=feasible and feasibility > 0,
            rule_id=rule.get("id", "unknown"),
        )
        self._log_strategy(strategy, model_analysis)
        return strategy

    def _best_rule(self, model_analysis, hardware_profile):
        matches = [
            rule
            for rule in self.rules
            if self._matches(rule, model_analysis, hardware_profile)
        ]
        if not matches:
            raise RuntimeError("No optimization rules matched")
        return sorted(matches, key=lambda rule: rule.get("priority", 0), reverse=True)[0]

    def _matches(self, rule, model_analysis, hardware_profile):
        if not self._match_any(rule.get("architectures", ["*"]), model_analysis.architecture_category):
            return False
        if not self._match_any(rule.get("hardware", ["*"]), hardware_profile.target):
            return False
        characteristics = rule.get("characteristics", [])
        if characteristics and not any(
            item in model_analysis.graph_characteristics for item in characteristics
        ):
            return False
        return self._conditions_match(rule.get("when", {}), model_analysis)

    def _conditions_match(self, conditions, model_analysis):
        for key, expected in conditions.items():
            if key == "tinyml_huge_model_rejection":
                actual = model_analysis.tinyml_suitability.huge_model_rejection
            elif key == "tinyml_suitable":
                actual = model_analysis.tinyml_suitability.microcontroller_suitable
            else:
                actual = None
            if actual != expected:
                return False
        return True

    def _match_any(self, patterns, value):
        return "*" in patterns or value in patterns

    def _executable_optimization(self, primary, pipeline):
        if primary in EXECUTABLE_OPTIMIZATIONS:
            return primary
        for item in pipeline:
            if item in EXECUTABLE_OPTIMIZATIONS and item != "graph":
                return item
        return "graph"

    def _deployment_recommendations(self, primary, pipeline, hardware_profile, rule):
        recommendations = []
        for item in pipeline:
            if item not in {"static_int8", "dynamic_int8", "fp16", "graph"}:
                recommendations.append(item)
        if primary == "reject":
            recommendations.append("reject_deployment")
        recommendations.extend(rule.get("next_steps", []))
        for runtime in hardware_profile.runtime_preferences:
            if runtime not in recommendations:
                recommendations.append(runtime)
        return recommendations

    def _dynamic_rejection_reason(self, model_analysis):
        if "Conv-heavy" in model_analysis.graph_characteristics:
            return "Dynamic quantization unsuitable for Conv-heavy graph because it primarily optimizes MatMul/Gemm weights."
        return "Dynamic quantization rejected by the selected architecture/hardware rule."

    def _feasibility_score(self, model_analysis, hardware_profile, feasible):
        if not feasible:
            return 0.0

        score = 100.0
        footprint_mb = model_analysis.tinyml_suitability.estimated_memory_footprint_mb
        if hardware_profile.memory_constraint_mb > 0:
            memory_ratio = footprint_mb / hardware_profile.memory_constraint_mb
            if memory_ratio > 1:
                score -= 70
            elif memory_ratio > 0.75:
                score -= 35
            elif memory_ratio > 0.5:
                score -= 15

        if model_analysis.tinyml_suitability.huge_model_rejection:
            score -= 40
        if hardware_profile.tinyml_compatibility and not model_analysis.tinyml_suitability.microcontroller_suitable:
            score -= 60
        return max(0.0, round(score, 2))

    def _log_strategy(self, strategy, model_analysis):
        if strategy.primary_optimization == "static_int8":
            logger.info(
                "Selected static INT8 quantization for %s deployment",
                strategy.hardware_target,
            )
        elif strategy.primary_optimization == "fp16":
            logger.info("Selected FP16 optimization for %s deployment", strategy.hardware_target)
        elif strategy.primary_optimization == "dynamic_int8":
            logger.info("Selected dynamic INT8 optimization for %s deployment", strategy.hardware_target)
        elif strategy.primary_optimization == "reject":
            logger.warning(
                "%s target rejected due to estimated memory footprint %.4f MB",
                strategy.hardware_target,
                model_analysis.tinyml_suitability.estimated_memory_footprint_mb,
            )
        else:
            logger.info("Selected %s optimization strategy", strategy.primary_optimization)

        for reason in strategy.decision_reasons:
            logger.info("Decision reason: %s", reason)
        for rejected in strategy.rejected_optimizations:
            logger.info("Rejected optimization strategy: %s", rejected)

    def _load_rules(self, path):
        with open(path, "r") as file:
            payload = yaml.safe_load(file) or {}
        return payload.get("rules", [])
