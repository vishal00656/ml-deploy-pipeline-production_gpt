from pathlib import Path
import json

from core.logger import logger


class RecommendationReporter:
    def write(
        self,
        model_analysis,
        hardware_profile,
        optimization_strategy,
        report_path="reports/optimization_recommendation.json",
    ):
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "detected_architecture": model_analysis.architecture_category,
            "detected_graph_characteristics": model_analysis.graph_characteristics,
            "detected_precision": model_analysis.precision,
            "hardware_target": hardware_profile.target,
            "hardware_profile": hardware_profile.to_dict(),
            "model_statistics": {
                "parameter_count": model_analysis.parameter_count,
                "initializer_count": model_analysis.initializer_count,
                "model_size_bytes": model_analysis.model_size_bytes,
                "model_size_mb": model_analysis.model_size_mb,
                "node_count": model_analysis.node_count,
                "operator_distribution": model_analysis.operator_distribution,
            },
            "recommended_optimization_strategy": optimization_strategy.to_dict(),
            "rejected_optimization_strategies": optimization_strategy.rejected_optimizations,
            "why_decisions_were_made": optimization_strategy.decision_reasons,
            "tinyml_suitability": model_analysis.tinyml_suitability.to_dict(),
            "deployment_feasibility_score": optimization_strategy.feasibility_score,
            "deployment_feasible": optimization_strategy.feasible,
        }

        with open(report_path, "w") as file:
            json.dump(payload, file, indent=2)

        logger.info("Optimization recommendation report written: %s", report_path)
        return payload

