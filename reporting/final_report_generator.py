from pathlib import Path
import json

from core.logger import logger


REPORT_FILES = {
    "benchmark": "benchmark_statistics.json",
    "accuracy": "accuracy_validation.json",
    "size": "model_size_analysis.json",
    "recommendation": "optimization_recommendation.json",
    "pruning": "pruning_statistics.json",
    "static_int8": "static_int8_quantization.json",
    "tflite": "tflite_conversion.json",
    "summary": "optimization_summary.json",
}


def generate_final_report(
    reports_dir="reports",
    output_path="output/final_execution_report.md",
    optimization_stats_path="output/optimization_stats.json",
):
    """Generate a concise deployment report beside the optimized artifact."""
    reports_dir = Path(reports_dir)
    output_path = Path(output_path)
    logger.info("Generating final execution report: %s", output_path)

    reports = _load_reports(reports_dir)
    output_stats = _read_json(Path(optimization_stats_path))
    if output_stats:
        reports["summary"] = {**reports.get("summary", {}), **output_stats}

    lines = [
        "# Final Optimization Report",
        "",
        *generate_executive_summary(reports),
        "",
        *generate_decision_summary(reports),
        "",
        *generate_optimization_summary(reports),
        "",
        *generate_size_summary(reports),
        "",
        *generate_benchmark_summary(reports),
        "",
        *generate_accuracy_summary(reports),
        "",
        *generate_quantization_summary(reports),
        "",
        *generate_final_verdict(reports),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Final execution report written: %s", output_path)
    return output_path


def generate_executive_summary(reports):
    summary = reports.get("summary", {})
    recommendation = reports.get("recommendation", {})
    tflite = reports.get("tflite", {}) or summary.get("tflite_conversion", {})
    strategy = recommendation.get("recommended_optimization_strategy", {})
    optimized_model = summary.get("optimized_model") or tflite.get("tflite_model")
    status = "SUCCESS" if summary.get("status") == "optimized" else "FAILED"

    return [
        "## Executive Summary",
        "",
        f"- Input Model: {_inline(summary.get('original_model'))}",
        f"- Optimized Model: {_inline(optimized_model)}",
        f"- Hardware Target: {_plain(summary.get('hardware_target') or recommendation.get('hardware_target'))}",
        f"- Optimization Selected: {_plain(summary.get('quantization') or strategy.get('primary_optimization'))}",
        f"- Runtime Recommendation: {_plain(summary.get('runtime_recommendation') or _runtime_recommendation(tflite))}",
        f"- Deployment Format: {_plain(_deployment_format(optimized_model, tflite))}",
        f"- Status: **{status}**",
    ]


def generate_decision_summary(reports):
    summary = reports.get("summary", {})
    recommendation = reports.get("recommendation", {})
    decision = summary.get("decision_engine", {})
    strategy = recommendation.get("recommended_optimization_strategy", {})
    selected = decision.get("primary_optimization") or strategy.get("primary_optimization")
    rejected = decision.get("rejected_optimizations") or recommendation.get(
        "rejected_optimization_strategies",
        [],
    )
    reasons = decision.get("reasons") or recommendation.get(
        "why_decisions_were_made",
        [],
    )

    lines = [
        "## Decision Engine Summary",
        "",
        f"- Selected Optimization: {_plain(selected)}",
        f"- Rejected Optimizations: {_plain(_join_list(rejected))}",
        "- Decision Reasons:",
    ]
    lines.extend(_bullet_lines(_limit_items(reasons, 4), "No decision reasons were available."))
    return lines


def generate_optimization_summary(reports):
    summary = reports.get("summary", {})
    static_int8 = _static_int8_report(reports)
    pruning = reports.get("pruning", {})
    tflite = reports.get("tflite", {}) or summary.get("tflite_conversion", {})
    steps = []

    if summary.get("graph_optimization") is not None or _pipeline_has(summary, "graph"):
        steps.append("Graph Optimization")
    if pruning or _pipeline_has(summary, "pruning"):
        steps.append("Structured Pruning")
    if static_int8.get("quantized_operator_count"):
        steps.append("Static INT8 Quantization")
    if summary.get("quantization") == "int8":
        steps.append("Dynamic INT8 Quantization")
    if summary.get("quantization") == "fp16":
        steps.append("FP16 Conversion")
    if tflite.get("status") == "converted":
        steps.append("TFLite Export")

    return [
        "## Optimization Steps Applied",
        "",
        *_bullet_lines(_dedupe(steps), "No optimization step information was available."),
    ]


def generate_size_summary(reports):
    summary = reports.get("summary", {})
    size = reports.get("size", {}) or summary.get("model_size_analysis", {})
    original = summary.get("original_size_mb") or _get(size, "models", "original", "file_size_mb")
    optimized = summary.get("optimized_size_mb") or _get(size, "models", "optimized", "file_size_mb")
    compression = summary.get("compression_percent")
    if compression is None:
        delta = _get(size, "comparisons", "original_vs_optimized", "size_delta_percent")
        compression = -delta if isinstance(delta, (int, float)) else None

    return [
        "## Model Size Comparison",
        "",
        "| Metric | Value |",
        "| -------------- | ----- |",
        f"| Original Size | {_format_mb(original)} |",
        f"| Optimized Size | {_format_mb(optimized)} |",
        f"| Compression | {_format_percent(compression)} |",
    ]


def generate_benchmark_summary(reports):
    benchmark = reports.get("benchmark", {}) or reports.get("summary", {}).get("benchmark", {})
    if benchmark.get("status") == "skipped":
        return [
            "## Benchmark Results",
            "",
            f"- {_plain(benchmark.get('reason') or 'Benchmarking was skipped.')}",
            f"- {_plain(benchmark.get('note') or 'No benchmark values are available.')}",
        ]

    original = benchmark.get("original", {})
    optimized = benchmark.get("optimized", {})
    comparison = benchmark.get("comparison", {})
    latency_gain = comparison.get("average_latency_improvement_percent")
    throughput_gain = comparison.get("throughput_improvement_percent")

    return [
        "## Benchmark Results",
        "",
        "| Metric | Before | After |",
        "| --------------- | ------ | ------ |",
        f"| Average Latency | {_format_ms(original.get('average_latency_ms'))} | {_format_ms(optimized.get('average_latency_ms'))} |",
        f"| Throughput | {_format_throughput(original.get('throughput_inferences_per_second'))} | {_format_throughput(optimized.get('throughput_inferences_per_second'))} |",
        "",
        f"Latency Improvement: {_format_percent(latency_gain)}",
        "",
        f"Throughput Improvement: {_format_percent(throughput_gain)}",
    ]


def generate_accuracy_summary(reports):
    accuracy = reports.get("accuracy", {}) or reports.get("summary", {}).get("accuracy_validation", {})
    if accuracy.get("status") == "skipped":
        return [
            "## Accuracy Validation",
            "",
            f"- {_plain(accuracy.get('reason') or 'Accuracy validation was skipped.')}",
            f"- {_plain(accuracy.get('note') or 'No accuracy values are available.')}",
            "",
            "Interpretation: Not available",
        ]

    aggregate = accuracy.get("aggregate", {})
    cosine = aggregate.get("cosine_similarity")
    return [
        "## Accuracy Validation",
        "",
        "| Metric | Value |",
        "| ------------------- | ----- |",
        f"| Cosine Similarity | {_format_number(cosine, 6)} |",
        f"| Mean Absolute Error | {_format_number(aggregate.get('mean_absolute_error'), 6)} |",
        f"| Mean Squared Error | {_format_number(aggregate.get('mean_squared_error'), 6)} |",
        "",
        _accuracy_interpretation(cosine),
    ]


def generate_quantization_summary(reports):
    static_int8 = _static_int8_report(reports)
    return [
        "## Quantization Statistics",
        "",
        f"- Quantized Operators: {_format_int(static_int8.get('quantized_operator_count'))}",
        f"- Conv Layers Quantized: {_format_int(static_int8.get('conv_quantization_count'))}",
        f"- Activation Tensors Quantized: {_format_int(static_int8.get('activation_quantization_count'))}",
    ]


def generate_final_verdict(reports):
    summary = reports.get("summary", {})
    benchmark = reports.get("benchmark", {}) or summary.get("benchmark", {})
    accuracy = reports.get("accuracy", {}) or summary.get("accuracy_validation", {})
    static_int8 = _static_int8_report(reports)
    tflite = reports.get("tflite", {}) or summary.get("tflite_conversion", {})
    compression = summary.get("compression_percent")
    speedup = _get(benchmark, "comparison", "average_latency_improvement_percent")
    cosine = _get(accuracy, "aggregate", "cosine_similarity")
    conclusions = []

    conclusions.append(
        "Optimization completed successfully."
        if summary.get("status") == "optimized"
        else "Optimization did not complete successfully."
    )
    if isinstance(compression, (int, float)):
        if compression >= 40:
            conclusions.append("Major model size reduction was achieved.")
        elif compression > 0:
            conclusions.append("Model size was reduced.")
        else:
            conclusions.append("Model size reduction was not achieved.")
    if isinstance(cosine, (int, float)):
        if cosine >= 0.99:
            conclusions.append("Accuracy preservation is excellent.")
        elif cosine >= 0.95:
            conclusions.append("Accuracy remained acceptable after optimization.")
        else:
            conclusions.append("Accuracy degradation was observed.")
    if isinstance(speedup, (int, float)):
        conclusions.append(
            "Latency improved in benchmark testing."
            if speedup > 0
            else "Latency did not improve in benchmark testing."
        )
    if static_int8.get("conv_quantization_count"):
        conclusions.append("CNN layers were successfully quantized.")
    if tflite.get("status") == "converted":
        conclusions.append("A TensorFlow Lite artifact is ready for Android deployment.")
    if summary.get("status") == "optimized":
        conclusions.append("Model is ready for deployment review on the target hardware.")

    return [
        "## Final Verdict",
        "",
        *_bullet_lines(_limit_items(_dedupe(conclusions), 6), "No final verdict could be generated."),
    ]


def _load_reports(reports_dir):
    reports = {}
    for key, filename in REPORT_FILES.items():
        reports[key] = _read_json(reports_dir / filename)
    return reports


def _read_json(path):
    try:
        if not path.exists():
            logger.info("Optional report not found: %s", path)
            return {}
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read report %s: %s", path, exc)
        return {}


def _static_int8_report(reports):
    return reports.get("static_int8", {}) or reports.get("summary", {}).get("graph", {})


def _accuracy_interpretation(cosine):
    if not isinstance(cosine, (int, float)):
        return "Interpretation: Not available"
    if cosine >= 0.99:
        return "Excellent accuracy preservation"
    if cosine >= 0.95:
        return "Good accuracy preservation"
    return "Accuracy degradation observed"


def _runtime_recommendation(tflite):
    if tflite.get("runtime") == "tflite":
        return "Use TensorFlow Lite on Android."
    return "Use the optimized model with the selected target runtime."


def _deployment_format(optimized_model, tflite):
    if optimized_model:
        suffix = Path(str(optimized_model)).suffix
        if suffix:
            return suffix
    if tflite.get("runtime") == "tflite":
        return ".tflite"
    return ".onnx"


def _pipeline_has(summary, step):
    return step in (summary.get("decision_engine", {}).get("pipeline") or [])


def _bullet_lines(items, fallback):
    if not items:
        return [f"- {fallback}"]
    return [f"- {_plain(item)}" for item in items]


def _limit_items(items, limit):
    return list(items or [])[:limit]


def _dedupe(items):
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _get(data, *keys, default=None):
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def _join_list(items):
    if not items:
        return "None"
    return ", ".join(str(item).replace("_", " ") for item in items)


def _inline(value):
    if value in (None, ""):
        return "Not available"
    return f"`{value}`"


def _plain(value):
    if value in (None, ""):
        return "Not available"
    if isinstance(value, list):
        return _join_list(value)
    return str(value).replace("_", " ")


def _format_mb(value):
    if not isinstance(value, (int, float)):
        return "Not available"
    return f"{value:.2f} MB"


def _format_ms(value):
    if not isinstance(value, (int, float)):
        return "Not available"
    return f"{value:.2f} ms"


def _format_throughput(value):
    if not isinstance(value, (int, float)):
        return "Not available"
    return f"{value:.2f}/sec"


def _format_percent(value):
    if not isinstance(value, (int, float)):
        return "Not available"
    return f"{value:.2f} %"


def _format_number(value, digits):
    if not isinstance(value, (int, float)):
        return "Not available"
    return f"{value:.{digits}f}"


def _format_int(value):
    if not isinstance(value, int):
        return "Not available"
    return f"{value:,}"

