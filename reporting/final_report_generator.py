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
    output_path="reports/final_execution_report.md",
):
    """Generate a presentation-friendly Markdown summary of a pipeline run."""
    reports_dir = Path(reports_dir)
    output_path = Path(output_path)
    logger.info("Generating final execution report: %s", output_path)

    reports = _load_reports(reports_dir)
    lines = _build_report_lines(reports)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Final execution report written: %s", output_path)
    return output_path


def _load_reports(reports_dir):
    reports = {}
    for key, filename in REPORT_FILES.items():
        path = reports_dir / filename
        reports[key] = _read_json(path)
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


def _build_report_lines(reports):
    summary = reports.get("summary", {})
    recommendation = reports.get("recommendation", {})
    benchmark = reports.get("benchmark") or summary.get("benchmark", {})
    accuracy = reports.get("accuracy") or summary.get("accuracy_validation", {})
    size = reports.get("size") or summary.get("model_size_analysis", {})
    static_int8 = reports.get("static_int8") or summary.get("graph", {})
    tflite = reports.get("tflite") or summary.get("tflite_conversion", {})
    pruning = reports.get("pruning", {})

    lines = [
        "# Final Optimization Report",
        "",
        * _model_information(summary, recommendation, tflite),
        "",
        * _decision_summary(summary, recommendation),
        "",
        * _optimization_steps(summary, static_int8, pruning, tflite),
        "",
        * _size_comparison(summary, size),
        "",
        * _benchmark_results(benchmark),
        "",
        * _accuracy_validation(accuracy),
        "",
        * _quantization_statistics(static_int8, tflite),
        "",
        * _final_verdict(summary, benchmark, accuracy, static_int8, tflite),
    ]
    return lines


def _model_information(summary, recommendation, tflite):
    strategy = _get(
        recommendation,
        "recommended_optimization_strategy",
        default={},
    )
    return [
        "## Model Information",
        f"- Input model: {_inline(summary.get('original_model') or _get(recommendation, 'model_statistics', 'path'))}",
        f"- Optimized model: {_inline(summary.get('optimized_model'))}",
        f"- Hardware target: {_text(summary.get('hardware_target') or recommendation.get('hardware_target'))}",
        f"- Optimization selected: {_text(summary.get('quantization') or strategy.get('primary_optimization'))}",
        f"- Runtime recommendation: {_text(summary.get('runtime_recommendation') or _runtime_recommendation(tflite))}",
        f"- Deployment format: {_text(Path(str(summary.get('optimized_model', ''))).suffix or tflite.get('runtime'))}",
    ]


def _decision_summary(summary, recommendation):
    decision = summary.get("decision_engine", {})
    strategy = recommendation.get("recommended_optimization_strategy", {})
    selected = (
        decision.get("primary_optimization")
        or strategy.get("primary_optimization")
        or "Not available"
    )
    rejected = (
        decision.get("rejected_optimizations")
        or recommendation.get("rejected_optimization_strategies")
        or []
    )
    reasons = (
        decision.get("reasons")
        or recommendation.get("why_decisions_were_made")
        or []
    )

    lines = [
        "## Decision Engine Summary",
        f"- Selected optimization: {_text(selected)}",
        f"- Rejected optimizations: {_text(_join_list(rejected))}",
        "- Reasoning:",
    ]
    lines.extend(_bullet_lines(reasons, fallback="No decision reasoning was available."))
    return lines


def _optimization_steps(summary, static_int8, pruning, tflite):
    steps = []
    if pruning:
        steps.append("Structured pruning was applied before export.")
    elif _has_pipeline_step(summary, "pruning"):
        steps.append("Structured pruning was selected for this hardware target.")

    if summary.get("graph_optimization"):
        steps.append("ONNX graph optimization was applied.")

    normalization = static_int8.get("fp32_normalization", {})
    if normalization.get("normalization_applied"):
        steps.append("FP16 or mixed precision tensors were normalized to FP32 before INT8 calibration.")

    if static_int8.get("quantized_operator_count"):
        steps.append("Static INT8 quantization was applied with representative calibration data.")

    quantization = summary.get("quantization")
    if quantization == "fp16":
        steps.append("FP16 conversion was applied for GPU-friendly inference.")
    elif quantization == "int8":
        steps.append("Dynamic INT8 quantization was applied.")
    elif quantization == "graph":
        steps.append("Graph optimization was used without precision quantization.")
    if tflite.get("status") == "converted":
        steps.append("The optimized ONNX model was converted to TensorFlow Lite for Android deployment.")

    return [
        "## Optimization Steps Applied",
        *_bullet_lines(steps, fallback="No optimization step details were available."),
    ]


def _size_comparison(summary, size):
    original = summary.get("original_size_mb") or _get(size, "models", "original", "file_size_mb")
    optimized = summary.get("optimized_size_mb") or _get(size, "models", "optimized", "file_size_mb")
    compression = summary.get("compression_percent")
    if compression is None:
        delta_percent = _get(size, "comparisons", "original_vs_optimized", "size_delta_percent")
        compression = -delta_percent if isinstance(delta_percent, (int, float)) else None

    return [
        "## Model Size Comparison",
        "| Metric | Value |",
        "|---|---:|",
        f"| Original size | {_format_mb(original)} |",
        f"| Optimized size | {_format_mb(optimized)} |",
        f"| Compression | {_format_percent(compression)} |",
    ]


def _benchmark_results(benchmark):
    if benchmark.get("status") == "skipped":
        return [
            "## Benchmark Results",
            f"- {benchmark.get('reason', 'Benchmarking was skipped.')}",
            f"- {benchmark.get('note', 'No benchmark numbers are available for this artifact.')}",
        ]
    original = benchmark.get("original", {})
    optimized = benchmark.get("optimized", {})
    comparison = benchmark.get("comparison", {})
    return [
        "## Benchmark Results",
        "| Metric | Before | After |",
        "|---|---:|---:|",
        f"| Average latency | {_format_ms(original.get('average_latency_ms'))} | {_format_ms(optimized.get('average_latency_ms'))} |",
        f"| Throughput | {_format_throughput(original.get('throughput_inferences_per_second'))} | {_format_throughput(optimized.get('throughput_inferences_per_second'))} |",
        "",
        f"Latency change: {_format_percent(comparison.get('average_latency_improvement_percent'))}. "
        f"Throughput change: {_format_percent(comparison.get('throughput_improvement_percent'))}.",
    ]


def _accuracy_validation(accuracy):
    if accuracy.get("status") == "skipped":
        return [
            "## Accuracy Validation",
            f"- {accuracy.get('reason', 'Accuracy validation was skipped.')}",
            f"- {accuracy.get('note', 'No accuracy numbers are available for this artifact.')}",
            "",
            "Interpretation: **not available**.",
        ]
    aggregate = accuracy.get("aggregate", {})
    cosine = aggregate.get("cosine_similarity")
    interpretation = _accuracy_interpretation(cosine)
    return [
        "## Accuracy Validation",
        "| Metric | Value |",
        "|---|---:|",
        f"| Cosine similarity | {_format_number(cosine, digits=6)} |",
        f"| Mean absolute error | {_format_number(aggregate.get('mean_absolute_error'), digits=6)} |",
        f"| Mean squared error | {_format_number(aggregate.get('mean_squared_error'), digits=6)} |",
        "",
        f"Interpretation: **{interpretation}**.",
    ]


def _quantization_statistics(static_int8, tflite):
    lines = [
        "## Quantization Statistics",
        "| Metric | Value |",
        "|---|---:|",
        f"| Quantized operators | {_format_int(static_int8.get('quantized_operator_count'))} |",
        f"| Conv layers quantized | {_format_int(static_int8.get('conv_quantization_count'))} |",
        f"| Activation tensors quantized | {_format_int(static_int8.get('activation_quantization_count'))} |",
    ]
    if tflite:
        lines.extend(
            [
                "",
                f"TFLite conversion: **{_text(tflite.get('status'))}**. "
                f"Runtime: **{_text(tflite.get('runtime'))}**. "
                f"Final artifact: {_inline(tflite.get('tflite_model'))}.",
            ]
        )
    return lines


def _final_verdict(summary, benchmark, accuracy, static_int8, tflite):
    compression = summary.get("compression_percent")
    speedup = _get(benchmark, "comparison", "average_latency_improvement_percent")
    cosine = _get(accuracy, "aggregate", "cosine_similarity")

    statements = []
    if summary.get("status") == "optimized":
        statements.append("Optimization completed successfully.")
    if isinstance(compression, (int, float)):
        if compression >= 40:
            statements.append("Major model size reduction was achieved.")
        elif compression > 0:
            statements.append("Model size was reduced.")
        else:
            statements.append("Model size did not improve in this run.")
    if isinstance(cosine, (int, float)):
        if cosine >= 0.99:
            statements.append("Accuracy stayed very close to the original model.")
        elif cosine >= 0.95:
            statements.append("Accuracy remained usable, but should be checked with real validation data.")
        else:
            statements.append("Accuracy drift is risky and needs review before deployment.")
    if isinstance(speedup, (int, float)):
        if speedup > 1:
            statements.append("Latency improved in the benchmark.")
        elif speedup > -1:
            statements.append("Latency was roughly unchanged in the benchmark.")
        else:
            statements.append("Latency did not improve in the benchmark.")
    if static_int8.get("conv_quantization_count"):
        statements.append("CNN layers were quantized with static INT8 calibration.")
    if tflite.get("status") == "converted":
        statements.append("A TensorFlow Lite artifact was generated for Android deployment.")
    if benchmark.get("status") == "skipped":
        statements.append("Runtime benchmarking should be performed with TensorFlow Lite tooling or on an Android device.")

    return [
        "## Final Verdict",
        *_bullet_lines(statements, fallback="Final verdict could not be determined from available reports."),
    ]


def _accuracy_interpretation(cosine):
    if not isinstance(cosine, (int, float)):
        return "not available"
    if cosine >= 0.99:
        return "excellent"
    if cosine >= 0.95:
        return "acceptable"
    return "risky"


def _has_pipeline_step(summary, step):
    return step in (summary.get("decision_engine", {}).get("pipeline") or [])


def _runtime_recommendation(tflite):
    if tflite.get("runtime") == "tflite":
        return "Use TensorFlow Lite on Android."
    return None


def _bullet_lines(items, fallback):
    if not items:
        return [f"- {fallback}"]
    return [f"- {_text(item)}" for item in items]


def _get(data, *keys, default=None):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _join_list(items):
    if not items:
        return "None reported"
    return ", ".join(str(item) for item in items)


def _inline(value):
    if value in (None, ""):
        return "Not available"
    return f"`{value}`"


def _text(value):
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
    return f"{value:.2f} / sec"


def _format_percent(value):
    if not isinstance(value, (int, float)):
        return "Not available"
    return f"{value:.2f}%"


def _format_number(value, digits=4):
    if not isinstance(value, (int, float)):
        return "Not available"
    return f"{value:.{digits}f}"


def _format_int(value):
    if not isinstance(value, int):
        return "Not available"
    return f"{value:,}"
