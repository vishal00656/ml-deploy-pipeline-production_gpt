from pathlib import Path
import shutil

from core.logger import logger


class FP16OptimizationError(RuntimeError):
    """Raised when ONNX FP16 optimization cannot be completed."""


class FP16Optimizer:
    """Convert ONNX float tensors to FP16 and validate the resulting graph."""

    technique = "onnxconverter_common_fp16"
    quantization = "fp16"

    def optimize(self, model_path, output_path):
        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Running ONNX FP16 conversion: %s -> %s",
            model_path,
            output_path,
        )

        try:
            onnx = self._import_onnx()

            model = onnx.load(str(model_path))
            input_precision = self._detect_precision(onnx, model)
            logger.info(
                "Detected ONNX model precision before FP16 optimization: %s",
                input_precision["precision"],
            )

            if self._is_already_fp16(input_precision):
                return self._skip_conversion(
                    model_path,
                    output_path,
                    input_precision,
                    self._skip_reason(input_precision),
                )

            convert_float_to_float16 = self._import_float16_converter()
            try:
                fp16_model = convert_float_to_float16(
                    model,
                    keep_io_types=True,
                )
            except ValueError as exc:
                if self._is_known_float16_ready_error(exc):
                    return self._skip_conversion(
                        model_path,
                        output_path,
                        input_precision,
                        f"Converter reported model is already FP16-ready: {exc}",
                    )
                raise
            onnx.save(fp16_model, str(output_path))

            graph_stats = self.validate(output_path)
            graph_stats["conversion_skipped"] = False
            graph_stats["skip_reason"] = None
            graph_stats["detected_precision"] = input_precision
            logger.info("FP16 ONNX conversion completed: %s", output_path)
            return graph_stats
        except Exception as exc:
            if output_path.exists():
                output_path.unlink()
            logger.exception("FP16 ONNX optimization failed: %s", exc)
            if isinstance(exc, FP16OptimizationError):
                raise
            raise FP16OptimizationError(
                f"Failed to convert ONNX model '{model_path}' to FP16"
            ) from exc

    def validate(self, model_path):
        onnx = self._import_onnx()
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)

        ops = [node.op_type for node in model.graph.node]
        precision = self._detect_precision(onnx, model)
        fp16_initializer_count = precision["fp16_initializer_count"]
        fp32_initializer_count = precision["fp32_initializer_count"]

        if fp16_initializer_count == 0:
            logger.warning(
                "FP16 graph validated, but no FP16 initializers were found. "
                "The source model may not contain convertible float weights."
            )

        return {
            "node_count": len(ops),
            "unique_ops": sorted(set(ops)),
            "fp16_initializer_count": fp16_initializer_count,
            "fp32_initializer_count": fp32_initializer_count,
            "model_precision": precision["precision"],
            "fp16_tensor_count": precision["fp16_tensor_count"],
            "fp32_tensor_count": precision["fp32_tensor_count"],
        }

    def _detect_precision(self, onnx, model):
        fp16_initializer_count = 0
        fp32_initializer_count = 0
        fp16_tensor_count = 0
        fp32_tensor_count = 0

        for initializer in model.graph.initializer:
            if initializer.data_type == onnx.TensorProto.FLOAT16:
                fp16_initializer_count += 1
                fp16_tensor_count += 1
            elif initializer.data_type == onnx.TensorProto.FLOAT:
                fp32_initializer_count += 1
                fp32_tensor_count += 1

        for value_info in (
            list(model.graph.input)
            + list(model.graph.output)
            + list(model.graph.value_info)
        ):
            elem_type = value_info.type.tensor_type.elem_type
            if elem_type == onnx.TensorProto.FLOAT16:
                fp16_tensor_count += 1
            elif elem_type == onnx.TensorProto.FLOAT:
                fp32_tensor_count += 1

        if (
            fp16_initializer_count > 0
            and fp32_initializer_count == 0
            and fp32_tensor_count > 0
        ):
            precision = "fp16_weights_fp32_io"
        elif fp16_tensor_count > 0 and fp32_initializer_count == 0:
            precision = "fp16"
        elif fp16_tensor_count > 0 and fp32_tensor_count > 0:
            precision = "mixed_fp16_fp32"
        elif fp32_tensor_count > 0 or fp32_initializer_count > 0:
            precision = "fp32"
        else:
            precision = "unknown"

        return {
            "precision": precision,
            "fp16_initializer_count": fp16_initializer_count,
            "fp32_initializer_count": fp32_initializer_count,
            "fp16_tensor_count": fp16_tensor_count,
            "fp32_tensor_count": fp32_tensor_count,
            "fp16_tensor_ratio": self._ratio(
                fp16_tensor_count,
                fp16_tensor_count + fp32_tensor_count,
            ),
            "fp16_initializer_ratio": self._ratio(
                fp16_initializer_count,
                fp16_initializer_count + fp32_initializer_count,
            ),
        }

    def _is_already_fp16(self, precision):
        return (
            (
                precision["fp16_tensor_count"] > 0
                and precision["fp32_initializer_count"] == 0
            )
            or precision["fp16_tensor_ratio"] >= 0.5
            or precision["fp16_initializer_ratio"] >= 0.5
        )

    def _skip_reason(self, precision):
        if precision["fp16_tensor_ratio"] >= 0.5:
            return (
                "Model contains majority FP16 tensors; redundant FP16 "
                "conversion skipped"
            )
        if precision["fp16_initializer_ratio"] >= 0.5:
            return (
                "Model contains majority FP16 initializers; redundant FP16 "
                "conversion skipped"
            )
        return (
            "Model already contains FP16 tensors and no FP32 initializers "
            "requiring conversion"
        )

    def _skip_conversion(
        self,
        model_path,
        output_path,
        detected_precision,
        skip_reason,
    ):
        logger.info("Skipping FP16 conversion: %s", skip_reason)
        self._copy_model(model_path, output_path)
        graph_stats = self.validate(output_path)
        graph_stats["conversion_skipped"] = True
        graph_stats["skip_reason"] = skip_reason
        graph_stats["detected_precision"] = detected_precision
        return graph_stats

    def _is_known_float16_ready_error(self, exc):
        return "already converted to float16" in str(exc).lower()

    def _ratio(self, numerator, denominator):
        if denominator == 0:
            return 0.0
        return round(numerator / denominator, 4)

    def _copy_model(self, model_path, output_path):
        if Path(model_path).resolve() == Path(output_path).resolve():
            return
        shutil.copy2(model_path, output_path)

    def _import_onnx(self):
        try:
            import onnx

            return onnx
        except ImportError as exc:
            raise FP16OptimizationError(
                "onnx is required to load and validate FP16 ONNX artifacts"
            ) from exc

    def _import_float16_converter(self):
        try:
            from onnxconverter_common.float16 import convert_float_to_float16

            return convert_float_to_float16
        except ImportError as exc:
            raise FP16OptimizationError(
                "onnxconverter-common is required for FP16 ONNX conversion"
            ) from exc
