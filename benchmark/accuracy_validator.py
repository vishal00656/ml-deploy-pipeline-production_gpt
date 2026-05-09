from pathlib import Path
import json

from core.logger import logger


class AccuracyValidationError(RuntimeError):
    """Raised when inference accuracy validation cannot be completed."""


class OnnxAccuracyValidator:
    """Compare original and optimized ONNX outputs with ONNX Runtime."""

    def __init__(self, providers=None):
        self.providers = providers or ["CPUExecutionProvider"]

    def validate_pair(self, original_model, optimized_model, report_path=None):
        original_model = Path(original_model)
        optimized_model = Path(optimized_model)

        logger.info(
            "Validating ONNX inference accuracy: %s vs %s",
            original_model,
            optimized_model,
        )

        try:
            ort = self._import_onnxruntime()
            np = self._import_numpy()

            original_session = self._create_session(ort, original_model)
            optimized_session = self._create_session(ort, optimized_model)
            inputs = self._create_dummy_inputs(original_session)
            optimized_inputs = self._adapt_inputs_for_session(
                inputs,
                optimized_session,
            )

            original_outputs = original_session.run(None, inputs)
            optimized_outputs = optimized_session.run(None, optimized_inputs)

            output_metrics = self._compare_outputs(
                np,
                original_outputs,
                optimized_outputs,
            )
            aggregate_metrics = self._aggregate_metrics(np, output_metrics)

            report = {
                "status": "validated",
                "backend": "onnxruntime",
                "original_model": str(original_model),
                "optimized_model": str(optimized_model),
                "provider": original_session.get_providers()[0],
                "inputs": self._input_summary(original_session.get_inputs()),
                "outputs": output_metrics,
                "aggregate": aggregate_metrics,
            }

            if report_path is not None:
                report_path = Path(report_path)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                with open(report_path, "w") as file:
                    json.dump(report, file, indent=2)
                logger.info("Accuracy validation report written: %s", report_path)

            logger.info(
                "Accuracy validation completed: MAE %.8f, MSE %.8f, cosine %.8f",
                aggregate_metrics["mean_absolute_error"],
                aggregate_metrics["mean_squared_error"],
                aggregate_metrics["cosine_similarity"],
            )
            return report
        except Exception as exc:
            logger.exception("ONNX accuracy validation failed: %s", exc)
            if isinstance(exc, AccuracyValidationError):
                raise
            raise AccuracyValidationError(
                "Failed to validate optimized ONNX inference accuracy"
            ) from exc

    def _import_onnxruntime(self):
        try:
            import onnxruntime as ort

            return ort
        except ImportError as exc:
            raise AccuracyValidationError(
                "onnxruntime is required for ONNX accuracy validation"
            ) from exc

    def _import_numpy(self):
        try:
            import numpy as np

            return np
        except ImportError as exc:
            raise AccuracyValidationError(
                "numpy is required for ONNX accuracy validation"
            ) from exc

    def _create_session(self, ort, model_path):
        if not model_path.exists():
            raise AccuracyValidationError(f"Model not found: {model_path}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        return ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=self.providers,
        )

    def _create_dummy_inputs(self, session):
        np = self._import_numpy()
        inputs = {}
        for input_meta in session.get_inputs():
            shape = self._concrete_shape(input_meta.shape)
            inputs[input_meta.name] = self._dummy_array(np, shape, input_meta.type)
        return inputs

    def _adapt_inputs_for_session(self, source_inputs, session):
        np = self._import_numpy()
        adapted = {}
        source_values = list(source_inputs.values())
        for index, input_meta in enumerate(session.get_inputs()):
            target_shape = self._concrete_shape(input_meta.shape)
            source = source_inputs.get(input_meta.name)
            if source is None and len(source_values) == 1:
                source = source_values[0]
            elif source is None and index < len(source_values):
                source = source_values[index]

            if source is None or list(source.shape) != target_shape:
                adapted[input_meta.name] = self._dummy_array(
                    np,
                    target_shape,
                    input_meta.type,
                )
                continue

            adapted[input_meta.name] = np.asarray(
                source,
                dtype=self._numpy_dtype(np, input_meta.type),
            )
        return adapted

    def _concrete_shape(self, shape):
        concrete_shape = []
        for dim in shape:
            if isinstance(dim, int) and dim > 0:
                concrete_shape.append(dim)
            else:
                concrete_shape.append(1)
        return concrete_shape

    def _dummy_array(self, np, shape, onnx_type):
        dtype = self._numpy_dtype(np, onnx_type)
        if dtype in (np.float16, np.float32, np.float64):
            return np.random.random_sample(shape).astype(dtype)
        if dtype == np.bool_:
            return np.zeros(shape, dtype=dtype)
        return np.random.randint(low=0, high=10, size=shape).astype(dtype)

    def _numpy_dtype(self, np, onnx_type):
        dtype_map = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(int16)": np.int16,
            "tensor(int8)": np.int8,
            "tensor(uint8)": np.uint8,
            "tensor(bool)": np.bool_,
        }

        if onnx_type not in dtype_map:
            raise AccuracyValidationError(
                f"Unsupported ONNX input type: {onnx_type}"
            )
        return dtype_map[onnx_type]

    def _compare_outputs(self, np, original_outputs, optimized_outputs):
        if len(original_outputs) != len(optimized_outputs):
            raise AccuracyValidationError(
                "Original and optimized models returned different output counts"
            )

        metrics = []
        for index, (original, optimized) in enumerate(
            zip(original_outputs, optimized_outputs)
        ):
            original_array = np.asarray(original, dtype=np.float64)
            optimized_array = np.asarray(optimized, dtype=np.float64)

            if original_array.shape != optimized_array.shape:
                raise AccuracyValidationError(
                    "Output shape mismatch at index "
                    f"{index}: {original_array.shape} != {optimized_array.shape}"
                )

            diff = original_array - optimized_array
            mae = float(np.mean(np.abs(diff)))
            mse = float(np.mean(np.square(diff)))
            cosine_similarity = self._cosine_similarity(
                np,
                original_array,
                optimized_array,
            )

            metrics.append(
                {
                    "index": index,
                    "shape": list(original_array.shape),
                    "mean_absolute_error": round(mae, 10),
                    "mean_squared_error": round(mse, 10),
                    "cosine_similarity": round(cosine_similarity, 10),
                }
            )

        return metrics

    def _cosine_similarity(self, np, original, optimized):
        original_flat = original.reshape(-1)
        optimized_flat = optimized.reshape(-1)
        denominator = np.linalg.norm(original_flat) * np.linalg.norm(optimized_flat)

        if denominator == 0:
            return 1.0 if np.allclose(original_flat, optimized_flat) else 0.0

        return float(np.dot(original_flat, optimized_flat) / denominator)

    def _aggregate_metrics(self, np, output_metrics):
        if not output_metrics:
            raise AccuracyValidationError("No model outputs were produced")

        return {
            "mean_absolute_error": round(
                float(np.mean([item["mean_absolute_error"] for item in output_metrics])),
                10,
            ),
            "mean_squared_error": round(
                float(np.mean([item["mean_squared_error"] for item in output_metrics])),
                10,
            ),
            "cosine_similarity": round(
                float(np.mean([item["cosine_similarity"] for item in output_metrics])),
                10,
            ),
        }

    def _input_summary(self, inputs):
        return [
            {
                "name": input_meta.name,
                "shape": self._concrete_shape(input_meta.shape),
                "type": input_meta.type,
            }
            for input_meta in inputs
        ]
