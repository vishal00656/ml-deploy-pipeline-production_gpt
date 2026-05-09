from pathlib import Path
import shutil

from core.logger import logger


class GraphOptimizationError(RuntimeError):
    """Raised when ONNX graph optimization cannot be completed."""


class OnnxGraphOptimizer:
    """Apply ONNX Runtime graph optimizations and operator fusions."""

    technique = "onnxruntime_graph_optimization"

    def __init__(
        self,
        optimization_level="all",
        providers=None,
        max_size_increase_percent=10.0,
    ):
        self.optimization_level = optimization_level
        self.providers = providers or ["CPUExecutionProvider"]
        self.max_size_increase_percent = max_size_increase_percent

    def optimize(self, model_path, output_path, hardware_profile=None):
        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        hardware_profile = hardware_profile or {}

        logger.info(
            "Running ONNX graph optimization: %s -> %s",
            model_path,
            output_path,
        )

        try:
            onnx = self._import_onnx()
            original_model = onnx.load(str(model_path))
            original_node_count = len(original_model.graph.node)
            original_size = self._model_size_bytes(model_path)

            passes_applied = []
            transformer_stats = self._try_transformer_optimizer(
                model_path,
                output_path,
                hardware_profile,
            )

            if transformer_stats["applied"]:
                passes_applied.extend(transformer_stats["passes_applied"])
            else:
                passes_applied.extend(
                    self._run_guarded_onnxruntime_graph_optimization(
                        model_path,
                        output_path,
                        original_size,
                    )
                )

            optimized_model = self._validate_graph(onnx, output_path)
            optimized_node_count = len(optimized_model.graph.node)
            optimized_size = self._model_size_bytes(output_path)

            stats = self._build_stats(
                model_path=model_path,
                output_path=output_path,
                original_node_count=original_node_count,
                optimized_node_count=optimized_node_count,
                original_size=original_size,
                optimized_size=optimized_size,
                passes_applied=passes_applied,
                transformer_stats=transformer_stats,
            )

            logger.info(
                "Graph optimization completed: %s -> %s nodes, %.2f%% node "
                "reduction",
                original_node_count,
                optimized_node_count,
                stats["node_count_reduction_percent"],
            )
            return stats
        except Exception as exc:
            if output_path.exists():
                output_path.unlink()
            logger.exception("ONNX graph optimization failed: %s", exc)
            if isinstance(exc, GraphOptimizationError):
                raise
            raise GraphOptimizationError(
                f"Failed to optimize ONNX graph '{model_path}'"
            ) from exc

    def _try_transformer_optimizer(self, model_path, output_path, hardware_profile):
        model_type = hardware_profile.get("graph_optimization_model_type")
        if not model_type:
            return {
                "applied": False,
                "reason": "No graph_optimization_model_type configured",
                "passes_applied": [],
            }

        try:
            from onnxruntime.transformers.optimizer import optimize_model
        except ImportError as exc:
            raise GraphOptimizationError(
                "onnxruntime transformer optimizer APIs are unavailable"
            ) from exc

        logger.info(
            "Using ONNX Runtime transformer optimizer for model type: %s",
            model_type,
        )
        optimized_model = optimize_model(
            str(model_path),
            model_type=model_type,
            num_heads=hardware_profile.get("num_heads", 0),
            hidden_size=hardware_profile.get("hidden_size", 0),
            opt_level=hardware_profile.get("graph_optimization_level"),
            only_onnxruntime=hardware_profile.get(
                "only_onnxruntime_graph_optimizations",
                False,
            ),
        )
        optimized_model.save_model_to_file(str(output_path))

        return {
            "applied": True,
            "model_type": model_type,
            "passes_applied": [
                "onnxruntime_transformer_optimizer",
                f"transformer_model_type:{model_type}",
            ],
        }

    def _run_onnxruntime_graph_optimization(self, model_path, output_path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise GraphOptimizationError(
                "onnxruntime is required for ONNX graph optimization"
            ) from exc

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = self._ort_optimization_level(ort)
        session_options.optimized_model_filepath = str(output_path)

        ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=self.providers,
        )

        if not output_path.exists():
            raise GraphOptimizationError(
                f"Optimized graph artifact was not created: {output_path}"
            )

        return [
            "onnxruntime_session_graph_optimization",
            f"graph_optimization_level:{self.optimization_level}",
            "operator_fusion",
            "constant_folding",
            "node_elimination",
        ]

    def _run_guarded_onnxruntime_graph_optimization(
        self,
        model_path,
        output_path,
        original_size,
    ):
        candidate_path = output_path.with_name(
            f"{output_path.stem}.candidate{output_path.suffix}"
        )
        retry_path = output_path.with_name(
            f"{output_path.stem}.extended_candidate{output_path.suffix}"
        )

        for path in (candidate_path, retry_path):
            if path.exists():
                path.unlink()

        selected_path = candidate_path
        selected_label = "onnxruntime_optimized_graph"
        passes = self._run_onnxruntime_graph_optimization(
            model_path,
            candidate_path,
        )
        candidate_increase = self._size_increase_percent(
            original_size,
            self._model_size_bytes(candidate_path),
        )

        if candidate_increase > self.max_size_increase_percent:
            logger.warning(
                "Graph optimization increased graph size by %.2f%%",
                candidate_increase,
            )

            if self.optimization_level == "all":
                original_level = self.optimization_level
                self.optimization_level = "extended"
                try:
                    retry_passes = self._run_onnxruntime_graph_optimization(
                        model_path,
                        retry_path,
                    )
                finally:
                    self.optimization_level = original_level

                retry_increase = self._size_increase_percent(
                    original_size,
                    self._model_size_bytes(retry_path),
                )
                if retry_increase < candidate_increase:
                    selected_path = retry_path
                    selected_label = "onnxruntime_extended_optimized_graph"
                    passes = retry_passes
                    candidate_increase = retry_increase

            if candidate_increase > self.max_size_increase_percent:
                logger.warning(
                    "Graph optimization artifact remains inflated by %.2f%%; "
                    "preserving original compact ONNX serialization for "
                    "downstream optimization",
                    candidate_increase,
                )
                shutil.copy2(model_path, output_path)
                passes = [
                    "graph_optimization_bloat_guard",
                    "original_serialization_preserved",
                ]
                selected_label = "original_copy_due_to_size_bloat"
            else:
                shutil.copy2(selected_path, output_path)
        else:
            shutil.copy2(selected_path, output_path)

        for path in (candidate_path, retry_path):
            if path.exists():
                path.unlink()

        return passes + [
            f"bloat_guard_threshold_percent:{self.max_size_increase_percent}",
            f"selected_artifact:{selected_label}",
        ]

    def _size_increase_percent(self, original_size, new_size):
        if original_size <= 0:
            return 0.0
        return ((new_size - original_size) / original_size) * 100

    def _ort_optimization_level(self, ort):
        levels = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }

        if self.optimization_level not in levels:
            raise GraphOptimizationError(
                f"Unsupported graph optimization level: {self.optimization_level}"
            )
        return levels[self.optimization_level]

    def _validate_graph(self, onnx, output_path):
        model = onnx.load(str(output_path))
        onnx.checker.check_model(model)
        return model

    def _import_onnx(self):
        try:
            import onnx

            return onnx
        except ImportError as exc:
            raise GraphOptimizationError(
                "onnx is required to validate optimized ONNX graphs"
            ) from exc

    def _model_size_bytes(self, model_path):
        return Path(model_path).stat().st_size

    def _build_stats(
        self,
        model_path,
        output_path,
        original_node_count,
        optimized_node_count,
        original_size,
        optimized_size,
        passes_applied,
        transformer_stats,
    ):
        node_reduction = original_node_count - optimized_node_count
        node_reduction_percent = 0.0
        if original_node_count > 0:
            node_reduction_percent = (node_reduction / original_node_count) * 100

        size_change_percent = 0.0
        if original_size > 0:
            size_change_percent = ((optimized_size - original_size) / original_size) * 100

        return {
            "status": "optimized",
            "technique": self.technique,
            "input_model": str(model_path),
            "optimized_graph_model": str(output_path),
            "original_node_count": original_node_count,
            "optimized_node_count": optimized_node_count,
            "node_count_reduction": node_reduction,
            "node_count_reduction_percent": round(node_reduction_percent, 2),
            "original_size_bytes": original_size,
            "optimized_size_bytes": optimized_size,
            "size_change_bytes": optimized_size - original_size,
            "size_change_percent": round(size_change_percent, 2),
            "optimization_passes_applied": passes_applied,
            "transformer_optimizer": transformer_stats,
        }
