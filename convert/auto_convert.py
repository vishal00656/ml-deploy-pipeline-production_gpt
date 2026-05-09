
from pathlib import Path
import shutil

from core.logger import logger
from convert.detect_format import detect_model_format
from convert.pytorch_to_onnx import export_pytorch_to_onnx
from optimize.pruning_optimizer import PruningOptimizationError, PyTorchStructuredPruner

class UniversalConverter:

    def convert_to_onnx(self, model_path, hardware_profile=None):
        model_path = Path(model_path)
        hardware_profile = hardware_profile or {}

        framework = detect_model_format(model_path)

        logger.info(f"Detected framework: {framework}")

        output_dir = Path("converted")
        output_dir.mkdir(exist_ok=True)

        output_model = output_dir / "model.onnx"

        if framework == "pytorch":
            export_input = self._maybe_prune_pytorch_model(
                model_path,
                hardware_profile,
            )
            return export_pytorch_to_onnx(export_input, output_model)

        if framework == "onnx":
            if model_path.resolve() != output_model.resolve():
                shutil.copy(model_path, output_model)
                logger.info(f"Copied existing ONNX model: {model_path.name}")
            else:
                logger.info(f"Using existing ONNX model in conversion cache: {model_path.name}")
            return str(output_model)

        logger.warning(
            "Real %s conversion is not implemented yet; using legacy copy "
            "fallback.",
            framework,
        )
        shutil.copy(model_path, output_model)

        logger.info(f"Converted {model_path.name} -> ONNX")

        return str(output_model)

    def _maybe_prune_pytorch_model(self, model_path, hardware_profile):
        pruning_ratio = float(hardware_profile.get("pruning_ratio", 0.0))
        pruning_enabled = hardware_profile.get(
            "pruning_enabled",
            hardware_profile.get("pruning", False),
        )

        if not pruning_enabled or pruning_ratio <= 0:
            logger.info("Structured pruning disabled for PyTorch export")
            return model_path

        output_dir = Path("output")
        reports_dir = Path("reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        pruned_model = output_dir / f"{model_path.stem}_pruned.pt"
        report_path = reports_dir / "pruning_statistics.json"

        pruner = PyTorchStructuredPruner(
            pruning_ratio=pruning_ratio,
            norm_order=hardware_profile.get("pruning_norm_order", 2),
            dim=hardware_profile.get("pruning_dim", 0),
        )
        try:
            pruner.prune(model_path, pruned_model, report_path=report_path)
            return pruned_model
        except PruningOptimizationError as exc:
            logger.warning(
                "Structured pruning was selected but could not be applied before "
                "ONNX export: %s. Continuing with the original model.",
                exc,
            )
            return model_path
