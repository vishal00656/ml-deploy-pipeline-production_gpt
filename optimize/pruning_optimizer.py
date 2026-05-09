from pathlib import Path
import json

from core.logger import logger


class PruningOptimizationError(RuntimeError):
    """Raised when structured pruning cannot be completed."""


class PyTorchStructuredPruner:
    """Apply PyTorch structured pruning to Conv/Linear modules."""

    technique = "pytorch_structured_ln_pruning"

    def __init__(self, pruning_ratio, norm_order=2, dim=0, device="cpu"):
        self.pruning_ratio = float(pruning_ratio)
        self.norm_order = norm_order
        self.dim = dim
        self.device = device
        self._validate_pruning_ratio()

    def prune(self, model_path, output_path, report_path=None):
        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Running PyTorch structured pruning: %s -> %s, ratio %.4f",
            model_path,
            output_path,
            self.pruning_ratio,
        )

        try:
            torch = self._import_torch()
            prune = self._import_prune()

            model = self._load_model(torch, model_path)
            model.to(self.device)
            model.eval()

            original_stats = self._parameter_stats(torch, model)
            pruned_modules = self._apply_structured_pruning(prune, model)

            if not pruned_modules:
                logger.warning(
                    "No supported Conv/Linear modules were found for pruning"
                )

            pruned_stats = self._parameter_stats(torch, model)
            torch.save(model, output_path)

            report = self._build_report(
                model_path=model_path,
                output_path=output_path,
                original_stats=original_stats,
                pruned_stats=pruned_stats,
                pruned_modules=pruned_modules,
            )

            if report_path is not None:
                report_path = Path(report_path)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                with open(report_path, "w") as file:
                    json.dump(report, file, indent=2)
                logger.info("Pruning statistics report written: %s", report_path)

            logger.info(
                "Structured pruning completed: %.2f%% sparsity, %.2f%% "
                "estimated compression improvement",
                report["sparsity_percent"],
                report["estimated_compression_improvement_percent"],
            )
            return report
        except Exception as exc:
            if output_path.exists():
                output_path.unlink()
            logger.exception("PyTorch structured pruning failed: %s", exc)
            if isinstance(exc, PruningOptimizationError):
                raise
            raise PruningOptimizationError(
                f"Failed to prune PyTorch model '{model_path}'"
            ) from exc

    def _validate_pruning_ratio(self):
        if self.pruning_ratio < 0 or self.pruning_ratio >= 1:
            raise PruningOptimizationError(
                "pruning_ratio must be greater than or equal to 0 and less than 1"
            )

    def _import_torch(self):
        try:
            import torch

            return torch
        except ImportError as exc:
            raise PruningOptimizationError(
                "PyTorch is required for structured pruning"
            ) from exc

    def _import_prune(self):
        try:
            import torch.nn.utils.prune as prune

            return prune
        except ImportError as exc:
            raise PruningOptimizationError(
                "torch.nn.utils.prune is required for structured pruning"
            ) from exc

    def _load_model(self, torch, model_path):
        try:
            artifact = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            artifact = torch.load(model_path, map_location=self.device)

        if isinstance(artifact, torch.nn.Module):
            return artifact

        if isinstance(artifact, dict):
            for key in ("model", "ema"):
                candidate = artifact.get(key)
                if isinstance(candidate, torch.nn.Module):
                    return candidate

        raise PruningOptimizationError(
            "The PyTorch artifact must contain a torch.nn.Module. "
            "State-dict-only checkpoints need the original model class "
            "before structured pruning can be applied."
        )

    def _apply_structured_pruning(self, prune, model):
        pruned_modules = []
        for name, module in model.named_modules():
            if not self._is_supported_module(module):
                continue

            prune.ln_structured(
                module,
                name="weight",
                amount=self.pruning_ratio,
                n=self.norm_order,
                dim=self.dim,
            )
            prune.remove(module, "weight")

            pruned_modules.append(
                {
                    "name": name,
                    "type": module.__class__.__name__,
                    "parameter": "weight",
                    "ratio": self.pruning_ratio,
                    "norm_order": self.norm_order,
                    "dim": self.dim,
                }
            )

        return pruned_modules

    def _is_supported_module(self, module):
        torch = self._import_torch()
        return isinstance(
            module,
            (
                torch.nn.Conv1d,
                torch.nn.Conv2d,
                torch.nn.Conv3d,
                torch.nn.Linear,
            ),
        )

    def _parameter_stats(self, torch, model):
        total = 0
        nonzero = 0
        for parameter in model.parameters():
            total += parameter.numel()
            nonzero += int(torch.count_nonzero(parameter).item())

        zero = total - nonzero
        sparsity = 0.0
        if total > 0:
            sparsity = (zero / total) * 100

        return {
            "total_parameters": total,
            "nonzero_parameters": nonzero,
            "zero_parameters": zero,
            "sparsity_percent": round(sparsity, 2),
        }

    def _build_report(
        self,
        model_path,
        output_path,
        original_stats,
        pruned_stats,
        pruned_modules,
    ):
        parameter_reduction = (
            original_stats["nonzero_parameters"] - pruned_stats["nonzero_parameters"]
        )
        parameter_reduction_percent = 0.0
        if original_stats["nonzero_parameters"] > 0:
            parameter_reduction_percent = (
                parameter_reduction / original_stats["nonzero_parameters"]
            ) * 100

        return {
            "status": "pruned",
            "technique": self.technique,
            "input_model": str(model_path),
            "pruned_model": str(output_path),
            "pruning_ratio": self.pruning_ratio,
            "structured_pruning_dim": self.dim,
            "original": original_stats,
            "pruned": pruned_stats,
            "parameter_reduction": parameter_reduction,
            "parameter_reduction_percent": round(parameter_reduction_percent, 2),
            "sparsity_percent": pruned_stats["sparsity_percent"],
            "estimated_compression_improvement_percent": round(
                parameter_reduction_percent,
                2,
            ),
            "pruned_modules": pruned_modules,
        }
