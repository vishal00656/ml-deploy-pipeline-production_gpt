from pathlib import Path
from inspect import signature
from typing import Any

from core.logger import logger


DEFAULT_DUMMY_INPUT_SHAPE = (1, 3, 224, 224)
DEFAULT_INPUT_NAME = "input"


class PyTorchOnnxExportError(RuntimeError):
    """Raised when a PyTorch model cannot be exported to ONNX."""


class PyTorchOnnxExporter:
    """Export PyTorch nn.Module artifacts to validated ONNX graphs."""

    def __init__(
        self,
        input_shape=DEFAULT_DUMMY_INPUT_SHAPE,
        opset_version=17,
        dynamic_batch=True,
        device="cpu",
    ):
        self.input_shape = tuple(input_shape)
        self.opset_version = opset_version
        self.dynamic_batch = dynamic_batch
        self.device = device

    def export(self, model_path, output_path):
        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Exporting PyTorch model to ONNX: "
            f"{model_path} -> {output_path}"
        )

        try:
            torch = self._import_torch()
            onnx = self._import_onnx()

            model = self._load_model(torch, model_path)
            model.to(self.device)
            model.eval()

            model_dtype = self._detect_model_dtype(torch, model)
            logger.info("Detected PyTorch model parameter dtype: %s", model_dtype)

            dummy_input = self._create_dummy_input(torch, model_dtype)
            output_names = self._infer_output_names(torch, model, dummy_input)
            dynamic_axes = self._build_dynamic_axes(output_names)

            logger.info(
                "Using dummy input shape %s, opset %s, dynamic batch axes: %s",
                self.input_shape,
                self.opset_version,
                self.dynamic_batch,
            )

            with torch.no_grad():
                export_kwargs = self._build_export_kwargs(
                    output_path,
                    output_names,
                    dynamic_axes,
                )
                torch.onnx.export(model, dummy_input, **export_kwargs)

            self._validate_onnx_graph(onnx, output_path)
            logger.info("PyTorch ONNX export completed: %s", output_path)
            return str(output_path)
        except Exception as exc:
            if output_path.exists():
                output_path.unlink()
            logger.exception("PyTorch to ONNX export failed: %s", exc)
            if isinstance(exc, PyTorchOnnxExportError):
                raise
            raise PyTorchOnnxExportError(
                f"Failed to export PyTorch model '{model_path}' to ONNX"
            ) from exc

    def _import_torch(self):
        try:
            import torch

            return torch
        except ImportError as exc:
            raise PyTorchOnnxExportError(
                "PyTorch is required for .pt/.pth ONNX export"
            ) from exc

    def _import_onnx(self):
        try:
            import onnx

            return onnx
        except ImportError as exc:
            raise PyTorchOnnxExportError(
                "onnx is required to validate exported ONNX graphs"
            ) from exc

    def _build_export_kwargs(self, output_path, output_names, dynamic_axes):
        export_kwargs = {
            "f": str(output_path),
            "export_params": True,
            "opset_version": self.opset_version,
            "do_constant_folding": True,
            "input_names": [DEFAULT_INPUT_NAME],
            "output_names": output_names,
            "dynamic_axes": dynamic_axes,
        }

        if "dynamo" in signature(self._import_torch().onnx.export).parameters:
            export_kwargs["dynamo"] = False

        return export_kwargs

    def _detect_model_dtype(self, torch, model):
        for parameter in model.parameters():
            if parameter.is_floating_point():
                return parameter.dtype

        for buffer in model.buffers():
            if buffer.is_floating_point():
                return buffer.dtype

        logger.warning(
            "No floating point parameters or buffers found; defaulting dummy "
            "input dtype to torch.float32"
        )
        return torch.float32

    def _create_dummy_input(self, torch, model_dtype):
        if model_dtype not in (torch.float16, torch.float32):
            logger.warning(
                "Detected dtype %s is not an explicitly supported export input "
                "dtype; defaulting dummy input to torch.float32",
                model_dtype,
            )
            model_dtype = torch.float32

        return torch.randn(
            *self.input_shape,
            device=self.device,
            dtype=model_dtype,
        )

    def _load_model(self, torch, model_path):
        standard_error = None
        try:
            artifact = self._torch_load(torch, model_path)
            model = self._extract_torch_module(torch, artifact)
            if model is not None:
                logger.info("Loaded PyTorch model with standard torch.load")
                return model

            if self._looks_like_ultralytics_artifact(artifact, model_path):
                logger.info(
                    "Checkpoint appears to be YOLO/Ultralytics; trying "
                    "Ultralytics loader fallback"
                )
                return self._load_ultralytics_model(torch, model_path)
        except Exception as exc:
            standard_error = exc
            logger.warning(
                "Standard torch.load model loading failed for %s: %s",
                model_path,
                exc,
            )
            logger.info("Trying Ultralytics loader fallback for %s", model_path)
            return self._load_ultralytics_model(torch, model_path)

        if standard_error is not None:
            raise PyTorchOnnxExportError(
                "Standard PyTorch loading failed and Ultralytics fallback was "
                f"not applicable for '{model_path}'"
            ) from standard_error

        raise PyTorchOnnxExportError(
            "The PyTorch artifact must contain a torch.nn.Module. "
            "State-dict-only checkpoints need the original model class "
            "to be reconstructed before ONNX export."
        )

    def _torch_load(self, torch, model_path):
        try:
            return torch.load(
                model_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            return torch.load(model_path, map_location=self.device)

    def _extract_torch_module(self, torch, artifact):
        if isinstance(artifact, torch.nn.Module):
            return artifact

        for key in ("model", "ema"):
            candidate = self._get_mapping_value(artifact, key)
            if isinstance(candidate, torch.nn.Module):
                return candidate

        return None

    def _load_ultralytics_model(self, torch, model_path):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise PyTorchOnnxExportError(
                "Ultralytics is required to load YOLOv5/YOLOv8 checkpoints. "
                "Install the 'ultralytics' package or provide an exportable "
                "torch.nn.Module checkpoint."
            ) from exc

        try:
            yolo_model = YOLO(str(model_path))
            model = getattr(yolo_model, "model", None)
            if not isinstance(model, torch.nn.Module):
                raise PyTorchOnnxExportError(
                    "Ultralytics loader did not return an exportable "
                    "torch.nn.Module"
                )

            model = self._prepare_ultralytics_module(model)
            logger.info(
                "Loaded YOLO/Ultralytics checkpoint with Ultralytics APIs"
            )
            return model
        except Exception as exc:
            if isinstance(exc, PyTorchOnnxExportError):
                raise
            raise PyTorchOnnxExportError(
                f"Failed to load YOLO/Ultralytics checkpoint '{model_path}'"
            ) from exc

    def _prepare_ultralytics_module(self, model):
        model.eval()

        if hasattr(model, "fuse"):
            try:
                model = model.fuse()
                logger.info("Fused Ultralytics model layers before ONNX export")
            except Exception as exc:
                logger.warning("Ultralytics layer fusion skipped: %s", exc)

        for attribute, value in (
            ("export", True),
            ("dynamic", self.dynamic_batch),
            ("format", "onnx"),
        ):
            if hasattr(model, attribute):
                try:
                    setattr(model, attribute, value)
                except Exception as exc:
                    logger.debug(
                        "Could not set Ultralytics export attribute %s: %s",
                        attribute,
                        exc,
                    )

        return model

    def _get_mapping_value(self, artifact: Any, key: str):
        if isinstance(artifact, dict):
            return artifact.get(key)
        return None

    def _looks_like_ultralytics_artifact(self, artifact: Any, model_path):
        if self._is_yolo_checkpoint_path(model_path):
            return True

        if not isinstance(artifact, dict):
            return False

        yolo_keys = {
            "model",
            "ema",
            "train_args",
            "yaml",
            "names",
            "stride",
            "nc",
        }
        return bool(yolo_keys.intersection(artifact.keys()))

    def _is_yolo_checkpoint_path(self, model_path):
        name = Path(model_path).name.lower()
        return (
            "yolo" in name
            or "ultralytics" in name
            or name.startswith(("best", "last"))
        )

    def _infer_output_names(self, torch, model, dummy_input):
        with torch.no_grad():
            outputs = model(dummy_input)

        output_count = max(1, len(self._flatten_outputs(torch, outputs)))
        if output_count == 1:
            return ["output"]
        return [f"output_{index}" for index in range(output_count)]

    def _flatten_outputs(self, torch, value):
        if torch.is_tensor(value):
            return [value]
        if isinstance(value, (list, tuple)):
            flattened = []
            for item in value:
                flattened.extend(self._flatten_outputs(torch, item))
            return flattened
        if isinstance(value, dict):
            flattened = []
            for item in value.values():
                flattened.extend(self._flatten_outputs(torch, item))
            return flattened
        return []

    def _build_dynamic_axes(self, output_names):
        if not self.dynamic_batch:
            return None

        dynamic_axes = {DEFAULT_INPUT_NAME: {0: "batch"}}
        for output_name in output_names:
            dynamic_axes[output_name] = {0: "batch"}
        return dynamic_axes

    def _validate_onnx_graph(self, onnx, output_path):
        model = onnx.load(str(output_path))
        onnx.checker.check_model(model)

        if not model.graph.node:
            raise PyTorchOnnxExportError(
                f"Exported ONNX graph has no nodes: {output_path}"
            )


def export_pytorch_to_onnx(
    model_path,
    output_path,
    input_shape=DEFAULT_DUMMY_INPUT_SHAPE,
    opset_version=17,
    dynamic_batch=True,
):
    exporter = PyTorchOnnxExporter(
        input_shape=input_shape,
        opset_version=opset_version,
        dynamic_batch=dynamic_batch,
    )
    return exporter.export(model_path, output_path)
