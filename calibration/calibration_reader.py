from pathlib import Path

from onnxruntime.quantization import CalibrationDataReader

from calibration.dataset_loader import CalibrationDatasetLoader
from calibration.image_preprocessor import ImagePreprocessor


class OnnxImageCalibrationDataReader(CalibrationDataReader):
    """Feeds representative image tensors into ONNX Runtime calibration."""

    def __init__(self, model_path, calibration_config):
        self.model_path = Path(model_path)
        self.config = calibration_config or {}
        self.input_metadata = self._load_input_metadata()
        self.image_input = self._select_image_input()
        self.preprocessor = ImagePreprocessor(
            image_size=tuple(self.config.get("image_size", [640, 640])),
            normalize=self.config.get("normalize", True),
            mean=self.config.get("mean", [0.0, 0.0, 0.0]),
            std=self.config.get("std", [1.0, 1.0, 1.0]),
            scale=self.config.get("scale", 255.0),
            color_mode=self.config.get("color_mode", "RGB"),
            layout=self.config.get("layout", "auto"),
        )
        loader = CalibrationDatasetLoader(
            self.config.get("calibration_image_path", "input/calibration"),
            sample_limit=self.config.get("calibration_sample_limit", 64),
        )
        self.image_paths = loader.load_image_paths()
        self.batch_size = max(1, int(self.config.get("batch_size", 1)))
        self._cursor = 0
        self.samples_yielded = 0

    def get_next(self):
        if self._cursor >= len(self.image_paths):
            return None

        batch_paths = self.image_paths[self._cursor : self._cursor + self.batch_size]
        self._cursor += len(batch_paths)
        self.samples_yielded += len(batch_paths)

        inputs = {}
        for item in self.input_metadata:
            if item["name"] == self.image_input["name"]:
                tensors = [
                    self.preprocessor.preprocess(
                        path,
                        input_shape=item["shape"],
                        input_type=item["type"],
                    )
                    for path in batch_paths
                ]
                inputs[item["name"]] = self._concat(tensors)
            else:
                inputs[item["name"]] = self.preprocessor.zeros(
                    item["shape"],
                    input_type=item["type"],
                )
        return inputs

    def rewind(self):
        self._cursor = 0
        self.samples_yielded = 0

    def statistics(self):
        return {
            "model": str(self.model_path),
            "calibration_image_path": self.config.get(
                "calibration_image_path",
                "input/calibration",
            ),
            "available_image_count": len(self.image_paths),
            "samples_used": self.samples_yielded,
            "batch_size": self.batch_size,
            "image_size": self.config.get("image_size", [640, 640]),
            "normalize": self.config.get("normalize", True),
            "mean": self.config.get("mean", [0.0, 0.0, 0.0]),
            "std": self.config.get("std", [1.0, 1.0, 1.0]),
            "scale": self.config.get("scale", 255.0),
            "input_name": self.image_input["name"],
            "input_shape": self.image_input["shape"],
            "input_type": self.image_input["type"],
        }

    def _concat(self, tensors):
        import numpy as np

        if len(tensors) == 1:
            return tensors[0]
        return np.concatenate(tensors, axis=0)

    def _load_input_metadata(self):
        import onnx

        model = onnx.load(str(self.model_path), load_external_data=False)
        initializer_names = {item.name for item in model.graph.initializer}
        inputs = []
        for value_info in model.graph.input:
            if value_info.name in initializer_names:
                continue
            tensor_type = value_info.type.tensor_type
            shape = []
            for dim in tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(None)
            inputs.append(
                {
                    "name": value_info.name,
                    "shape": shape,
                    "type": self._onnx_type_name(onnx, tensor_type.elem_type),
                }
            )
        if not inputs:
            raise RuntimeError(f"No runtime inputs found in ONNX model: {self.model_path}")
        return inputs

    def _select_image_input(self):
        for item in self.input_metadata:
            shape = item["shape"]
            if len(shape) == 4:
                return item
        return self.input_metadata[0]

    def _onnx_type_name(self, onnx, elem_type):
        mapping = {
            onnx.TensorProto.FLOAT: "tensor(float)",
            onnx.TensorProto.FLOAT16: "tensor(float16)",
            onnx.TensorProto.DOUBLE: "tensor(double)",
            onnx.TensorProto.UINT8: "tensor(uint8)",
            onnx.TensorProto.INT8: "tensor(int8)",
            onnx.TensorProto.INT32: "tensor(int32)",
            onnx.TensorProto.INT64: "tensor(int64)",
        }
        return mapping.get(elem_type, "tensor(float)")
