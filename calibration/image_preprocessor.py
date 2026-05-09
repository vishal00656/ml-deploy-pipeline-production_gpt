from pathlib import Path

import numpy as np
from PIL import Image


class ImagePreprocessor:
    def __init__(
        self,
        image_size=(640, 640),
        normalize=True,
        mean=None,
        std=None,
        scale=255.0,
        color_mode="RGB",
        layout="auto",
    ):
        self.image_size = tuple(image_size)
        self.normalize = bool(normalize)
        self.mean = np.asarray(mean or [0.0, 0.0, 0.0], dtype=np.float32)
        self.std = np.asarray(std or [1.0, 1.0, 1.0], dtype=np.float32)
        self.scale = float(scale)
        self.color_mode = color_mode
        self.layout = layout

    def preprocess(self, image_path, input_shape=None, input_type="tensor(float)"):
        image = Image.open(Path(image_path)).convert(self.color_mode)
        width, height = self._target_size(input_shape)
        image = image.resize((width, height), Image.BILINEAR)

        array = np.asarray(image, dtype=np.float32)
        if self.normalize:
            array = array / self.scale
            array = (array - self.mean) / self.std

        layout = self._layout(input_shape)
        if layout == "NCHW":
            array = np.transpose(array, (2, 0, 1))

        array = np.expand_dims(array, axis=0)
        return array.astype(self._numpy_dtype(input_type), copy=False)

    def zeros(self, input_shape, input_type="tensor(float)"):
        shape = self._concrete_shape(input_shape)
        return np.zeros(shape, dtype=self._numpy_dtype(input_type))

    def _target_size(self, input_shape):
        if input_shape and len(input_shape) == 4:
            concrete = self._concrete_shape(input_shape)
            layout = self._layout(input_shape)
            if layout == "NCHW":
                return concrete[3], concrete[2]
            return concrete[2], concrete[1]
        return self.image_size[1], self.image_size[0]

    def _layout(self, input_shape):
        if self.layout != "auto":
            return self.layout.upper()
        if input_shape and len(input_shape) == 4:
            concrete = self._concrete_shape(input_shape)
            if concrete[1] in (1, 3):
                return "NCHW"
            if concrete[3] in (1, 3):
                return "NHWC"
        return "NCHW"

    def _concrete_shape(self, shape):
        if not shape:
            return [1, 3, self.image_size[0], self.image_size[1]]

        concrete = []
        for index, dim in enumerate(shape):
            if isinstance(dim, int) and dim > 0:
                concrete.append(dim)
            elif index == 0:
                concrete.append(1)
            elif len(shape) == 4 and index in (2, 3):
                concrete.append(self.image_size[0 if index == 2 else 1])
            elif len(shape) == 4 and index == 1:
                concrete.append(3)
            else:
                concrete.append(1)
        return concrete

    def _numpy_dtype(self, input_type):
        mapping = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(uint8)": np.uint8,
            "tensor(int8)": np.int8,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
        }
        return mapping.get(input_type, np.float32)

