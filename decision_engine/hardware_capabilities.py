from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class HardwareProfile:
    target: str
    name: str
    fp16_support: bool
    int8_support: bool
    tensorrt_support: bool
    tflite_preference: bool
    tinyml_compatibility: bool
    gpu_available: bool
    memory_constraint_mb: int
    preferred_optimization_styles: List[str] = field(default_factory=list)
    runtime_preferences: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def to_legacy_profile(self):
        return {
            "target": self.target,
            "name": self.name,
            "format": self.runtime_preferences[0]
            if self.runtime_preferences
            else "onnx",
            "quantization": "fp16"
            if self.fp16_support and self.gpu_available
            else "int8",
            "graph_optimization": True,
            "graph_optimization_level": "all",
            "pruning_enabled": "pruning" in self.preferred_optimization_styles,
            "pruning_ratio": 0.2
            if "pruning" in self.preferred_optimization_styles
            else 0.0,
        }


SUPPORTED_HARDWARE: Dict[str, HardwareProfile] = {
    "raspberry_pi": HardwareProfile(
        target="raspberry_pi",
        name="Raspberry Pi CPU",
        fp16_support=False,
        int8_support=True,
        tensorrt_support=False,
        tflite_preference=True,
        tinyml_compatibility=False,
        gpu_available=False,
        memory_constraint_mb=1024,
        preferred_optimization_styles=["static_int8", "pruning", "graph"],
        runtime_preferences=["tflite", "onnxruntime"],
        notes=["CPU-bound ARM target; static INT8 usually beats weight-only quantization."],
    ),
    "jetson": HardwareProfile(
        target="jetson",
        name="NVIDIA Jetson",
        fp16_support=True,
        int8_support=True,
        tensorrt_support=True,
        tflite_preference=False,
        tinyml_compatibility=False,
        gpu_available=True,
        memory_constraint_mb=4096,
        preferred_optimization_styles=["fp16", "tensorrt", "graph"],
        runtime_preferences=["tensorrt", "onnxruntime"],
        notes=["CUDA/TensorRT target; FP16 is a strong default for CNNs."],
    ),
    "android": HardwareProfile(
        target="android",
        name="Android",
        fp16_support=True,
        int8_support=True,
        tensorrt_support=False,
        tflite_preference=True,
        tinyml_compatibility=False,
        gpu_available=True,
        memory_constraint_mb=2048,
        preferred_optimization_styles=["static_int8", "fp16", "graph"],
        runtime_preferences=["tflite", "onnxruntime"],
        notes=["Mobile target; TFLite is preferred for deployment packaging."],
    ),
    "esp32": HardwareProfile(
        target="esp32",
        name="ESP32 Microcontroller",
        fp16_support=False,
        int8_support=True,
        tensorrt_support=False,
        tflite_preference=True,
        tinyml_compatibility=True,
        gpu_available=False,
        memory_constraint_mb=1,
        preferred_optimization_styles=["tinyml", "static_int8", "pruning"],
        runtime_preferences=["tflite_micro"],
        notes=["Microcontroller target with severe SRAM and flash constraints."],
    ),
    "cpu": HardwareProfile(
        target="cpu",
        name="Generic CPU",
        fp16_support=False,
        int8_support=True,
        tensorrt_support=False,
        tflite_preference=False,
        tinyml_compatibility=False,
        gpu_available=False,
        memory_constraint_mb=8192,
        preferred_optimization_styles=["dynamic_int8", "graph"],
        runtime_preferences=["onnxruntime"],
        notes=["General CPU target; dynamic INT8 is useful for MatMul/Linear-heavy models."],
    ),
    "edge_ai": HardwareProfile(
        target="edge_ai",
        name="Edge AI Accelerator",
        fp16_support=True,
        int8_support=True,
        tensorrt_support=False,
        tflite_preference=True,
        tinyml_compatibility=False,
        gpu_available=True,
        memory_constraint_mb=2048,
        preferred_optimization_styles=["static_int8", "fp16", "graph"],
        runtime_preferences=["tflite", "onnxruntime"],
        notes=["Mixed accelerator target; static INT8 is preferred when calibration exists."],
    ),
    "tinyml": HardwareProfile(
        target="tinyml",
        name="TinyML Class Device",
        fp16_support=False,
        int8_support=True,
        tensorrt_support=False,
        tflite_preference=True,
        tinyml_compatibility=True,
        gpu_available=False,
        memory_constraint_mb=2,
        preferred_optimization_styles=["tinyml", "static_int8", "pruning"],
        runtime_preferences=["tflite_micro"],
        notes=["TinyML deployment requires very small models and simple operators."],
    ),
}


class HardwareCapabilities:
    def get_profile(self, target: str, legacy_profile: Optional[dict] = None):
        key = (target or "").lower()
        if key not in SUPPORTED_HARDWARE:
            supported = ", ".join(sorted(SUPPORTED_HARDWARE))
            raise ValueError(f"Unsupported hardware target '{target}'. Supported: {supported}")

        profile = SUPPORTED_HARDWARE[key]
        if not legacy_profile:
            return profile

        profile_dict = profile.to_dict()
        if "name" in legacy_profile:
            profile_dict["name"] = legacy_profile["name"]
        if "quantization" in legacy_profile and legacy_profile["quantization"]:
            quantization = str(legacy_profile["quantization"]).lower()
            if quantization not in profile_dict["preferred_optimization_styles"]:
                profile_dict["preferred_optimization_styles"] = [
                    quantization,
                    *profile_dict["preferred_optimization_styles"],
                ]
        return HardwareProfile(**profile_dict)

