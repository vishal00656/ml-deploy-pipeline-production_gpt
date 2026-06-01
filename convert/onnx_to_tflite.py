from pathlib import Path
import json
import shutil
import subprocess
import sys

from core.logger import logger


class TFLiteConversionError(RuntimeError):
    """Raised when ONNX to TFLite conversion cannot be completed."""


def convert_onnx_to_tflite(input_onnx, output_tflite, quantization="fp16"):
    input_onnx = Path(input_onnx)
    output_tflite = Path(output_tflite)
    output_tflite.parent.mkdir(parents=True, exist_ok=True)
    saved_model_dir = output_tflite.with_suffix("").with_name(
        f"{output_tflite.stem}_saved_model"
    )

    logger.info(
        "Converting ONNX to TFLite for Android: %s -> %s (%s)",
        input_onnx,
        output_tflite,
        quantization,
    )

    if not input_onnx.exists():
        raise TFLiteConversionError(f"ONNX model not found: {input_onnx}")

    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)

    try:
        _run_onnx2tf(input_onnx, saved_model_dir)
        _run_tflite_converter(saved_model_dir, output_tflite, quantization)
    except Exception as exc:
        if output_tflite.exists():
            output_tflite.unlink()
        if isinstance(exc, TFLiteConversionError):
            raise
        raise TFLiteConversionError("Failed to convert ONNX model to TFLite") from exc
    finally:
        if saved_model_dir.exists():
            shutil.rmtree(saved_model_dir)

    if not output_tflite.exists():
        raise TFLiteConversionError(f"TFLite artifact was not created: {output_tflite}")

    report = {
        "status": "converted",
        "source_onnx": str(input_onnx),
        "tflite_model": str(output_tflite),
        "quantization": quantization,
        "runtime": "tflite",
        "deployment_target": "android",
        "artifact_size_bytes": output_tflite.stat().st_size,
        "artifact_size_mb": round(output_tflite.stat().st_size / (1024 * 1024), 4),
    }
    _write_report(report)
    logger.info("TFLite artifact generated for Android deployment: %s", output_tflite)
    return report


def _run_onnx2tf(input_onnx, saved_model_dir):
    onnx2tf_command = _find_onnx2tf_command()
    if onnx2tf_command is None:
        raise TFLiteConversionError(
            "onnx2tf command was not found. Install dependencies with: "
            "pip install -r requirements.txt"
        )

    command = [
        str(onnx2tf_command),
        "-i",
        str(input_onnx),
        "-o",
        str(saved_model_dir),
    ]
    _run_command(command, "onnx2tf conversion")


def _find_onnx2tf_command():
    command = shutil.which("onnx2tf")
    if command:
        return command
    scripts_dir = Path(sys.executable).parent
    for name in ("onnx2tf.exe", "onnx2tf"):
        candidate = scripts_dir / name
        if candidate.exists():
            return candidate
    return None


def _run_tflite_converter(saved_model_dir, output_tflite, quantization):
    script = (
        "from pathlib import Path\n"
        "import sys\n"
        "import tensorflow as tf\n"
        "saved_model = sys.argv[1]\n"
        "output_path = Path(sys.argv[2])\n"
        "quantization = sys.argv[3].lower()\n"
        "converter = tf.lite.TFLiteConverter.from_saved_model(saved_model)\n"
        "if quantization == 'fp16':\n"
        "    converter.optimizations = [tf.lite.Optimize.DEFAULT]\n"
        "    converter.target_spec.supported_types = [tf.float16]\n"
        "elif quantization in {'int8', 'static_int8'}:\n"
        "    converter.optimizations = [tf.lite.Optimize.DEFAULT]\n"
        "elif quantization in {'dynamic', 'dynamic_range'}:\n"
        "    converter.optimizations = [tf.lite.Optimize.DEFAULT]\n"
        "elif quantization in {'none', 'float32', 'graph'}:\n"
        "    pass\n"
        "else:\n"
        "    raise ValueError(f'Unsupported TFLite quantization: {quantization}')\n"
        "model = converter.convert()\n"
        "output_path.write_bytes(model)\n"
    )
    _run_command(
        [
            sys.executable,
            "-c",
            script,
            str(saved_model_dir),
            str(output_tflite),
            str(quantization),
        ],
        "TensorFlow Lite conversion",
    )


def _run_command(command, label):
    logger.info("Starting %s", label)
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        logger.info("%s stdout: %s", label, result.stdout.strip())
    if result.stderr:
        logger.info("%s stderr: %s", label, result.stderr.strip())
    if result.returncode != 0:
        raise TFLiteConversionError(
            f"{label} failed with exit code {result.returncode}"
        )


def _write_report(report):
    report_path = Path("reports/tflite_conversion.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
