from pathlib import Path
import re
import shutil

from core.logger import logger


ARTIFACT_SUFFIXES = {".onnx", ".tflite", ".pt", ".pth", ".json", ".md"}
RESERVED_ARTIFACT_DOCS = {"README.md"}
INTERMEDIATE_MARKERS = (
    ".candidate",
    ".extended_candidate",
    "_fp32_normalized",
    "_graph",
    "_graph_optimized",
    "_int8",
    "_fp16",
    "_static_int8",
    "static_int8_smoke",
    "final_report_empty_smoke",
)


def cleanup_output_directory(output_dir="output", preserve_paths=None):
    """Remove stale artifacts before a new pipeline run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preserve = _resolve_preserve_paths(output_dir, preserve_paths)

    logger.info("Artifact cleanup started: %s", output_dir)
    for path in sorted(output_dir.iterdir(), key=lambda item: str(item)):
        if path.name in RESERVED_ARTIFACT_DOCS:
            logger.info("Preserving artifact directory documentation: %s", path)
            continue
        if _is_preserved(path, preserve):
            logger.info("Preserving artifact during cleanup: %s", path)
            continue
        if path.name == "optimization_stats.json":
            logger.info("Preserving optimization stats: %s", path)
            continue
        if path.is_dir() or path.suffix.lower() in ARTIFACT_SUFFIXES:
            _delete_path(path, output_dir)


def cleanup_intermediate_artifacts(
    output_dir="output",
    final_artifact=None,
    preserve_paths=None,
):
    """Delete intermediate files after a successful optimization run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preserve = _resolve_preserve_paths(
        output_dir,
        [final_artifact, output_dir / "optimization_stats.json", *(preserve_paths or [])],
    )

    logger.info("Artifact intermediate cleanup started: %s", output_dir)
    for path in sorted(output_dir.iterdir(), key=lambda item: str(item)):
        if path.name in RESERVED_ARTIFACT_DOCS:
            logger.info("Preserving artifact directory documentation: %s", path)
            continue
        if _is_preserved(path, preserve):
            logger.info("Preserving final artifact: %s", path)
            continue
        if path.is_dir():
            _delete_path(path, output_dir)
            continue
        if _is_intermediate_artifact(path):
            _delete_path(path, output_dir)


def generate_final_artifact_name(model_name, hardware, optimization, suffix=".onnx"):
    model = _slug(model_name or "model")
    target = _slug(hardware or "hardware")
    strategy = _slug(_optimization_label(optimization))
    extension = suffix if str(suffix).startswith(".") else f".{suffix}"
    return f"{model}_{target}_{strategy}{extension}"


def preserve_required_outputs(
    output_dir="output",
    final_artifact=None,
    pruned_model=None,
):
    output_dir = Path(output_dir)
    preserve = [output_dir / "optimization_stats.json"]
    if final_artifact is not None:
        preserve.append(Path(final_artifact))
    if pruned_model is not None and Path(pruned_model).exists():
        preserve.append(Path(pruned_model))
    cleanup_intermediate_artifacts(
        output_dir=output_dir,
        final_artifact=final_artifact,
        preserve_paths=preserve,
    )
    return preserve


def _optimization_label(optimization):
    value = str(optimization or "optimized").lower()
    if value == "int8":
        return "dynamic_int8"
    return value


def _is_intermediate_artifact(path):
    name = path.name.lower()
    if path.name == "optimization_stats.json":
        return False
    return path.suffix.lower() in ARTIFACT_SUFFIXES or any(
        marker in name for marker in INTERMEDIATE_MARKERS
    )


def _delete_path(path, output_dir):
    if not _is_inside(path, output_dir):
        logger.warning("Skipping deletion outside output directory: %s", path)
        return
    logger.info("Deleting intermediate artifact: %s", path)
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _resolve_preserve_paths(output_dir, preserve_paths):
    preserve = set()
    for item in preserve_paths or []:
        if item is None:
            continue
        path = Path(item)
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if _is_inside(resolved, output_dir) or resolved.exists():
            preserve.add(resolved)
    return preserve


def _is_preserved(path, preserve):
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return any(resolved == item or item in resolved.parents for item in preserve)


def _is_inside(path, parent):
    try:
        resolved = Path(path).resolve()
        root = Path(parent).resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents


def _slug(value):
    text = Path(str(value)).stem if Path(str(value)).suffix else str(value)
    text = text.strip().lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "artifact"


class ArtifactManager:
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def cleanup_output_directory(self, preserve_paths=None):
        return cleanup_output_directory(self.output_dir, preserve_paths)

    def cleanup_intermediate_artifacts(self, final_artifact=None, preserve_paths=None):
        return cleanup_intermediate_artifacts(
            self.output_dir,
            final_artifact=final_artifact,
            preserve_paths=preserve_paths,
        )

    def generate_final_artifact_name(self, model_name, hardware, optimization, suffix=".onnx"):
        return generate_final_artifact_name(model_name, hardware, optimization, suffix)

    def preserve_required_outputs(self, final_artifact=None, pruned_model=None):
        return preserve_required_outputs(
            self.output_dir,
            final_artifact=final_artifact,
            pruned_model=pruned_model,
        )
