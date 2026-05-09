from pathlib import Path


class CalibrationDatasetError(RuntimeError):
    """Raised when calibration samples cannot be loaded."""


class CalibrationDatasetLoader:
    IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

    def __init__(self, image_dir, sample_limit=None):
        self.image_dir = Path(image_dir)
        self.sample_limit = sample_limit

    def load_image_paths(self):
        if not self.image_dir.exists():
            raise CalibrationDatasetError(
                f"Calibration image directory not found: {self.image_dir}"
            )
        if not self.image_dir.is_dir():
            raise CalibrationDatasetError(
                f"Calibration image path is not a directory: {self.image_dir}"
            )

        image_paths = sorted(
            path
            for path in self.image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.IMAGE_EXTENSIONS
        )
        if self.sample_limit is not None:
            image_paths = image_paths[: int(self.sample_limit)]
        if not image_paths:
            raise CalibrationDatasetError(
                "No calibration images found. Add representative images to "
                f"{self.image_dir} or update configs/calibration.yaml."
            )
        return image_paths

