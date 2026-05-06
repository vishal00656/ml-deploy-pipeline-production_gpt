
from pathlib import Path
import shutil

class ArtifactManager:
    def __init__(self):
        self.output = Path('artifacts')
        self.output.mkdir(exist_ok=True)

    def save(self, file_path):
        src = Path(file_path)
        dst = self.output / src.name
        shutil.copy(src, dst)
        return dst
