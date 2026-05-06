
import yaml
from pathlib import Path

class ConfigManager:
    def __init__(self, base='configs'):
        self.base = Path(base)

    def load_hardware(self, target):
        path = self.base / 'hardware' / f'{target}.yaml'
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def load_pipeline(self):
        path = self.base / 'pipeline.yaml'
        with open(path, 'r') as f:
            return yaml.safe_load(f)
