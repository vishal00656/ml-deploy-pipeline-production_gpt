# Contributing

Thank you for helping improve Universal ML Deployment Pipeline.

## Development Setup

Use Python 3.11 and install the pinned dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

## Contribution Guidelines

- Preserve existing pipeline behavior unless the pull request explicitly changes
  it and includes validation evidence.
- Do not commit generated models, datasets, calibration images, benchmark
  artifacts, logs, virtual environments, or downloaded model caches.
- Keep source changes focused and consistent with the existing module layout.
- Add or update documentation when user-facing behavior changes.
- For optimization, quantization, benchmarking, or deployment changes, include
  before/after evidence from representative models where practical.

## Validation

Before opening a pull request, run:

```bash
python -m compileall -q benchmark calibration convert core decision_engine deploy optimize pipeline reporting scripts tests main.py
python scripts/download_model.py --help
```

If your change affects runtime behavior, run a representative optimization
command and include the generated summary in the pull request description.

## Pull Requests

Please include:

- A concise description of the change.
- The motivation and expected impact.
- Validation commands and results.
- Any compatibility notes for models, hardware profiles, or runtimes.
