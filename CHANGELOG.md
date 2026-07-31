# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-07-31

### Added

- Public-release documentation with project overview, architecture, usage,
  GitHub Actions, supported hardware, supported formats, citation, and release
  placeholders.
- MIT License.
- Contribution guidelines.
- Placeholder README files for generated artifact directories.

### Changed

- Pinned direct Python dependencies in `requirements.txt` for reproducible v1.0
  installs.
- Expanded `.gitignore` for Python, ML artifacts, runtime output, datasets,
  logs, temporary files, benchmark artifacts, and downloaded model caches.

### Removed

- Generated model binaries and runtime logs from version control.
- Duplicate `input/README.txt` in favor of `input/README.md`.

### Notes

- Optimization algorithms, decision-engine behavior, benchmarking logic,
  quantization code, deployment flow, and GitHub Actions behavior are unchanged
  for this release-prep update.
