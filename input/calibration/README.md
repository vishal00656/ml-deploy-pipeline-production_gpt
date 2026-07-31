Place representative calibration images here for static INT8 quantization.

For YOLO/CNN Raspberry Pi deployments, use images that resemble expected
production inputs. The default preprocessing configuration is in
configs/calibration.yaml.

The GitHub Actions workflow generates temporary sample images automatically for
test runs. For production-quality quantization, replace those samples with a
small representative image set from your deployment domain.
