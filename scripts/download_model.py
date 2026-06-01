import argparse
from pathlib import Path


SUPPORTED_MODELS = {"resnet50", "mobilenetv2"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--output-dir", default="input")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model}.pt"

    if output_path.exists():
        print(f"Reusing existing model: {output_path}")
        return str(output_path)

    print(f"Generating model artifact: {output_path}")
    model = build_model(args.model)
    save_model(model, output_path)
    print(f"Saved model: {output_path}")
    return str(output_path)


def build_model(model_name):
    try:
        from torchvision import models

        if model_name == "resnet50":
            return models.resnet50(weights=None)
        if model_name == "mobilenetv2":
            return models.mobilenet_v2(weights=None)
    except Exception as exc:
        print(f"torchvision model generation unavailable: {exc}")
        print("Falling back to a small exportable CNN for workflow validation.")
        return fallback_cnn()

    raise ValueError(f"Unsupported model: {model_name}")


def fallback_cnn():
    import torch

    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 1000),
    )


def save_model(model, output_path):
    import torch

    model.eval()
    torch.save(model, output_path)


if __name__ == "__main__":
    main()

