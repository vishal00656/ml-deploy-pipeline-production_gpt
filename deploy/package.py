
from pathlib import Path
from zipfile import ZipFile

def package():
    output_zip = Path("deployment_bundle.zip")

    with ZipFile(output_zip, "w") as z:
        for folder in ["output", "reports", "configs"]:
            path = Path(folder)
            if path.exists():
                for file in path.rglob("*"):
                    if file.is_file():
                        z.write(file)

    print(f"Deployment package created: {output_zip}")

if __name__ == "__main__":
    package()
