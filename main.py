from pathlib import Path
import subprocess
import sys


def main():
    project_dir = Path(__file__).resolve().parent

    required_files = [
        "data/processed/bitbrains_daily_usage.csv",
        "data/processed/azure_usage_sample.csv",
        "data/processed/alibaba_usage_sample.csv",
    ]

    for relative_path in required_files:
        path = project_dir / relative_path

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

    scripts = [
        ("Bitbrains hybrid evaluation", "evaluate_hybrid_final.py"),
        ("Bitbrains prediction evaluation", "prediction.py"),
        ("Azure hybrid evaluation", "evaluate_azure.py"),
        ("Alibaba hybrid evaluation", "evaluate_alibaba.py"),
    ]

    for title, script in scripts:
        print(f"\n=== {title} ===", flush=True)

        subprocess.run(
            [sys.executable, str(project_dir / script)],
            cwd=project_dir,
            check=True,
        )

    print("\nMulti-provider pipeline completed successfully.", flush=True)


if __name__ == "__main__":
    main()
