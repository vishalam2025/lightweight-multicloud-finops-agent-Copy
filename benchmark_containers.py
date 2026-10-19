import csv
import json
import random
import statistics
import subprocess
from pathlib import Path

IMAGE = "finops-da3"

PROFILES = {
    "Small": 0.5,
    "Medium": 1.0,
    "Large": 2.0,
}

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_trial(profile):
    command = [
        "docker", "run", "--rm",
        f"--cpus={PROFILES[profile]}",
        "--memory=512m",
        IMAGE,
        "python", "benchmark_workload.py",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    measurement = json.loads(result.stdout.strip())

    if measurement["prime_counts"] != [33860, 33860]:
        raise ValueError("Unexpected workload result")

    return measurement["execution_time_sec"]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    # Resolve the tag once so every trial uses the same image.
    global IMAGE
    IMAGE = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    print("=== Warm-up runs ===")
    for profile in PROFILES:
        elapsed = run_trial(profile)
        print(f"{profile}: {elapsed:.4f} seconds (excluded)")

    raw_rows = []
    rng = random.Random(42)

    print("\n=== Measured trials ===")
    for trial in range(1, 6):
        order = list(PROFILES)
        rng.shuffle(order)

        for profile in order:
            elapsed = run_trial(profile)

            raw_rows.append({
                "profile": profile,
                "cpu_limit": PROFILES[profile],
                "memory_mb": 512,
                "trial": trial,
                "execution_time_sec": elapsed,
                "image_id": IMAGE,
            })

            # Save progress after every completed trial.
            write_csv(OUTPUT_DIR / "container_benchmark_raw.csv", raw_rows)

            print(f"Trial {trial}, {profile}: {elapsed:.4f} seconds")

    summary_rows = []

    for profile, cpu_limit in PROFILES.items():
        times = [
            row["execution_time_sec"]
            for row in raw_rows
            if row["profile"] == profile
        ]

        summary_rows.append({
            "profile": profile,
            "cpu_limit": cpu_limit,
            "memory_mb": 512,
            "trials": len(times),
            "median_sec": statistics.median(times),
            "mean_sec": statistics.mean(times),
            "std_sec": statistics.stdev(times),
            "min_sec": min(times),
            "max_sec": max(times),
        })

    write_csv(
        OUTPUT_DIR / "container_benchmark_summary.csv",
        summary_rows,
    )

    print("\n=== Summary ===")
    for row in summary_rows:
        print(
            f"{row['profile']}: "
            f"median={row['median_sec']:.4f}s, "
            f"std={row['std_sec']:.4f}s"
        )

    print("\nResults saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
