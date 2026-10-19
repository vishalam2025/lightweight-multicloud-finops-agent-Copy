from pathlib import Path
import pandas as pd

folder = Path("data/raw/azure")

files = sorted(folder.glob("*vm_cpu_readings*.csv.gz"))

if not files:
    raise FileNotFoundError("Azure CPU readings file not found")

print("=== Schema ===")
print((folder / "schema.csv").read_text(encoding="utf-8-sig"))

print("\n=== First five CPU records ===")
sample = pd.read_csv(
    files[0],
    compression="gzip",
    header=None,
    nrows=5,
)

print(sample.to_string(index=False, header=False))
print("\nNumber of columns:", sample.shape[1])
