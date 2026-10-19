from pathlib import Path
from io import StringIO
import tarfile
import pandas as pd

archive = Path("data/raw/alibaba/machine_usage.tar.gz")

with tarfile.open(archive, mode="r|gz") as tar:
    for member in tar:
        if (
            member.isfile()
            and Path(member.name).name == "machine_usage.csv"
        ):
            print("File inside archive:", member.name)

            stream = tar.extractfile(member)

            if stream is None:
                raise ValueError("Could not open the CSV member")

            with stream:
                lines = []

                for _ in range(5):
                    line = stream.readline()

                    if not line:
                        break

                    lines.append(line.decode("utf-8"))

            sample = pd.read_csv(
                StringIO("".join(lines)),
                header=None,
            )

            print("\n=== First five records ===")
            print(sample.to_string(index=False, header=False))
            print("\nNumber of columns:", sample.shape[1])
            break
    else:
        raise FileNotFoundError(
            "machine_usage.csv was not found inside the archive"
        )
