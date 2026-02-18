from __future__ import annotations

import csv
from pathlib import Path


def read_rows(path: Path, delimiter: str) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Could not read header from {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def main() -> None:
    """
    Create `data/class5.csv` by combining:
    - `data/20.csv` (comma-separated, appears to be all ProdTaken=1)
    - `data/80.csv` (tab-separated, appears to be all ProdTaken=0)

    The combined file will contain both classes (0 and 1), which you need to train.
    """
    project_root = Path(__file__).resolve().parents[1]
    p20 = project_root / "data" / "20.csv"
    p80 = project_root / "data" / "80.csv"
    out = project_root / "data" / "class5.csv"

    if not p20.exists():
        raise FileNotFoundError(f"Missing: {p20}")
    if not p80.exists():
        raise FileNotFoundError(f"Missing: {p80}")

    header20, rows20 = read_rows(p20, delimiter=",")
    header80, rows80 = read_rows(p80, delimiter="\t")

    # Ensure headers match (same column names in same order)
    if header20 != header80:
        # If they have same set but different order, normalize to header20.
        if set(header20) != set(header80):
            raise ValueError(
                "The two files do not have the same columns.\n"
                f"20.csv columns: {header20}\n"
                f"80.csv columns: {header80}"
            )
        header = header20
    else:
        header = header20

    combined = rows20 + rows80
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter=",")
        writer.writeheader()
        for row in combined:
            writer.writerow({k: row.get(k, "") for k in header})

    print("Done.")
    print(f"Created: {out}")
    print(f"Rows from 20.csv: {len(rows20)}")
    print(f"Rows from 80.csv: {len(rows80)}")
    print(f"Total rows: {len(combined)}")


if __name__ == "__main__":
    main()

