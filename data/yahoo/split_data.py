"""Split sp500_20y_daily.csv into chronological train / validation / test sets.

The split is chronological (no shuffling) so that no future information leaks
into training. Run once and commit the outputs so everyone trains, tunes and
evaluates on exactly the same data:

    python data/yahoo/split_data.py

Uses only the standard library; rows are copied verbatim from the source file.
"""
import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
SOURCE = DATA_DIR / "sp500_20y_daily.csv"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# Test gets the remainder (0.15).


def main() -> None:
    with open(SOURCE, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Drop incomplete rows, sort by date (ISO format sorts lexicographically).
        rows = sorted((r for r in reader if r and all(r)), key=lambda r: r[0])

    n = len(rows)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    parts = {
        "train": rows[:train_end],
        "val": rows[train_end:val_end],
        "test": rows[val_end:],
    }

    for name, part in parts.items():
        out = DATA_DIR / f"sp500_{name}.csv"
        with open(out, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(part)
        print(f"{name:>5}: {len(part):5d} rows  {part[0][0]} -> {part[-1][0]}  -> {out.name}")


if __name__ == "__main__":
    main()
