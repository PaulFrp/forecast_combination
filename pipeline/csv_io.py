"""Read and write the project's shared CSV contract."""

import csv
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 22)
KEY = ("origin_date", "horizon")
DAILY = ("date", "spy_open", "spy_high", "spy_low", "spy_close", "spy_volume", "spy_rv", "vix_close")
ACTUALS = (*KEY, "actual", "baseline")
FORECAST = ("model_id", *KEY, "prediction")


def read_csv(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """Preserve literal cells and reject malformed rows before interpreting values."""
    with Path(path).open(encoding="utf-8", newline="") as source:
        reader = csv.reader(source, strict=True)
        try:
            if next(reader, None) != list(columns):
                raise ValueError(f"{path}: expected header {','.join(columns)}")
            rows = []
            for row in reader:
                if len(row) != len(columns):
                    raise ValueError(f"{path}: wrong column count on line {reader.line_num}: {row}")
                rows.append(row)
        except csv.Error as error:
            raise ValueError(f"{path}: invalid CSV on line {reader.line_num}: {error}") from error
    return pd.DataFrame(rows, columns=list(columns), dtype="str")


def iso_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected YYYY-MM-DD, got {value!r}") from error
    if parsed.isoformat() != value:
        raise ValueError(f"expected YYYY-MM-DD, got {value!r}")
    return value


def validate(frame: pd.DataFrame, columns: tuple[str, ...], source: Path) -> pd.DataFrame:
    if list(frame.columns) != list(columns):
        raise ValueError(f"{source}: expected header {','.join(columns)}")
    result = frame.astype(object).where(frame.notna(), "").astype(str)
    date_column = "date" if columns == DAILY else "origin_date"
    key = ["date"] if columns == DAILY else list(KEY)
    for row in result.itertuples(index=False):
        try:
            iso_date(getattr(row, date_column))
            if columns != DAILY and row.horizon not in {"1", "5", "22"}:
                raise ValueError(f"invalid horizon {row.horizon!r}")
        except ValueError as error:
            raise ValueError(f"{source}: {tuple(getattr(row, field) for field in key)}: {error}") from error
    duplicate = result.duplicated(key, keep=False)
    if duplicate.any():
        raise ValueError(f"{source}: duplicate key {tuple(result.loc[duplicate, key].iloc[0])}")
    if columns != DAILY:
        result["horizon"] = result["horizon"].astype(int)

    positive = {"spy_open", "spy_high", "spy_low", "spy_close", "vix_close", "prediction"}
    nonnegative = {"spy_volume", "spy_rv", "actual", "baseline"}
    for column in (name for name in columns if name in positive | nonnegative):
        values = pd.to_numeric(result[column], errors="coerce").astype(float)
        bad_sign = values.le(0) if column in positive else values.lt(0)
        bad = result[column].ne("") & (~np.isfinite(values) | bad_sign)
        if bad.any():
            row = result.loc[bad].iloc[0]
            raise ValueError(f"{source}: {tuple(row[name] for name in key)}: invalid {column} {row[column]!r}")
        result[column] = values
    if columns == FORECAST:
        ids = result["model_id"].drop_duplicates().tolist()
        if len(ids) != 1 or not ids[0] or ids[0].strip() != ids[0]:
            raise ValueError(f"{source}: expected one nonblank model_id, got {ids}")
    return result.sort_values(key, ignore_index=True)


def require_schedule(frame: pd.DataFrame, schedule: pd.DataFrame, source: Path) -> None:
    keys = pd.MultiIndex.from_frame(frame[list(KEY)])
    expected = pd.MultiIndex.from_frame(schedule[list(KEY)])
    missing = expected.difference(keys)
    extra = keys.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(f"{source}: schedule mismatch ({len(missing)} missing {missing.tolist()[:3]}, "
                         f"{len(extra)} extra {extra.tolist()[:3]})")


def load(path: Path, columns: tuple[str, ...], *, schedule: pd.DataFrame | None = None) -> pd.DataFrame:
    result = validate(read_csv(path, columns), columns, path)
    if schedule is not None:
        require_schedule(result, schedule, path)
    return result


def export(frame: pd.DataFrame, path: Path, columns: tuple[str, ...],
           *, schedule: pd.DataFrame | None = None) -> None:
    result = validate(frame, columns, path)
    if schedule is not None:
        require_schedule(result, schedule, path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
