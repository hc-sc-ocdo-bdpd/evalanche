from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "case_id",
    "input",
    "expected_output",
    "model_name",
    "model_output",
}


def load_eval_cases(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {sorted(missing)}"
        )

    return df


def save_results(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)