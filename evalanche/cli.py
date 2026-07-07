from __future__ import annotations

import argparse

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

from evalanche.config import load_config
from evalanche.io import load_eval_cases, save_results
from evalanche.judges import CriteriaJudge
from evalanche.reporting import (
    print_failures,
    print_model_leaderboard,
    print_summary,
    save_model_summary,
)


def run_judge(config_path: str) -> None:
    load_dotenv()

    config = load_config(config_path)
    cases = load_eval_cases(config.run.input_path)

    judge = CriteriaJudge(config)

    records = []
    for row in tqdm(cases.to_dict(orient="records"), desc="Judging cases"):
        records.append(judge.judge_case(row))

    results = cases.merge(
        pd.DataFrame(records),
        on=["case_id", "model_name"],
        how="left",
    )

    save_results(results, config.run.output_path)
    summary_path = save_model_summary(results, config.run.output_path)

    print_summary(results)
    print_model_leaderboard(results)
    print_failures(results)

    print(f"\nSaved case results to: {config.run.output_path}")
    print(f"Saved model summary to: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalanche")
    subparsers = parser.add_subparsers(dest="command", required=True)

    judge_parser = subparsers.add_parser("judge")
    judge_parser.add_argument(
        "--config",
        required=True,
        help="Path to a judge evaluation YAML config.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "judge":
        run_judge(args.config)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()