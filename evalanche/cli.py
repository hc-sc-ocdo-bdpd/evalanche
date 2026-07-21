from __future__ import annotations

import argparse

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

from evalanche.config import (
    load_config,
    load_evaluation_config,
    load_generation_config,
    load_metrics_config,
)
from evalanche.evaluation import run_evaluation
from evalanche.generation import generate_outputs
from evalanche.io import load_eval_cases, save_results
from evalanche.judges import CriteriaJudge
from evalanche.metadata import save_run_metadata
from evalanche.metrics import run_deterministic_metrics
from evalanche.recommendation import save_recommendation_report
from evalanche.reporting import (
    print_failures,
    print_model_leaderboard,
    print_summary,
    save_model_summary,
)
from evalanche.routing import JUDGE


def run_generate(config_path: str) -> None:
    load_dotenv()

    config = load_generation_config(config_path)
    outputs, output_path, metadata_path = generate_outputs(
        config=config,
        config_path=config_path,
    )

    success_count = int((outputs["generation_status"] == "success").sum())
    error_count = int((outputs["generation_status"] == "error").sum())

    print(f"\nGenerated outputs: {len(outputs)}")
    print(f"Successful generations: {success_count}")
    print(f"Failed generations: {error_count}")
    print(f"Saved generated outputs to: {output_path}")
    print(f"Saved generation metadata to: {metadata_path}")


def run_metrics(config_path: str) -> None:
    load_dotenv()

    config = load_metrics_config(config_path)

    _, output_path, summary_path, metadata_path = run_deterministic_metrics(
        config_path=config_path,
        config=config,
    )

    print(f"\nSaved metric results to: {output_path}")
    print(f"Saved metrics summary to: {summary_path}")
    print(f"Saved metrics metadata to: {metadata_path}")


def run_judge(config_path: str) -> None:
    load_dotenv()

    config = load_config(config_path)
    cases = load_eval_cases(config.run.input_path)

    judge_cases = cases[
        cases["evaluation_type"] == JUDGE
    ].copy()

    skipped_count = len(cases) - len(judge_cases)

    if judge_cases.empty:
        print(
            "No cases were routed to LLM judge evaluation. "
            "Nothing to judge."
        )
        return

    judge = CriteriaJudge(config)

    records = []

    for row in tqdm(
        judge_cases.to_dict(orient="records"),
        desc="Judging cases",
    ):
        records.append(judge.judge_case(row))

    results = judge_cases.merge(
        pd.DataFrame(records),
        on=["case_id", "model_name"],
        how="left",
    )

    save_results(results, config.run.output_path)

    summary_path = save_model_summary(results, config.run.output_path)

    metadata_path = save_run_metadata(
        config_path=config_path,
        config=config,
        cases=cases,
        results=results,
        case_results_path=config.run.output_path,
        model_summary_path=summary_path,
    )

    recommendation_path = save_recommendation_report(
        config=config,
        results=results,
        case_results_path=config.run.output_path,
        model_summary_path=summary_path,
        metadata_path=metadata_path,
    )

    print_summary(results)
    print_model_leaderboard(results)
    print_failures(results)
    
    if skipped_count:
        print(
            f"\nSkipped {skipped_count} case(s) routed "
            "to deterministic evaluation."
        )

    print(f"\nSaved case results to: {config.run.output_path}")
    print(f"Saved model summary to: {summary_path}")
    print(f"Saved run metadata to: {metadata_path}")
    print(f"Saved recommendation report to: {recommendation_path}")
    
def run_combined_evaluation(config_path: str) -> None:
    load_dotenv()

    config = load_evaluation_config(config_path)

    (
        _,
        output_path,
        summary_path,
        metadata_path,
        report_path,
    ) = run_evaluation(
        config,
        config_path=config_path,
    )

    print(f"\nSaved combined case results to: {output_path}")
    print(f"Saved combined model summary to: {summary_path}")
    print(f"Saved combined run metadata to: {metadata_path}")
    print(f"Saved combined recommendation to: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalanche")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--config",
        required=True,
        help="Path to a combined evaluation YAML config.",
    )

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument(
        "--config",
        required=True,
        help="Path to a generation YAML config.",
    )

    metrics_parser = subparsers.add_parser("metrics")
    metrics_parser.add_argument(
        "--config",
        required=True,
        help="Path to a deterministic metrics YAML config.",
    )

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

    if args.command == "generate":
        run_generate(args.config)
    elif args.command == "metrics":
        run_metrics(args.config)
    elif args.command == "judge":
        run_judge(args.config)
    elif args.command == "evaluate":
        run_combined_evaluation(args.config)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()