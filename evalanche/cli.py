from __future__ import annotations

import argparse
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

from evalanche.config import (
    load_config,
    load_evaluation_config,
    load_generation_config,
    load_metrics_config,
)
from evalanche.dataset_manifest import (
    save_dataset_verification,
    verify_dataset_manifest_file,
)
from evalanche.dpd_benchmark import (
    DPD_BENCHMARK_SEED,
    DPD_BENCHMARK_VERSION,
    DPD_SOURCE_MANIFEST,
    create_dpd_benchmark_slice,
)
from evalanche.dpd_snapshot import create_dpd_source_snapshot
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
        comparison_path,
        selection_path,
        metadata_path,
        report_path,
    ) = run_evaluation(
        config,
        config_path=config_path,
    )

    print(f"\nSaved combined case results to: {output_path}")
    print(f"Saved combined model summary to: {summary_path}")
    print(
        "Saved combined pairwise comparisons to: "
        f"{comparison_path}"
    )
    print(
        "Saved combined model selection to: "
        f"{selection_path}"
    )
    print(f"Saved combined run metadata to: {metadata_path}")
    print(f"Saved combined recommendation to: {report_path}")


def run_verify_dataset(
    manifest_path: str,
    *,
    root_path: str,
    output_path: str | None = None,
) -> dict[str, Any]:
    try:
        verification = verify_dataset_manifest_file(
            manifest_path,
            root_path=root_path,
        )
    except (OSError, ValueError) as error:
        print(f"Dataset manifest validation failed: {error}")
        raise SystemExit(1) from None

    print(
        "\nDataset: "
        f"{verification['dataset_id']} "
        f"{verification['dataset_version']}"
    )
    print(f"Manifest: {verification['manifest_path']}")
    print(
        "Verified files: "
        f"{verification['files_passed']}/"
        f"{verification['files_checked']}"
    )

    if output_path is not None:
        saved_path = save_dataset_verification(
            verification,
            output_path,
        )
        print(f"Saved verification report to: {saved_path}")

    if not verification["valid"]:
        print("Status: INVALID")
        for file_result in verification["files"]:
            for issue in file_result["issues"]:
                print(f"- {file_result['file_id']}: {issue}")
        raise SystemExit(1)

    print("Status: VALID")
    return verification


def run_snapshot_dpd(
    *,
    source_date: str,
    root_path: str,
    timeout: float,
) -> dict[str, Any]:
    try:
        result = create_dpd_source_snapshot(
            root_path=root_path,
            source_date=source_date,
            timeout=timeout,
        )
    except (OSError, ValueError) as error:
        print(f"DPD source snapshot failed: {error}")
        raise SystemExit(1) from None

    print(
        "\nDataset: "
        f"{result['dataset_id']} {result['dataset_version']}"
    )
    print(f"Source date: {result['source_date']}")
    print(f"Snapshot: {result['snapshot_path']}")
    print(f"Manifest: {result['manifest_path']}")
    print(f"Verified archives: {len(result['archives'])}")
    print("Status: VALID")
    return result


def run_build_dpd_benchmark(
    *,
    root_path: str,
    source_manifest_path: str,
    benchmark_version: str,
    seed: int,
) -> dict[str, Any]:
    try:
        result = create_dpd_benchmark_slice(
            root_path=root_path,
            source_manifest_path=source_manifest_path,
            benchmark_version=benchmark_version,
            seed=seed,
        )
    except (OSError, ValueError) as error:
        print(f"DPD benchmark slice build failed: {error}")
        raise SystemExit(1) from None

    print(
        "\nDataset: "
        f"{result['dataset_id']} {result['dataset_version']}"
    )
    print(
        "Parent: "
        f"{result['source_dataset_id']} "
        f"{result['source_dataset_version']}"
    )
    print(f"Products: {result['product_count']}")
    print(f"Cases: {result['case_count']}")
    print(
        "Product splits: "
        f"{result['development_product_count']} development, "
        f"{result['heldout_product_count']} heldout"
    )
    print(f"Output: {result['output_path']}")
    print(f"Manifest: {result['manifest_path']}")
    print("Verified files: 3/3")
    print("Status: VALID")
    return result


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

    dataset_parser = subparsers.add_parser("verify-dataset")
    dataset_parser.add_argument(
        "--manifest",
        required=True,
        help="Path to a versioned dataset manifest YAML file.",
    )
    dataset_parser.add_argument(
        "--root",
        default=".",
        help="Root directory used to resolve manifest file paths.",
    )
    dataset_parser.add_argument(
        "--output",
        help="Optional path for a JSON verification report.",
    )

    dpd_snapshot_parser = subparsers.add_parser("snapshot-dpd")
    dpd_snapshot_parser.add_argument(
        "--source-date",
        required=True,
        help=(
            "Expected official archive date in YYYY-MM-DD format. The "
            "download fails if HTTP Last-Modified does not match."
        ),
    )
    dpd_snapshot_parser.add_argument(
        "--root",
        default=".",
        help="Repository root for the snapshot data and manifest.",
    )
    dpd_snapshot_parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-archive HTTP timeout in seconds.",
    )

    dpd_benchmark_parser = subparsers.add_parser(
        "build-dpd-benchmark"
    )
    dpd_benchmark_parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing the frozen DPD snapshot.",
    )
    dpd_benchmark_parser.add_argument(
        "--source-manifest",
        default=DPD_SOURCE_MANIFEST.as_posix(),
        help="Path to the frozen DPD source manifest.",
    )
    dpd_benchmark_parser.add_argument(
        "--version",
        default=DPD_BENCHMARK_VERSION,
        help="Semantic version for the benchmark release.",
    )
    dpd_benchmark_parser.add_argument(
        "--seed",
        type=int,
        default=DPD_BENCHMARK_SEED,
        help="Recorded seed used for deterministic sampling.",
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
    elif args.command == "verify-dataset":
        run_verify_dataset(
            args.manifest,
            root_path=args.root,
            output_path=args.output,
        )
    elif args.command == "snapshot-dpd":
        run_snapshot_dpd(
            source_date=args.source_date,
            root_path=args.root,
            timeout=args.timeout,
        )
    elif args.command == "build-dpd-benchmark":
        run_build_dpd_benchmark(
            root_path=args.root,
            source_manifest_path=args.source_manifest,
            benchmark_version=args.version,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()