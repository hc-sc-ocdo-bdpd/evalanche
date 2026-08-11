from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

from evalanche.access_sets import (
    explicit_model_scope,
    historical_results_scope,
    resolve_access_set,
)
from evalanche.config import (
    load_config,
    load_evaluation_config,
    load_generation_config,
    load_metrics_config,
)
from evalanche.benchmark_runner import (
    rescore_registered_benchmark,
    run_experimental_benchmark,
    run_registered_benchmark,
)
from evalanche.benchmark_campaign import (
    build_tier_campaign_preflight,
    execute_tier_campaign,
    select_models_for_promotion,
)
from evalanche.benchmark_tiers import materialize_tier_cohort, tier_is_ancestor
from evalanche.benchmark_experiment import (
    build_benchmark_experiment_summary,
    print_benchmark_experiment_summary,
)
from evalanche.dataset_manifest import (
    save_dataset_verification,
    verify_dataset_manifest_file,
)
from evalanche.dpd_analysis import (
    DPD_ANALYSIS_ALL_PASS_SAMPLE,
    DPD_ANALYSIS_COMPARISON_MODEL,
    DPD_ANALYSIS_CONFIG_PATH,
    DPD_ANALYSIS_OUTPUT_DIR,
    DPD_ANALYSIS_PRIMARY_MODEL,
    DPD_ANALYSIS_RESULTS_PATH,
    DPD_ANALYSIS_REVIEW_SEED,
    DPD_ANALYSIS_SHARED_FAILURE_SAMPLE,
    build_dpd_census_analysis,
)
from evalanche.dpd_benchmark import (
    DPD_BENCHMARK_SEED,
    DPD_BENCHMARK_VERSION,
    DPD_SOURCE_MANIFEST,
    create_dpd_benchmark_slice,
)
from evalanche.dpd_census import create_dpd_benchmark_census
from evalanche.dpd_comparison import (
    DPD_COMPARISON_PRODUCT_COUNT,
    DPD_COMPARISON_SEED,
    assemble_dpd_comparison_outputs,
    build_dpd_comparison_sample,
)
from evalanche.dpd_census_comparison import (
    assemble_dpd_census_comparison_outputs,
)
from evalanche.dpd_evidence_audit import (
    DPD_EVIDENCE_AUDIT_ARCHIVE_PATH,
    DPD_EVIDENCE_AUDIT_CONFIG_PATH,
    DPD_EVIDENCE_AUDIT_OUTPUT_PATH,
    DPD_EVIDENCE_AUDIT_REVIEW_PATH,
    DPD_EVIDENCE_AUDIT_SUMMARY_PATH,
    build_dpd_evidence_audit,
)
from evalanche.dpd_snapshot import create_dpd_source_snapshot
from evalanche.evaluation import run_evaluation
from evalanche.evidence_catalog import (
    EvidenceCatalogError,
    build_evidence_status,
    evaluate_evidence_catalog,
    render_evidence_status_markdown,
)
from evalanche.generation import generate_outputs, preflight_generation
from evalanche.io import load_eval_cases, save_results
from evalanche.judges import CriteriaJudge
from evalanche.judges.calibration import validate_judge_protocol
from evalanche.metadata import save_run_metadata
from evalanche.metrics import run_deterministic_metrics
from evalanche.recommendation import save_comparison_report
from evalanche.reporting import (
    print_failures,
    print_model_leaderboard,
    print_summary,
    save_model_summary,
)
from evalanche.leaderboard import (
    build_benchmark_index,
    build_leaderboard,
)
from evalanche.registry import load_registry, validate_registry
from evalanche.result_bundle import register_result_bundle
from evalanche.product_monograph import (
    PM_COHORT_PATH,
    PM_OVERRIDES_PATH,
    PM_RAW_DIR,
    PM_SOURCES_PATH,
    acquire_product_monographs,
    build_product_monograph_native_pdf_benchmark,
    build_product_monograph_benchmark,
    verify_product_monograph_sources,
)
from evalanche.product_monograph_review import (
    check_product_monograph_label_review,
)
from evalanche.product_monograph_expansion import (
    EXPANSION_CONFIG_PATH,
    check_expanded_product_monograph_audit,
    expand_product_monograph_benchmark,
    refresh_expanded_product_monograph_release,
)
from evalanche.routing import JUDGE
from evalanche.task_initializer import (
    initialize_task_bundle,
    normalize_family,
    parse_criterion,
    parse_model_route,
    task_execution_blocker,
    validate_task_bundle,
)


def run_evidence_status(
    *,
    root_path: str = ".",
    catalog_path: str = "docs/evidence/catalog.yaml",
    as_of_value: str | None = None,
    output_path: str = "docs/evidence/status.md",
    write: bool = False,
    check: bool = False,
    fail_on_outdated: bool = False,
) -> None:
    try:
        as_of = date.fromisoformat(as_of_value) if as_of_value else None
    except ValueError:
        print("Evidence status failed: --as-of must be YYYY-MM-DD")
        raise SystemExit(1) from None

    try:
        catalog, report = build_evidence_status(
            root_path=root_path,
            catalog_path=catalog_path,
            as_of=as_of,
        )
        if check and as_of is None:
            report = evaluate_evidence_catalog(
                catalog,
                root_path=root_path,
                as_of=catalog.status_report_as_of,
            )
        rendered = render_evidence_status_markdown(
            report,
            report_path=output_path,
        )
    except (OSError, EvidenceCatalogError) as error:
        print(f"Evidence status failed: {error}")
        raise SystemExit(1) from None

    root = Path(root_path)
    target = Path(output_path)
    if not target.is_absolute():
        target = root / target

    if write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
        print(f"Saved evidence status to: {target}")
    elif check:
        try:
            saved = target.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"Evidence status snapshot is missing: {target}")
            raise SystemExit(1) from None
        if saved != rendered:
            print(
                "Evidence status snapshot is out of date. Regenerate it "
                "after reviewing the catalog."
            )
            raise SystemExit(1) from None
        print(f"Evidence status snapshot matches: {target}")

    counts = report.counts
    print("\nEvidence maintenance")
    print(f"As of: {report.as_of.isoformat()}")
    print(f"Entries: {len(report.items)}")
    print(f"Current: {counts['current']}")
    print(f"Stale: {counts['stale']}")
    print(f"Superseded: {counts['superseded']}")
    for item in report.items:
        if item.effective_status == "current":
            continue
        print(
            f"- {item.effective_status.upper()}: "
            f"{item.entry.entry_id}, review by {item.entry.review_by}"
        )
    print("Status: " + ("NEEDS REVIEW" if report.outdated_count else "CURRENT"))
    print("No network or model calls were made.")

    if fail_on_outdated and report.outdated_count:
        raise SystemExit(2)


def run_generate(
    config_path: str,
    *,
    preflight_only: bool = False,
) -> None:
    load_dotenv()

    config = load_generation_config(config_path)
    if preflight_only:
        try:
            preflight = preflight_generation(config=config)
        except (OSError, ValueError) as error:
            print(f"Generation preflight failed: {error}")
            raise SystemExit(1) from None

        print("\nGeneration cost preflight")
        print(f"Expected outputs: {preflight['expected_output_count']}")
        print(f"Checkpointed outputs: {preflight['checkpointed_output_count']}")
        print(f"Pending outputs: {preflight['pending_output_count']}")
        print(f"Projected token cost: ${preflight['projected_total_cost_usd']:.2f} USD")
        print(f"Safety multiplier: {preflight['cost_safety_multiplier']:.2f}x")
        print(
            f"Budgeted projection: ${preflight['projected_budgeted_total_usd']:.2f} USD"
        )
        print(
            "Configured limit: "
            + (
                f"${preflight['maximum_estimated_cost_usd']:.2f} USD"
                if preflight["maximum_estimated_cost_usd"] is not None
                else "not set"
            )
        )
        print("Status: READY")
        print("No model calls were made.")
        return

    try:
        outputs, output_path, metadata_path = generate_outputs(
            config=config,
            config_path=config_path,
        )
    except (OSError, ValueError) as error:
        print(f"Generation stopped safely: {error}")
        raise SystemExit(1) from None

    success_count = int((outputs["generation_status"] == "success").sum())
    error_count = int((outputs["generation_status"] == "error").sum())
    run_status = str(outputs.attrs.get("run_status", "completed"))
    expected_output_count = int(
        outputs.attrs.get("expected_output_count", len(outputs))
    )
    checkpoint_path = outputs.attrs.get("checkpoint_path")

    print(f"\nGenerated outputs: {len(outputs)}")
    print(f"Expected outputs: {expected_output_count}")
    print(f"Successful generations: {success_count}")
    print(f"Failed generations: {error_count}")
    print(f"Generation status: {run_status.upper()}")
    print(f"Saved generated outputs to: {output_path}")
    print(f"Saved generation metadata to: {metadata_path}")
    if checkpoint_path:
        print(f"Saved resumable checkpoint to: {checkpoint_path}")

    if run_status != "completed":
        print(
            "Generation stopped safely before every configured output "
            "was completed. Re-run the same command to resume after "
            "reviewing the recorded status and cost evidence."
        )
        raise SystemExit(2)


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

    judge_cases = cases[cases["evaluation_type"] == JUDGE].copy()

    skipped_count = len(cases) - len(judge_cases)

    if judge_cases.empty:
        print("No cases were routed to LLM judge evaluation. Nothing to judge.")
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

    comparison_report_path = save_comparison_report(
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
        print(f"\nSkipped {skipped_count} case(s) routed to deterministic evaluation.")

    print(f"\nSaved case results to: {config.run.output_path}")
    print(f"Saved model summary to: {summary_path}")
    print(f"Saved run metadata to: {metadata_path}")
    print(f"Saved comparison report to: {comparison_report_path}")


def run_validate_judge(
    protocol_path: str,
    *,
    root_path: str = ".",
    output_dir: str | None = None,
) -> None:
    try:
        artifacts = validate_judge_protocol(
            protocol_path,
            root_path=root_path,
            output_dir=output_dir,
        )
    except (OSError, ValueError) as error:
        print(f"Judge validation failed: {error}")
        raise SystemExit(1) from None

    print("\nJudge validation evidence")
    print(f"Target level: {artifacts.target_level}")
    print(f"Achieved level: {artifacts.achieved_level}")
    print(f"Target met: {'yes' if artifacts.target_met else 'no'}")
    print(f"Report: {artifacts.report_markdown_path}")
    print(f"Machine-readable report: {artifacts.report_json_path}")
    print(f"Gate results: {artifacts.gates_path}")
    print(f"Disagreements: {artifacts.disagreements_path}")
    print(f"Subgroups: {artifacts.subgroups_path}")

    if not artifacts.target_met:
        print(
            "The declared target level was not achieved. Review the saved "
            "gate results before using judge scores for stronger claims."
        )
        raise SystemExit(2)


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
    print(f"Saved combined pairwise comparisons to: {comparison_path}")
    print(f"Saved combined model selection to: {selection_path}")
    print(f"Saved combined run metadata to: {metadata_path}")
    print(f"Saved combined comparison to: {report_path}")


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

    print(f"\nDataset: {verification['dataset_id']} {verification['dataset_version']}")
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

    print(f"\nDataset: {result['dataset_id']} {result['dataset_version']}")
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

    print(f"\nDataset: {result['dataset_id']} {result['dataset_version']}")
    print(f"Parent: {result['source_dataset_id']} {result['source_dataset_version']}")
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


def run_build_dpd_census(
    *,
    root_path: str,
    source_manifest_path: str,
) -> dict[str, Any]:
    try:
        result = create_dpd_benchmark_census(
            root_path=root_path,
            source_manifest_path=source_manifest_path,
        )
    except (OSError, ValueError) as error:
        print(f"DPD benchmark census build failed: {error}")
        raise SystemExit(1) from None

    verification = result["verification"]
    print(f"\nDataset: {result['dataset_id']} {result['dataset_version']}")
    print(f"Parent: {result['source_dataset_id']} {result['source_dataset_version']}")
    print(f"Products: {result['product_count']}")
    print(f"Cases: {result['case_count']}")
    print(
        "Bounded demo: "
        f"{result['demo_product_count']} products, "
        f"{result['demo_case_count']} cases"
    )
    print(
        "Source-ambiguous families excluded: "
        f"{result['excluded_ambiguous_family_count']}"
    )
    print(f"Output: {result['output_path']}")
    print(f"Manifest: {result['manifest_path']}")
    print(
        "Verified files: "
        f"{verification['files_passed']}/"
        f"{verification['files_checked']}"
    )
    print("Status: VALID")
    return result


def run_build_dpd_comparison(
    *,
    root_path: str,
    seed: int,
    product_count: int,
) -> dict[str, Any]:
    result = build_dpd_comparison_sample(
        root_path=root_path,
        seed=seed,
        product_count=product_count,
    )
    print("\nHealth Canada DPD model-comparison sample")
    print(f"Products: {result['product_count']}")
    print(f"Cases: {result['case_count']}")
    print(f"Seed: {result['seed']}")
    print(f"Output: {result['output_path']}")
    print(f"Report: {result['report_path']}")
    print("Status: VALID")
    return result


def run_assemble_dpd_comparison(
    *,
    root_path: str,
    required_models: list[str] | None = None,
) -> dict[str, Any]:
    result = assemble_dpd_comparison_outputs(
        root_path=root_path,
        required_models=required_models,
    )
    print("\nHealth Canada DPD model-comparison outputs")
    print("Models: " + ", ".join(result["available_models"]))
    print(f"Cases per model: {result['case_count_per_model']}")
    print(f"Combined rows: {result['combined_row_count']}")
    print(f"Output: {result['output_path']}")
    print(f"Metadata: {result['metadata_path']}")
    print("Status: READY FOR EVALUATION")
    return result


def run_assemble_dpd_census_comparison(
    *,
    root_path: str,
    required_models: list[str] | None = None,
) -> dict[str, Any]:
    result = assemble_dpd_census_comparison_outputs(
        root_path=root_path,
        required_models=required_models,
    )
    print("\nHealth Canada DPD full-census comparison outputs")
    print("Models: " + ", ".join(result["available_models"]))
    print(f"Cases per model: {result['case_count_per_model']}")
    print(f"Combined rows: {result['combined_row_count']}")
    print(f"Output: {result['output_path']}")
    print(f"Metadata: {result['metadata_path']}")
    print("Status: READY FOR EVALUATION")
    return result


def run_analyze_dpd_census(
    *,
    root_path: str,
    results_path: str,
    config_path: str,
    output_dir: str,
    primary_model: str,
    comparison_model: str,
    shared_failure_sample: int,
    all_pass_sample: int,
    review_seed: int,
) -> dict[str, Any]:
    print("Analyzing saved DPD census results. No model calls will be made.")
    try:
        result = build_dpd_census_analysis(
            root_path=root_path,
            results_path=results_path,
            config_path=config_path,
            output_dir=output_dir,
            primary_model=primary_model,
            comparison_model=comparison_model,
            shared_failure_sample=shared_failure_sample,
            all_pass_sample=all_pass_sample,
            review_seed=review_seed,
        )
    except (OSError, ValueError) as error:
        print(f"DPD census analysis failed: {error}")
        raise SystemExit(1) from None

    print("\nHealth Canada DPD census analysis")
    print("Models: " + ", ".join(result["source"]["models"]))
    print(f"Cases: {result['benchmark']['case_count']}")
    print(f"Product families: {result['benchmark']['product_family_count']}")
    print(f"Manual review cases: {result['review_set']['total_unique_cases']}")
    print(f"Output: {result['output_dir']}")
    print(f"Manifest: {result['manifest_path']}")
    print("Status: ANALYSIS COMPLETE")
    print(
        "Next: run audit-dpd-review to refresh the selected-case "
        "evidence audit and release manifests."
    )
    print("No model calls were made.")
    return result


def run_audit_dpd_review(
    *,
    root_path: str,
    review_path: str,
    config_path: str,
    archive_path: str,
    output_path: str,
    summary_path: str,
) -> dict[str, Any]:
    print(
        "Auditing the DPD review set against frozen source evidence. "
        "No model calls will be made."
    )
    try:
        result = build_dpd_evidence_audit(
            root_path=root_path,
            review_path=review_path,
            config_path=config_path,
            archive_path=archive_path,
            output_path=output_path,
            summary_path=summary_path,
        )
    except (OSError, ValueError) as error:
        print(f"DPD evidence audit failed: {error}")
        raise SystemExit(1) from None

    verification = result["verification"]
    print("\nHealth Canada DPD evidence audit")
    print(f"Review cases: {verification['review_cases']}")
    print(f"Frozen-source rebuild matches: {verification['source_rebuild_matches']}")
    print(
        "Independent expected-answer parses: "
        f"{verification['independent_expected_parses_match']}"
    )
    print(f"Model outputs rescored: {verification['model_outputs_rescored']}")
    print(f"Strict-score disagreements: {verification['strict_score_disagreements']}")
    print(f"Output: {result['output']['path']}")
    print(f"Summary: {result['summary_path']}")
    print(
        "Release manifests refreshed: "
        + (
            "yes"
            if result["release_manifests_refreshed"]
            else "no, custom output paths were used"
        )
    )
    print("Status: AUTOMATED EVIDENCE AUDIT COMPLETE")
    print("Human sign-off: NOT CLAIMED")
    print("No model calls were made.")
    return result


def run_registry_validate(*, root_path: str) -> dict[str, Any]:
    try:
        registry = load_registry(root_path)
        result = validate_registry(registry)
        access_paths = sorted((registry.root / "configs/access_sets").glob("*.yaml"))
        access_issues: list[str] = []
        for path in access_paths:
            try:
                resolve_access_set(registry=registry, path=path)
            except (OSError, ValueError) as error:
                access_issues.append(f"Access set {path.name}: {error}")
        result["access_sets"] = len(access_paths)
        result["issues"].extend(access_issues)
        result["valid"] = not result["issues"]
    except (OSError, ValueError) as error:
        print(f"Registry validation failed: {error}")
        raise SystemExit(1) from None

    print("\nEvalanche registry")
    print(f"Models: {result['models']}")
    print(f"Benchmarks: {result['benchmarks']}")
    print(f"Access sets: {result['access_sets']}")
    if not result["valid"]:
        print("Status: INVALID")
        for issue in result["issues"]:
            print(f"- {issue}")
        raise SystemExit(1)
    print("Status: VALID")
    return result


def _prompt(
    label: str,
    *,
    default: str | None = None,
    required: bool = False,
) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""
        print("A value is required.")


def _prompt_bool(label: str, *, default: bool = False) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        value = input(f"{label} [{default_text}]: ").strip().casefold()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Enter yes or no.")


def _split_values(value: str, *, separator: str = ",") -> list[str]:
    return [item.strip() for item in value.split(separator) if item.strip()]


def run_init_task(
    *,
    root_path: str,
    output_dir: str,
    task_id: str | None,
    description: str | None,
    family: str | None,
    languages: list[str] | None,
    slice_columns: list[str] | None,
    unacceptable_errors: list[str] | None,
    model_routes: list[str] | None,
    required_output_fields: list[str] | None,
    criteria: list[str] | None,
    measured_construct: str | None,
    intended_use: str | None,
    sample_input: str | None,
    sample_expected_output: str | None,
    artifact_policy: str | None,
    contains_sensitive_data: bool,
    privacy_notes: str | None,
    starting_tier: str | None,
    non_interactive: bool,
    replace: bool,
) -> dict[str, Any]:
    try:
        if non_interactive:
            missing = [
                name
                for name, value in (
                    ("--task-id", task_id),
                    ("--description", description),
                    ("--family", family),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "non-interactive initialization requires " + ", ".join(missing)
                )
        else:
            print("\nGuided task initialization")
            print("No provider calls will be made.")
            task_id = task_id or _prompt(
                "Stable task ID (lowercase snake_case)",
                required=True,
            )
            description = description or _prompt(
                "Short task description",
                required=True,
            )
            family = family or _prompt(
                "Task family (classification, structured-extraction, "
                "factual-response, open-ended-response)",
                required=True,
            )
            if languages is None:
                languages = _split_values(
                    _prompt("Languages, comma separated", default="en")
                )
            if slice_columns is None:
                raw_slices = _prompt(
                    "Additional slice columns, comma separated",
                    default="",
                )
                slice_columns = _split_values(raw_slices)
            if unacceptable_errors is None:
                raw_errors = _prompt(
                    "Unacceptable errors, separated by semicolons",
                    default="",
                )
                unacceptable_errors = _split_values(
                    raw_errors,
                    separator=";",
                )
            contains_sensitive_data = _prompt_bool(
                "Could real task cases contain sensitive information?",
                default=contains_sensitive_data,
            )
            artifact_policy = artifact_policy or _prompt(
                "Artifact policy (local_only or shareable)",
                default="local_only",
            )
            privacy_notes = privacy_notes or _prompt(
                "Privacy or local-artifact note",
                default=(
                    "Keep cases and generated outputs local until sharing is reviewed."
                ),
            )
            normalized_family = normalize_family(family)
            if normalized_family == "structured_extraction" and (
                required_output_fields is None
            ):
                required_output_fields = _split_values(
                    _prompt(
                        "Required JSON fields, comma separated",
                        default="",
                    )
                )
            if normalized_family == "open_ended_response":
                measured_construct = measured_construct or _prompt(
                    "What should the judge measure?",
                    default="",
                )
                intended_use = intended_use or _prompt(
                    "How may the judge evidence be used?",
                    default="",
                )
                if criteria is None:
                    raw_criteria = _prompt(
                        "Criteria as NAME=WEIGHT=DESCRIPTION, separated by semicolons",
                        default="",
                    )
                    criteria = _split_values(
                        raw_criteria,
                        separator=";",
                    )
            if not contains_sensitive_data:
                sample_input = sample_input or _prompt(
                    "Optional first development input",
                    default="",
                )
                sample_expected_output = sample_expected_output or _prompt(
                    "Optional expected output for that input",
                    default="",
                )
            if model_routes is None:
                raw_models = _prompt(
                    "Confirmed routes as MODEL_ID=PROVIDER_ROUTE, separated "
                    "by semicolons",
                    default="",
                )
                model_routes = _split_values(raw_models, separator=";")
            starting_tier = starting_tier or _prompt(
                "Starting tier (smoke or screen)",
                default="smoke",
            )

        parsed_models = [parse_model_route(value) for value in (model_routes or [])]
        parsed_criteria = [parse_criterion(value) for value in (criteria or [])]
        result = initialize_task_bundle(
            root_path=root_path,
            output_dir=output_dir,
            task_id=str(task_id),
            description=str(description),
            family=str(family),
            languages=languages,
            slice_columns=slice_columns,
            unacceptable_errors=unacceptable_errors,
            models=parsed_models,
            required_output_fields=required_output_fields,
            criteria=parsed_criteria,
            measured_construct=measured_construct or None,
            intended_use=intended_use or None,
            sample_input=sample_input or None,
            sample_expected_output=sample_expected_output or None,
            artifact_policy=(artifact_policy or "local_only"),
            contains_sensitive_data=contains_sensitive_data,
            privacy_notes=privacy_notes,
            starting_tier=(starting_tier or "smoke"),
            replace=replace,
        )
    except (EOFError, OSError, ValueError) as error:
        print(f"Task initialization failed: {error}")
        raise SystemExit(1) from None

    validation = result["validation"]
    print("\nEvalanche task starter")
    print(f"Task: {result['task_id']}")
    print(f"Bundle: {result['task_root']}")
    if result["backup"] is not None:
        print(f"Previous bundle preserved at: {result['backup']}")
    print("Structure: " + ("VALID" if validation["schema_valid"] else "INVALID"))
    if validation["ready_for_plan"]:
        print("Readiness: READY FOR OFFLINE PLAN")
    else:
        print("Readiness: NEEDS USER INPUT")
        for item in validation["unresolved_values"]:
            print(f"- {item}")
    print("Next: open the generated README.md.")
    print("No provider calls were made.")
    return result


def run_validate_task(
    *,
    root_path: str,
    require_cost_ready: bool = False,
) -> dict[str, Any]:
    try:
        result = validate_task_bundle(root_path)
    except (OSError, ValueError) as error:
        print(f"Task validation failed: {error}")
        raise SystemExit(1) from None

    print("\nEvalanche task bundle")
    print(f"Task: {result['task_id'] or 'unknown'}")
    print(f"Root: {result['task_root']}")
    print("Structure: " + ("VALID" if result["schema_valid"] else "INVALID"))
    if result["structural_issues"]:
        print("Structural issues:")
        for item in result["structural_issues"]:
            print(f"- {item}")
    if result["unresolved_values"]:
        print("Required task input:")
        for item in result["unresolved_values"]:
            print(f"- {item}")
    if result["cost_blockers"]:
        print("Cost preflight blockers:")
        for item in result["cost_blockers"]:
            print(f"- {item}")
    if result["warnings"]:
        print("Warnings:")
        for item in result["warnings"]:
            print(f"- {item}")
    print("No provider calls were made.")

    if not result["schema_valid"]:
        print("Status: INVALID")
        raise SystemExit(1)
    if not result["ready_for_plan"]:
        print("Status: NEEDS USER INPUT")
        raise SystemExit(2)
    if require_cost_ready and not result["ready_for_cost_preflight"]:
        print("Status: NOT READY FOR COST PREFLIGHT")
        raise SystemExit(2)
    print(
        "Status: "
        + (
            "READY FOR COST PREFLIGHT"
            if result["ready_for_cost_preflight"]
            else "READY FOR OFFLINE PLAN"
        )
    )
    return result


def run_register_result(
    *,
    root_path: str,
    benchmark_reference: str,
    model_id: str,
    results_path: str,
    provenance_path: str | None,
) -> dict[str, Any]:
    try:
        registry = load_registry(root_path)
        benchmark = registry.resolve_benchmark(benchmark_reference)
        result = register_result_bundle(
            registry=registry,
            benchmark=benchmark,
            model_id=model_id,
            results_path=results_path,
            provenance_path=provenance_path,
        )
    except (OSError, ValueError) as error:
        print(f"Result registration failed: {error}")
        raise SystemExit(1) from None

    print("\nRegistered result bundle")
    print(f"Benchmark: {benchmark.benchmark_id}@{benchmark.version}")
    print(f"Model: {model_id}")
    print(f"Run: {result['run_id']}")
    print(f"Cases: {result['summary']['cases']}")
    print(f"Output: {result['run_dir']}")
    print("Status: REGISTERED")
    return result


def run_build_leaderboard(
    *,
    root_path: str,
    benchmark_reference: str | None,
    build_all: bool,
) -> list[dict[str, Any]]:
    try:
        registry = load_registry(root_path)
        if build_all:
            benchmarks = list(registry.benchmarks.values())
        elif benchmark_reference is not None:
            benchmarks = [registry.resolve_benchmark(benchmark_reference)]
        else:
            raise ValueError("Provide --benchmark or use --all.")
        results = [
            build_leaderboard(
                registry=registry,
                benchmark=benchmark,
            )
            for benchmark in benchmarks
        ]
        index_path = build_benchmark_index(registry=registry)
    except (OSError, ValueError) as error:
        print(f"Leaderboard build failed: {error}")
        raise SystemExit(1) from None

    print("\nEvalanche leaderboards")
    print(f"Built leaderboards: {len(results)}")
    for result in results:
        print(
            f"- {result['benchmark_id']}@"
            f"{result['benchmark_version']}: "
            f"{result['models']} model(s)"
        )
    print(f"Index: {index_path}")
    print("Status: LEADERBOARD BUILT")
    return results


def _resolve_access_confirmed_models(
    *,
    registry: Any,
    benchmark: Any,
    model_ids: list[str] | None,
    all_compatible: bool,
    access_set_path: str | None,
) -> tuple[list[str], dict[str, Any]]:
    explicit_ids = list(dict.fromkeys(model_ids or []))
    if all_compatible:
        if access_set_path is None:
            raise ValueError(
                "--all-compatible requires --access-set. Compatibility "
                "does not prove that a model is available to the user."
            )
        resolution = resolve_access_set(
            registry=registry,
            path=access_set_path,
            benchmark=benchmark,
        )
        selected = resolution["compatible_model_ids"]
        scope = dict(resolution["scope"])
        scope["excluded_incompatible"] = resolution["incompatible_models"]
        return selected, scope

    if not explicit_ids:
        raise ValueError("No models were explicitly selected")
    if access_set_path is None:
        return explicit_ids, explicit_model_scope(explicit_ids)

    resolution = resolve_access_set(
        registry=registry,
        path=access_set_path,
        benchmark=benchmark,
    )
    outside = sorted(set(explicit_ids) - set(resolution["access_set"].model_ids))
    if outside:
        raise ValueError(
            "Explicitly selected models are not in the access set: "
            + ", ".join(outside)
        )
    scope = dict(resolution["scope"])
    scope["selected_model_ids"] = explicit_ids
    return explicit_ids, scope


def run_tier_campaign_from_registry(
    *,
    root_path: str,
    benchmark_reference: str,
    tier_name: str,
    model_ids: list[str] | None,
    all_compatible: bool,
    promote_from: str | None,
    plan_only: bool,
    preflight_only: bool,
    experiment: bool,
    maximum_cost_usd: float | None,
    access_set_path: str | None,
) -> dict[str, Any]:
    try:
        registry = load_registry(root_path)
        benchmark = registry.resolve_benchmark(benchmark_reference)
        promotion = None
        access_scope: dict[str, Any]
        if promote_from is not None:
            if not tier_is_ancestor(
                benchmark,
                ancestor=promote_from,
                descendant=tier_name,
            ):
                raise ValueError(
                    f"Promotion tier {promote_from!r} is not an ancestor "
                    f"of {tier_name!r}"
                )
            if access_set_path is None:
                raise ValueError(
                    "--promote-from requires --access-set so only "
                    "access-confirmed models can be promoted."
                )
            resolution = resolve_access_set(
                registry=registry,
                path=access_set_path,
                benchmark=benchmark,
            )
            candidate_ids = resolution["compatible_model_ids"]
            access_scope = dict(resolution["scope"])
            access_scope["excluded_incompatible"] = resolution["incompatible_models"]
            promotion = select_models_for_promotion(
                registry=registry,
                benchmark=benchmark,
                source_tier=promote_from,
                model_ids=candidate_ids,
                access_scope=access_scope,
            )
            selected_ids = promotion["selected_model_ids"]
        else:
            selected_ids, access_scope = _resolve_access_confirmed_models(
                registry=registry,
                benchmark=benchmark,
                model_ids=model_ids,
                all_compatible=all_compatible,
                access_set_path=access_set_path,
            )
        if not selected_ids:
            raise ValueError("No models qualify for this tier campaign")
        preflight = build_tier_campaign_preflight(
            registry=registry,
            benchmark=benchmark,
            tier_name=tier_name,
            model_ids=selected_ids,
            maximum_cost_usd=maximum_cost_usd,
            access_scope=access_scope,
        )
        materialized = materialize_tier_cohort(
            registry=registry,
            benchmark=benchmark,
            tier_name=tier_name,
        )
    except (OSError, ValueError) as error:
        print(f"Benchmark tier preflight failed: {error}")
        raise SystemExit(1) from None

    print("\nEvalanche tier campaign")
    print(f"Campaign: {preflight['campaign_id']}")
    print(f"Benchmark: {benchmark_reference}")
    print(f"Tier: {tier_name}")
    print(f"Cumulative cases: {materialized['plan']['cumulative']['cases']}")
    print(f"New tier cases: {materialized['plan']['execution']['cases']}")
    if promotion is not None:
        print(f"Promotion source: {promote_from}")
    print("\nModel cost preflight")
    for row in preflight["rows"]:
        print(
            f"- {row['model_id']}: {row['pending_requests']} pending call(s), "
            f"${float(row['projected_budgeted_total_usd']):.4f} USD "
            f"budgeted, {row['cost_sample_source']}"
        )
    print(
        "Aggregate budgeted projection: "
        f"${preflight['projected_budgeted_total_usd']:.4f} USD"
    )
    print(
        "Campaign limit: "
        + (
            f"${maximum_cost_usd:.4f} USD"
            if maximum_cost_usd is not None
            else "not set"
        )
    )
    print(f"Campaign plan: {preflight['paths']['campaign']}")

    if preflight["within_budget"] is False:
        print("Status: BLOCKED BY AGGREGATE BUDGET")
        print("No model calls were made.")
        raise SystemExit(2)
    if plan_only or preflight_only:
        print("Status: PREFLIGHT READY")
        print("No model calls were made.")
        return {"preflight": preflight, "execution": None}
    if preflight["pending_models"] == 0:
        summary = build_benchmark_experiment_summary(
            registry=registry,
            benchmark=benchmark,
            model_ids=selected_ids,
            tier_name=tier_name,
            access_scope=access_scope,
        )
        print_benchmark_experiment_summary(summary)
        print("Status: TIER ALREADY COMPLETE")
        print("No model calls were made.")
        return {"preflight": preflight, "execution": None}
    if not experiment:
        print(
            "Tier campaigns are local comparisons. Add --experiment to "
            "execute without publishing a leaderboard."
        )
        raise SystemExit(1)
    if maximum_cost_usd is None:
        print(
            "Tier execution requires --max-cost-usd. Run --preflight-only "
            "first to choose the cap."
        )
        raise SystemExit(1)

    try:
        execution = execute_tier_campaign(
            registry=registry,
            preflight=preflight,
        )
    except (OSError, ValueError) as error:
        print(f"Benchmark tier campaign stopped safely: {error}")
        raise SystemExit(2) from None
    print("\nTier campaign complete")
    print(
        "Models completed now: " + (", ".join(execution["completed_models"]) or "none")
    )
    print(f"Observed budgeted cost: ${execution['observed_budgeted_cost_usd']:.4f} USD")
    print(f"Campaign record: {execution['paths']['campaign']}")
    print("Status: LOCAL EXPERIMENT COMPLETE, NOT REGISTERED")
    return {"preflight": preflight, "execution": execution}


def run_benchmark_from_registry(
    *,
    root_path: str,
    benchmark_reference: str,
    model_ids: list[str] | None,
    all_compatible: bool,
    plan_only: bool,
    experiment: bool,
    tier_name: str | None = None,
    promote_from: str | None = None,
    preflight_only: bool = False,
    maximum_cost_usd: float | None = None,
    access_set_path: str | None = None,
) -> dict[str, Any]:
    task_profile_path = Path(root_path).resolve() / "task.yaml"
    if task_profile_path.is_file() and not plan_only:
        blocker = task_execution_blocker(
            root_path,
            require_paid_execution=not preflight_only,
        )
        if blocker is not None:
            print(f"Benchmark run blocked: {blocker}")
            raise SystemExit(1)
        if experiment and tier_name is None:
            print(
                "Benchmark run blocked: generated task bundles must use a "
                "manifest-defined tier with aggregate cost preflight."
            )
            raise SystemExit(1)
    if tier_name is not None:
        return run_tier_campaign_from_registry(
            root_path=root_path,
            benchmark_reference=benchmark_reference,
            tier_name=tier_name,
            model_ids=model_ids,
            all_compatible=all_compatible,
            promote_from=promote_from,
            plan_only=plan_only,
            preflight_only=preflight_only,
            experiment=experiment,
            maximum_cost_usd=maximum_cost_usd,
            access_set_path=access_set_path,
        )
    if promote_from is not None or preflight_only or maximum_cost_usd is not None:
        print("--promote-from, --preflight-only, and --max-cost-usd require --tier.")
        raise SystemExit(1)
    try:
        registry = load_registry(root_path)
        benchmark = registry.resolve_benchmark(benchmark_reference)
        required_capabilities = set(benchmark.required_capabilities)
        selected_ids, access_scope = _resolve_access_confirmed_models(
            registry=registry,
            benchmark=benchmark,
            model_ids=model_ids,
            all_compatible=all_compatible,
            access_set_path=access_set_path,
        )
        if not selected_ids:
            raise ValueError("No compatible models were selected.")
        models = [registry.resolve_model(model_id) for model_id in selected_ids]
        incompatible = [
            model.model_id
            for model in models
            if not required_capabilities.issubset(model.capabilities)
        ]
        if incompatible:
            raise ValueError(
                f"Selected models do not satisfy benchmark capabilities: {incompatible}"
            )

        results = []
        for model in models:
            if experiment and not plan_only:
                result = run_experimental_benchmark(
                    registry=registry,
                    benchmark=benchmark,
                    model=model,
                )
            else:
                result = run_registered_benchmark(
                    registry=registry,
                    benchmark=benchmark,
                    model=model,
                    plan_only=plan_only,
                )
            results.append(result)
    except (OSError, ValueError) as error:
        print(f"Benchmark run failed: {error}")
        raise SystemExit(1) from None

    print("\nEvalanche benchmark runs")
    for result in results:
        plan = result["plan"]
        print(f"\nBenchmark: {plan['benchmark']}")
        print(f"Model: {plan['model_id']}")
        print(f"Cases: {plan['case_count']}")
        print(f"Planned model calls: {plan['planned_model_calls']}")
        print(f"Plan: {plan['paths']['run_plan']}")
        if plan_only:
            print("Status: PLAN READY")
        elif experiment:
            path = result["evaluation_path"].relative_to(registry.root)
            print(f"Evaluation: {path}")
            print("Status: LOCAL EXPERIMENT, NOT REGISTERED")
        else:
            print(f"Run: {result['bundle']['run_id']}")
            print(f"Leaderboard: {result['leaderboard']['html_path']}")
            print("Status: COMPLETE AND REGISTERED")

    experiment_summary = None
    if plan_only:
        print("\nNo model calls were made.")
    elif experiment:
        try:
            experiment_summary = build_benchmark_experiment_summary(
                registry=registry,
                benchmark=benchmark,
                model_ids=selected_ids,
                access_scope=access_scope,
            )
        except (OSError, ValueError) as error:
            print(f"Experiment summary failed: {error}")
            raise SystemExit(1) from None
        print_benchmark_experiment_summary(experiment_summary)
        report = experiment_summary["paths"]["report"].relative_to(registry.root)
        print(f"Experiment report: {report}")
        print("No leaderboard was published or changed.")
    return {
        "runs": results,
        "experiment_summary": experiment_summary,
    }


def run_summarize_benchmark(
    *,
    root_path: str,
    benchmark_reference: str,
    model_ids: list[str] | None,
    output_dir: str | None,
    tier_name: str | None = None,
    access_set_path: str | None = None,
    all_results: bool = False,
) -> dict[str, Any]:
    try:
        registry = load_registry(root_path)
        benchmark = registry.resolve_benchmark(benchmark_reference)
        if all_results:
            selected_ids = None
            access_scope = historical_results_scope()
        elif access_set_path is not None:
            resolution = resolve_access_set(
                registry=registry,
                path=access_set_path,
                benchmark=benchmark,
            )
            selected_ids = resolution["compatible_model_ids"]
            access_scope = dict(resolution["scope"])
            access_scope["excluded_incompatible"] = resolution["incompatible_models"]
            if not selected_ids:
                raise ValueError(
                    "No access-confirmed models in the access set satisfy "
                    "the benchmark capabilities."
                )
        elif model_ids:
            selected_ids = list(dict.fromkeys(model_ids))
            access_scope = explicit_model_scope(selected_ids)
        else:
            raise ValueError("Choose --model, --access-set, or --all-results.")
        result = build_benchmark_experiment_summary(
            registry=registry,
            benchmark=benchmark,
            model_ids=selected_ids,
            output_dir=output_dir,
            tier_name=tier_name,
            access_scope=access_scope,
        )
    except (OSError, ValueError) as error:
        print(f"Benchmark summary failed: {error}")
        raise SystemExit(1) from None

    print_benchmark_experiment_summary(result)
    print(
        "\nModels discovered: " + ", ".join(result["summary"]["model_name"].astype(str))
    )
    print(f"Status: {result['comparison_status'].upper()}")
    if tier_name is not None:
        print(f"Tier: {tier_name}")
    print(f"Report: {result['paths']['report']}")
    print("No model calls were made.")
    return result


def run_rescore_benchmark_from_registry(
    *,
    root_path: str,
    benchmark_reference: str,
    model_id: str,
) -> dict[str, Any]:
    try:
        registry = load_registry(root_path)
        benchmark = registry.resolve_benchmark(benchmark_reference)
        model = registry.resolve_model(model_id)
        result = rescore_registered_benchmark(
            registry=registry,
            benchmark=benchmark,
            model=model,
        )
    except (OSError, ValueError) as error:
        print(f"Benchmark rescore failed: {error}")
        raise SystemExit(1) from None

    plan = result["plan"]
    print("\nEvalanche benchmark rescore")
    print(f"Benchmark: {plan['benchmark']}")
    print(f"Model: {plan['model_id']}")
    print(f"Cases: {plan['case_count']}")
    print("Model generation calls: 0")
    print(f"Run: {result['bundle']['run_id']}")
    print(f"Leaderboard: {result['leaderboard']['html_path']}")
    print("Status: RESCORED AND REGISTERED")
    return result


def run_verify_product_monograph_sources(
    *,
    root_path: str,
    sources_path: str,
    raw_dir: str,
) -> dict[str, Any]:
    try:
        result = verify_product_monograph_sources(
            root_path=root_path,
            sources_path=sources_path,
            raw_dir=raw_dir,
        )
    except (OSError, ValueError) as error:
        print(f"Product Monograph source verification failed: {error}")
        raise SystemExit(1) from None

    print("\nProduct Monograph source lock")
    print(f"Documents: {result['documents']}")
    print(
        "Languages: "
        + ", ".join(
            f"{language}={count}"
            for language, count in sorted(result["languages"].items())
        )
    )
    print(f"Unique hashes: {result['unique_hashes']}")
    if not result["valid"]:
        print("Status: INVALID")
        for issue in result["issues"]:
            print(f"- {issue}")
        raise SystemExit(1)
    print("Status: VALID")
    return result


def run_acquire_product_monographs(
    *,
    root_path: str,
    sources_path: str,
    raw_dir: str,
    timeout: float,
) -> dict[str, Any]:
    try:
        result = acquire_product_monographs(
            root_path=root_path,
            sources_path=sources_path,
            raw_dir=raw_dir,
            timeout=timeout,
        )
    except (OSError, ValueError) as error:
        print(f"Product Monograph acquisition failed: {error}")
        raise SystemExit(1) from None

    print("\nProduct Monograph acquisition")
    print(f"Documents: {result['documents']}")
    print(f"Downloaded: {result['downloaded']}")
    print(f"Reused: {result['reused']}")
    print(f"Output: {result['raw_dir']}")
    print("Status: VALID")
    return result


def run_build_product_monograph(
    *,
    root_path: str,
    cohort_path: str,
    overrides_path: str,
    sources_path: str,
    raw_dir: str,
) -> dict[str, Any]:
    try:
        result = build_product_monograph_benchmark(
            root_path=root_path,
            cohort_path=cohort_path,
            overrides_path=overrides_path,
            sources_path=sources_path,
            raw_dir=raw_dir,
        )
    except (OSError, ValueError) as error:
        print(f"Product Monograph benchmark build failed: {error}")
        raise SystemExit(1) from None

    verification = result["verification"]
    print("\nHealth Canada Product Monograph benchmark")
    print(f"Dataset: {result['dataset_id']} {result['dataset_version']}")
    print(f"Products: {result['product_count']}")
    print(f"Cases: {result['case_count']}")
    print(f"Evidence items: {result['evidence_item_count']}")
    print(f"Output: {result['output_dir']}")
    print(f"Manifest: {result['manifest_path']}")
    print(
        f"Verified files: {verification['files_passed']}/"
        f"{verification['files_checked']}"
    )
    print("Status: VALID")
    print("Independent human sign-off: NOT CLAIMED")
    return result


def run_build_product_monograph_native_pdf(
    *,
    root_path: str,
    source_cases_path: str,
    sources_path: str,
    raw_dir: str,
) -> dict[str, Any]:
    try:
        result = build_product_monograph_native_pdf_benchmark(
            root_path=root_path,
            source_cases_path=source_cases_path,
            sources_path=sources_path,
            raw_dir=raw_dir,
        )
    except (OSError, ValueError) as error:
        print(f"Native-PDF benchmark build failed: {error}")
        raise SystemExit(1) from None

    verification = result["verification"]
    print("\nHealth Canada Product Monograph native-PDF benchmark")
    print(f"Dataset: {result['dataset_id']} {result['dataset_version']}")
    print(f"Products: {result['product_count']}")
    print(f"Cases: {result['case_count']}")
    print(
        "Approved label items: "
        f"{result['approved_review_item_count']}/"
        f"{result['review_item_count']}"
    )
    print(f"Output: {result['output_dir']}")
    print(f"Manifest: {result['manifest_path']}")
    print(
        f"Verified files: {verification['files_passed']}/"
        f"{verification['files_checked']}"
    )
    print("Status: DRAFT, NOT RANKABLE")
    return result


def run_check_product_monograph_label_review(
    *,
    root_path: str,
    review_path: str,
    require_complete: bool,
) -> dict[str, Any]:
    try:
        result = check_product_monograph_label_review(
            root_path=root_path,
            review_path=review_path,
        )
    except (OSError, ValueError) as error:
        print(f"Product Monograph label review check failed: {error}")
        raise SystemExit(1) from None

    print("\nProduct Monograph native-PDF label review")
    print(f"Items reviewed: {result['reviewed']}/{result['items']}")
    print(f"Items approved: {result['approved']}/{result['items']}")
    print(f"Cases fully approved: {result['cases_approved']}/{result['cases']}")
    print(
        "Statuses: "
        + ", ".join(
            f"{status}={count}" for status, count in result["status_counts"].items()
        )
    )
    print(f"Review file: {result['review_path']}")
    if result["issues"]:
        print("Status: INVALID REVIEW METADATA")
        for issue in result["issues"]:
            print(f"- {issue}")
        raise SystemExit(1)
    if result["promotion_ready"]:
        print("Status: LABEL REVIEW COMPLETE")
    else:
        print("Status: REVIEW INCOMPLETE")
        print(f"Remaining approvals: {result['remaining']}. No model calls were made.")
        if require_complete:
            raise SystemExit(2)
    return result


def run_expand_product_monograph_benchmark(
    *,
    root_path: str,
    config_path: str,
    max_candidates: int | None,
    show_progress: bool,
) -> dict[str, Any]:
    try:
        result = expand_product_monograph_benchmark(
            root_path=root_path,
            config_path=config_path,
            max_candidates=max_candidates,
            progress=print if show_progress else None,
        )
    except (OSError, ValueError) as error:
        print(f"Product Monograph expansion failed: {error}")
        raise SystemExit(1) from None

    print("\nHealth Canada Product Monograph expansion")
    print(f"Version: {result['version']}")
    print(f"Products: {result['products']}")
    print(f"Evidence-window cases: {result['cases']}")
    print(f"Native-PDF cases: {result['native_pdf_cases']}")
    print(f"Provisional fact items: {result['facts']}")
    print(f"Candidates screened: {result['screened_candidates']}")
    print(f"Candidates excluded: {result['screening_exclusions']}")
    print(f"Output: {result['output_dir']}")
    print("Status: DRAFT, HUMAN AUDIT REQUIRED")
    print("Model calls: 0")
    return result


def run_refresh_product_monograph_expansion(
    *,
    root_path: str,
    config_path: str,
) -> dict[str, Any]:
    try:
        result = refresh_expanded_product_monograph_release(
            root_path=root_path,
            config_path=config_path,
        )
    except (OSError, ValueError) as error:
        print(f"Product Monograph expansion refresh failed: {error}")
        raise SystemExit(1) from None

    print("\nProduct Monograph expansion audit refresh")
    print(f"Version: {result['version']}")
    print(f"Products: {result['products']}")
    print(f"Cases: {result['cases']}")
    print(f"Fact items: {result['facts']}")
    for label, verification in result["verifications"].items():
        print(
            f"{label} files verified: {verification['files_passed']}/"
            f"{verification['files_checked']}"
        )
    print("Network calls: 0")
    print("Model calls: 0")
    print("Status: REFRESHED")
    return result


def run_check_product_monograph_expansion_audit(
    *,
    root_path: str,
    require_complete: bool,
) -> dict[str, Any]:
    try:
        result = check_expanded_product_monograph_audit(root_path=root_path)
    except (OSError, ValueError) as error:
        print(f"Product Monograph expansion audit check failed: {error}")
        raise SystemExit(1) from None

    fact_approved = int(result["fact_status_counts"].get("approved", 0))
    product_approved = sum(
        int(counts.get("approved", 0))
        for counts in result["product_status_counts"].values()
    )
    total_checks = int(result["fact_items"]) + 3 * int(result["products"])
    approved_checks = fact_approved + product_approved
    print("\nProduct Monograph expansion human audit")
    print(f"Fact approvals: {fact_approved}/{result['fact_items']}")
    print(
        f"Product identity, scope, and bilingual approvals: "
        f"{product_approved}/{3 * int(result['products'])}"
    )
    print(f"Total approvals: {approved_checks}/{total_checks}")
    print(f"Fact audit: {result['fact_audit_path']}")
    print(f"Product audit: {result['product_audit_path']}")
    if result["issues"]:
        print("Status: INVALID AUDIT OR ARTIFACT METADATA")
        for issue in result["issues"]:
            print(f"- {issue}")
        raise SystemExit(1)
    if result["promotion_ready"]:
        print("Status: HUMAN AUDIT COMPLETE")
    else:
        print("Status: DRAFT, REVIEW INCOMPLETE")
        print(f"Remaining approvals: {total_checks - approved_checks}")
        if require_complete:
            raise SystemExit(2)
    print("Model calls: 0")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalanche")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evidence_parser = subparsers.add_parser(
        "evidence-status",
        help="Validate evidence metadata and report review freshness offline.",
    )
    evidence_parser.add_argument(
        "--root",
        default=".",
        help="Evalanche repository root.",
    )
    evidence_parser.add_argument(
        "--catalog",
        default="docs/evidence/catalog.yaml",
        help="Repository-relative evidence catalog path.",
    )
    evidence_parser.add_argument(
        "--as-of",
        help="Evaluate freshness on this YYYY-MM-DD date. Defaults to today.",
    )
    evidence_parser.add_argument(
        "--output",
        default="docs/evidence/status.md",
        help="Repository-relative Markdown status snapshot path.",
    )
    evidence_output_group = evidence_parser.add_mutually_exclusive_group()
    evidence_output_group.add_argument(
        "--write",
        action="store_true",
        help="Write the generated Markdown status snapshot.",
    )
    evidence_output_group.add_argument(
        "--check",
        action="store_true",
        help=("Confirm the saved snapshot matches the catalog's recorded status date."),
    )
    evidence_parser.add_argument(
        "--fail-on-outdated",
        action="store_true",
        help="Exit with status 2 if any evidence is stale or superseded.",
    )

    init_task_parser = subparsers.add_parser(
        "init-task",
        help="Create an offline starter bundle for a user-owned task.",
    )
    init_task_parser.add_argument(
        "--root",
        default=".",
        help="Evalanche repository root.",
    )
    init_task_parser.add_argument(
        "--output-dir",
        default="local_tasks",
        help="Repository-relative parent directory for local task bundles.",
    )
    init_task_parser.add_argument("--task-id")
    init_task_parser.add_argument("--description")
    init_task_parser.add_argument(
        "--family",
        help=(
            "classification, structured-extraction, factual-response, or "
            "open-ended-response"
        ),
    )
    init_task_parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        help="Required language tag. Repeat for more than one.",
    )
    init_task_parser.add_argument(
        "--slice",
        action="append",
        dest="slice_columns",
        help="Additional lowercase slice column. Repeat as needed.",
    )
    init_task_parser.add_argument(
        "--unacceptable-error",
        action="append",
        dest="unacceptable_errors",
        help="An error that makes an output unusable. Repeat as needed.",
    )
    init_task_parser.add_argument(
        "--model",
        action="append",
        dest="model_routes",
        help=(
            "Explicitly available MODEL_ID=PROVIDER_ROUTE. Repeat for each "
            "confirmed route."
        ),
    )
    init_task_parser.add_argument(
        "--output-field",
        action="append",
        dest="required_output_fields",
        help="Required JSON field for structured extraction.",
    )
    init_task_parser.add_argument(
        "--criterion",
        action="append",
        help=("Open-ended criterion as NAME=WEIGHT=DESCRIPTION. Repeat as needed."),
    )
    init_task_parser.add_argument("--measured-construct")
    init_task_parser.add_argument("--intended-use")
    init_task_parser.add_argument("--sample-input")
    init_task_parser.add_argument("--sample-expected-output")
    init_task_parser.add_argument(
        "--artifact-policy",
        choices=("local_only", "shareable"),
    )
    init_task_parser.add_argument(
        "--contains-sensitive-data",
        action="store_true",
        help=(
            "Generate placeholders only and record that real cases require "
            "additional handling review."
        ),
    )
    init_task_parser.add_argument("--privacy-notes")
    init_task_parser.add_argument(
        "--starting-tier",
        choices=("smoke", "screen"),
    )
    init_task_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Require core values as arguments instead of prompting.",
    )
    init_task_parser.add_argument(
        "--replace",
        action="store_true",
        help="Preserve the existing bundle as a timestamped backup.",
    )

    validate_task_parser = subparsers.add_parser(
        "validate-task",
        help="Validate a generated task bundle without provider calls.",
    )
    validate_task_parser.add_argument(
        "--root",
        required=True,
        help="Generated task-bundle root.",
    )
    validate_task_parser.add_argument(
        "--require-cost-ready",
        action="store_true",
        help="Also require verified pricing for every selected model.",
    )

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
    generate_parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate resume state and projected configured cost without "
            "making model calls."
        ),
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

    validate_judge_parser = subparsers.add_parser("validate-judge")
    validate_judge_parser.add_argument(
        "--protocol",
        required=True,
        help="Path to a versioned judge validation protocol YAML file.",
    )
    validate_judge_parser.add_argument(
        "--root",
        default=".",
        help="Root used to resolve protocol data paths.",
    )
    validate_judge_parser.add_argument(
        "--output-dir",
        help=(
            "Optional output directory. The default is a versioned path "
            "under results/judge_validation."
        ),
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

    dpd_benchmark_parser = subparsers.add_parser("build-dpd-benchmark")
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

    dpd_census_parser = subparsers.add_parser("build-dpd-census")
    dpd_census_parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing the frozen DPD snapshot.",
    )
    dpd_census_parser.add_argument(
        "--source-manifest",
        default=DPD_SOURCE_MANIFEST.as_posix(),
        help="Path to the frozen DPD source manifest.",
    )

    dpd_comparison_parser = subparsers.add_parser("build-dpd-comparison")
    dpd_comparison_parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing the frozen DPD census.",
    )
    dpd_comparison_parser.add_argument(
        "--seed",
        type=int,
        default=DPD_COMPARISON_SEED,
        help="Recorded seed used for deterministic sample selection.",
    )
    dpd_comparison_parser.add_argument(
        "--products",
        type=int,
        default=DPD_COMPARISON_PRODUCT_COUNT,
        help=(
            "Number of product families to sample. Each contributes one "
            "English and one French case."
        ),
    )

    assemble_comparison_parser = subparsers.add_parser("assemble-dpd-comparison")
    assemble_comparison_parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing completed generation outputs.",
    )
    assemble_comparison_parser.add_argument(
        "--require-model",
        action="append",
        dest="required_models",
        help=("Model name that must be present. Repeat to require multiple models."),
    )

    assemble_census_parser = subparsers.add_parser("assemble-dpd-census-comparison")
    assemble_census_parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing completed census outputs.",
    )
    assemble_census_parser.add_argument(
        "--require-model",
        action="append",
        dest="required_models",
        help=(
            "Model name that must have a complete 14,034-case output. "
            "Repeat to require multiple models."
        ),
    )

    analyze_census_parser = subparsers.add_parser("analyze-dpd-census")
    analyze_census_parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing results and published reports.",
    )
    analyze_census_parser.add_argument(
        "--results",
        default=DPD_ANALYSIS_RESULTS_PATH.as_posix(),
        help="Combined case-level DPD census result CSV.",
    )
    analyze_census_parser.add_argument(
        "--config",
        default=DPD_ANALYSIS_CONFIG_PATH.as_posix(),
        help="Evaluation config that defines canonical JSON scoring.",
    )
    analyze_census_parser.add_argument(
        "--output-dir",
        default=DPD_ANALYSIS_OUTPUT_DIR.as_posix(),
        help="Directory for compact analysis and review artifacts.",
    )
    analyze_census_parser.add_argument(
        "--primary-model",
        default=DPD_ANALYSIS_PRIMARY_MODEL,
        help="Frontier model whose every failure must be reviewed.",
    )
    analyze_census_parser.add_argument(
        "--comparison-model",
        default=DPD_ANALYSIS_COMPARISON_MODEL,
        help="Frontier comparator for directional disagreement review.",
    )
    analyze_census_parser.add_argument(
        "--shared-failure-sample",
        type=int,
        default=DPD_ANALYSIS_SHARED_FAILURE_SAMPLE,
        help="Stratified shared lower-tier failure sample size.",
    )
    analyze_census_parser.add_argument(
        "--all-pass-sample",
        type=int,
        default=DPD_ANALYSIS_ALL_PASS_SAMPLE,
        help="Stratified all-model pass sample size.",
    )
    analyze_census_parser.add_argument(
        "--review-seed",
        type=int,
        default=DPD_ANALYSIS_REVIEW_SEED,
        help="Recorded seed for deterministic review samples.",
    )

    audit_review_parser = subparsers.add_parser("audit-dpd-review")
    audit_review_parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing the frozen DPD evidence.",
    )
    audit_review_parser.add_argument(
        "--review",
        default=DPD_EVIDENCE_AUDIT_REVIEW_PATH.as_posix(),
        help="Generated DPD review worksheet.",
    )
    audit_review_parser.add_argument(
        "--config",
        default=DPD_EVIDENCE_AUDIT_CONFIG_PATH.as_posix(),
        help="Evaluation config defining canonical JSON scoring.",
    )
    audit_review_parser.add_argument(
        "--archive",
        default=DPD_EVIDENCE_AUDIT_ARCHIVE_PATH.as_posix(),
        help="Frozen marketed DPD source archive.",
    )
    audit_review_parser.add_argument(
        "--output",
        default=DPD_EVIDENCE_AUDIT_OUTPUT_PATH.as_posix(),
        help="Completed automated evidence-audit CSV.",
    )
    audit_review_parser.add_argument(
        "--summary",
        default=DPD_EVIDENCE_AUDIT_SUMMARY_PATH.as_posix(),
        help="Evidence-audit summary JSON.",
    )

    registry_parser = subparsers.add_parser("registry-validate")
    registry_parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing model and benchmark manifests.",
    )

    register_parser = subparsers.add_parser("register-result")
    register_parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing the registry.",
    )
    register_parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark reference in benchmark_id@version form.",
    )
    register_parser.add_argument("--model", required=True)
    register_parser.add_argument(
        "--results",
        required=True,
        help="Completed single-model evaluation CSV.",
    )
    register_parser.add_argument(
        "--provenance",
        help=(
            "Optional original result artifact recorded as provenance when "
            "--results is a temporary single-model migration view."
        ),
    )

    leaderboard_parser = subparsers.add_parser("build-leaderboard")
    leaderboard_parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing the registry.",
    )
    leaderboard_group = leaderboard_parser.add_mutually_exclusive_group(required=True)
    leaderboard_group.add_argument(
        "--benchmark",
        help="Benchmark reference in benchmark_id@version form.",
    )
    leaderboard_group.add_argument(
        "--all",
        action="store_true",
        help="Build every registered benchmark leaderboard.",
    )

    run_benchmark_parser = subparsers.add_parser("run-benchmark")
    run_benchmark_parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing the registry.",
    )
    run_benchmark_parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark reference in benchmark_id@version form.",
    )
    run_model_group = run_benchmark_parser.add_mutually_exclusive_group(required=True)
    run_model_group.add_argument(
        "--model",
        action="append",
        dest="models",
        help=(
            "Registered model ID. Repeat to run multiple models in one "
            "command. Each ID is an explicit access confirmation for this "
            "run."
        ),
    )
    run_model_group.add_argument(
        "--all-compatible",
        action="store_true",
        help=(
            "Use every compatible model in --access-set. The access set is "
            "required because compatibility does not prove availability."
        ),
    )
    run_model_group.add_argument(
        "--promote-from",
        metavar="TIER",
        help=(
            "Select every completed model that passes the named tier's "
            "manifest-defined promotion gates. Requires --tier and "
            "--access-set."
        ),
    )
    run_benchmark_parser.add_argument(
        "--access-set",
        help=(
            "YAML file confirming which registered deployment routes are "
            "available. Required with --all-compatible or --promote-from."
        ),
    )
    run_benchmark_parser.add_argument(
        "--tier",
        help=(
            "Run a manifest-defined cumulative tier. Tier execution is "
            "incremental and remains a local experiment."
        ),
    )
    execution_group = run_benchmark_parser.add_mutually_exclusive_group()
    execution_group.add_argument(
        "--plan-only",
        action="store_true",
        help="Resolve and validate the run without making model calls.",
    )
    execution_group.add_argument(
        "--experiment",
        action="store_true",
        help=(
            "Execute locally without registration or leaderboard changes. "
            "This is the only execution mode allowed for a draft benchmark."
        ),
    )
    execution_group.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Materialize the tier, inspect resume state, and estimate the "
            "aggregate cost without making model calls."
        ),
    )
    run_benchmark_parser.add_argument(
        "--max-cost-usd",
        type=float,
        help=(
            "Maximum budgeted USD for all selected models in this tier. "
            "Required for tier execution."
        ),
    )

    summarize_parser = subparsers.add_parser("summarize-benchmark")
    summarize_parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing run plans and local results.",
    )
    summarize_parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark reference in benchmark_id@version form.",
    )
    summarize_scope_group = summarize_parser.add_mutually_exclusive_group(required=True)
    summarize_scope_group.add_argument(
        "--model",
        action="append",
        dest="models",
        help=("Include this access-confirmed model. Repeat for more models."),
    )
    summarize_scope_group.add_argument(
        "--access-set",
        help=(
            "Include compatible completed results for the deployments in "
            "this access set."
        ),
    )
    summarize_scope_group.add_argument(
        "--all-results",
        action="store_true",
        help=(
            "Discover every compatible completed result as historical "
            "evidence without asserting current model access."
        ),
    )
    summarize_parser.add_argument(
        "--output-dir",
        help=(
            "Optional repository-relative output directory. The default is "
            "under ignored results/benchmark_experiments/."
        ),
    )
    summarize_parser.add_argument(
        "--tier",
        help=("Summarize the cumulative cohort for this manifest-defined tier."),
    )

    rescore_benchmark_parser = subparsers.add_parser("rescore-benchmark")
    rescore_benchmark_parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing the registry.",
    )
    rescore_benchmark_parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark reference in benchmark_id@version form.",
    )
    rescore_benchmark_parser.add_argument("--model", required=True)

    verify_pm_parser = subparsers.add_parser("verify-product-monograph-sources")
    verify_pm_parser.add_argument("--root", default=".")
    verify_pm_parser.add_argument(
        "--sources",
        default=PM_SOURCES_PATH.as_posix(),
    )
    verify_pm_parser.add_argument(
        "--raw-dir",
        default=PM_RAW_DIR.as_posix(),
    )

    acquire_pm_parser = subparsers.add_parser("acquire-product-monographs")
    acquire_pm_parser.add_argument("--root", default=".")
    acquire_pm_parser.add_argument(
        "--sources",
        default=PM_SOURCES_PATH.as_posix(),
    )
    acquire_pm_parser.add_argument(
        "--raw-dir",
        default=PM_RAW_DIR.as_posix(),
    )
    acquire_pm_parser.add_argument("--timeout", type=float, default=120.0)

    build_pm_parser = subparsers.add_parser("build-product-monograph-benchmark")
    build_pm_parser.add_argument("--root", default=".")
    build_pm_parser.add_argument(
        "--cohort",
        default=PM_COHORT_PATH.as_posix(),
    )
    build_pm_parser.add_argument(
        "--overrides",
        default=PM_OVERRIDES_PATH.as_posix(),
    )
    build_pm_parser.add_argument(
        "--sources",
        default=PM_SOURCES_PATH.as_posix(),
    )
    build_pm_parser.add_argument(
        "--raw-dir",
        default=PM_RAW_DIR.as_posix(),
    )

    build_pm_pdf_parser = subparsers.add_parser(
        "build-product-monograph-native-pdf-benchmark"
    )
    build_pm_pdf_parser.add_argument("--root", default=".")
    build_pm_pdf_parser.add_argument(
        "--source-cases",
        default=(
            "data/hc/benchmarks/product_monograph_structured_extraction/"
            "0.1.0/cases.csv.gz"
        ),
    )
    build_pm_pdf_parser.add_argument(
        "--sources",
        default=PM_SOURCES_PATH.as_posix(),
    )
    build_pm_pdf_parser.add_argument(
        "--raw-dir",
        default=PM_RAW_DIR.as_posix(),
    )

    check_pm_review_parser = subparsers.add_parser(
        "check-product-monograph-label-review"
    )
    check_pm_review_parser.add_argument("--root", default=".")
    check_pm_review_parser.add_argument(
        "--review",
        default=(
            "data/hc/benchmarks/"
            "product_monograph_native_pdf_extraction/0.1.0/"
            "label_review.csv"
        ),
        help="Review CSV whose human-only columns were edited.",
    )
    check_pm_review_parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit unsuccessfully unless every label item is approved.",
    )

    expand_pm_parser = subparsers.add_parser(
        "expand-product-monograph-benchmark",
        help=("Build the complete 1.0.0 bilingual expansion and audit package."),
    )
    expand_pm_parser.add_argument("--root", default=".")
    expand_pm_parser.add_argument(
        "--config",
        default=EXPANSION_CONFIG_PATH.as_posix(),
    )
    expand_pm_parser.add_argument(
        "--max-candidates",
        type=int,
        help="Optional screening cap for failure-path testing.",
    )
    expand_pm_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-candidate screening progress.",
    )

    refresh_pm_parser = subparsers.add_parser(
        "refresh-product-monograph-expansion",
        help="Refresh audit summaries and artifact hashes without network calls.",
    )
    refresh_pm_parser.add_argument("--root", default=".")
    refresh_pm_parser.add_argument(
        "--config",
        default=EXPANSION_CONFIG_PATH.as_posix(),
    )

    check_pm_expansion_parser = subparsers.add_parser(
        "check-product-monograph-expansion-audit",
        help="Validate expanded fact, product, and artifact audit gates.",
    )
    check_pm_expansion_parser.add_argument("--root", default=".")
    check_pm_expansion_parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit unsuccessfully unless every fact and product check is approved.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "evidence-status":
        run_evidence_status(
            root_path=args.root,
            catalog_path=args.catalog,
            as_of_value=args.as_of,
            output_path=args.output,
            write=args.write,
            check=args.check,
            fail_on_outdated=args.fail_on_outdated,
        )
    elif args.command == "init-task":
        run_init_task(
            root_path=args.root,
            output_dir=args.output_dir,
            task_id=args.task_id,
            description=args.description,
            family=args.family,
            languages=args.languages,
            slice_columns=args.slice_columns,
            unacceptable_errors=args.unacceptable_errors,
            model_routes=args.model_routes,
            required_output_fields=args.required_output_fields,
            criteria=args.criterion,
            measured_construct=args.measured_construct,
            intended_use=args.intended_use,
            sample_input=args.sample_input,
            sample_expected_output=args.sample_expected_output,
            artifact_policy=args.artifact_policy,
            contains_sensitive_data=args.contains_sensitive_data,
            privacy_notes=args.privacy_notes,
            starting_tier=args.starting_tier,
            non_interactive=args.non_interactive,
            replace=args.replace,
        )
    elif args.command == "validate-task":
        run_validate_task(
            root_path=args.root,
            require_cost_ready=args.require_cost_ready,
        )
    elif args.command == "generate":
        run_generate(
            args.config,
            preflight_only=args.preflight_only,
        )
    elif args.command == "metrics":
        run_metrics(args.config)
    elif args.command == "judge":
        run_judge(args.config)
    elif args.command == "validate-judge":
        run_validate_judge(
            args.protocol,
            root_path=args.root,
            output_dir=args.output_dir,
        )
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
    elif args.command == "build-dpd-census":
        run_build_dpd_census(
            root_path=args.root,
            source_manifest_path=args.source_manifest,
        )
    elif args.command == "build-dpd-comparison":
        run_build_dpd_comparison(
            root_path=args.root,
            seed=args.seed,
            product_count=args.products,
        )
    elif args.command == "assemble-dpd-comparison":
        run_assemble_dpd_comparison(
            root_path=args.root,
            required_models=args.required_models,
        )
    elif args.command == "assemble-dpd-census-comparison":
        run_assemble_dpd_census_comparison(
            root_path=args.root,
            required_models=args.required_models,
        )
    elif args.command == "analyze-dpd-census":
        run_analyze_dpd_census(
            root_path=args.root,
            results_path=args.results,
            config_path=args.config,
            output_dir=args.output_dir,
            primary_model=args.primary_model,
            comparison_model=args.comparison_model,
            shared_failure_sample=args.shared_failure_sample,
            all_pass_sample=args.all_pass_sample,
            review_seed=args.review_seed,
        )
    elif args.command == "audit-dpd-review":
        run_audit_dpd_review(
            root_path=args.root,
            review_path=args.review,
            config_path=args.config,
            archive_path=args.archive,
            output_path=args.output,
            summary_path=args.summary,
        )
    elif args.command == "registry-validate":
        run_registry_validate(root_path=args.root)
    elif args.command == "register-result":
        run_register_result(
            root_path=args.root,
            benchmark_reference=args.benchmark,
            model_id=args.model,
            results_path=args.results,
            provenance_path=args.provenance,
        )
    elif args.command == "build-leaderboard":
        run_build_leaderboard(
            root_path=args.root,
            benchmark_reference=args.benchmark,
            build_all=args.all,
        )
    elif args.command == "run-benchmark":
        run_benchmark_from_registry(
            root_path=args.root,
            benchmark_reference=args.benchmark,
            model_ids=args.models,
            all_compatible=args.all_compatible,
            plan_only=args.plan_only,
            experiment=args.experiment,
            tier_name=args.tier,
            promote_from=args.promote_from,
            preflight_only=args.preflight_only,
            maximum_cost_usd=args.max_cost_usd,
            access_set_path=args.access_set,
        )
    elif args.command == "summarize-benchmark":
        run_summarize_benchmark(
            root_path=args.root,
            benchmark_reference=args.benchmark,
            model_ids=args.models,
            output_dir=args.output_dir,
            tier_name=args.tier,
            access_set_path=args.access_set,
            all_results=args.all_results,
        )
    elif args.command == "rescore-benchmark":
        run_rescore_benchmark_from_registry(
            root_path=args.root,
            benchmark_reference=args.benchmark,
            model_id=args.model,
        )
    elif args.command == "verify-product-monograph-sources":
        run_verify_product_monograph_sources(
            root_path=args.root,
            sources_path=args.sources,
            raw_dir=args.raw_dir,
        )
    elif args.command == "acquire-product-monographs":
        run_acquire_product_monographs(
            root_path=args.root,
            sources_path=args.sources,
            raw_dir=args.raw_dir,
            timeout=args.timeout,
        )
    elif args.command == "build-product-monograph-benchmark":
        run_build_product_monograph(
            root_path=args.root,
            cohort_path=args.cohort,
            overrides_path=args.overrides,
            sources_path=args.sources,
            raw_dir=args.raw_dir,
        )
    elif args.command == "build-product-monograph-native-pdf-benchmark":
        run_build_product_monograph_native_pdf(
            root_path=args.root,
            source_cases_path=args.source_cases,
            sources_path=args.sources,
            raw_dir=args.raw_dir,
        )
    elif args.command == "check-product-monograph-label-review":
        run_check_product_monograph_label_review(
            root_path=args.root,
            review_path=args.review,
            require_complete=args.require_complete,
        )
    elif args.command == "expand-product-monograph-benchmark":
        run_expand_product_monograph_benchmark(
            root_path=args.root,
            config_path=args.config,
            max_candidates=args.max_candidates,
            show_progress=not args.quiet,
        )
    elif args.command == "refresh-product-monograph-expansion":
        run_refresh_product_monograph_expansion(
            root_path=args.root,
            config_path=args.config,
        )
    elif args.command == "check-product-monograph-expansion-audit":
        run_check_product_monograph_expansion_audit(
            root_path=args.root,
            require_complete=args.require_complete,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
