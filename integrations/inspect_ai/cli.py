from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

from integrations.inspect_ai import (
    DEFAULT_BENCHMARK_REFERENCE,
    DEFAULT_HISTORICAL_OUTPUTS_PATH,
    DEFAULT_MODEL_ID,
    INSPECT_AI_VERSION,
    OPENAI_VERSION,
)
from integrations.inspect_ai.adapter import (
    build_parity_report,
    configure_inspect_azure_environment,
    inspect_model_route,
    json_safe,
    load_dpd_cases,
    load_dpd_context,
    load_dpd_historical_import,
    load_dpd_replay_records,
)

_CAMPAIGN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _resolve_models(context: Any, args: argparse.Namespace) -> list[str]:
    if args.all_models and args.models:
        raise ValueError("Choose either --all-models or --model, not both.")
    if args.all_models:
        return sorted(context.registry.models)
    model_ids = args.models or [DEFAULT_MODEL_ID]
    unknown = sorted(set(model_ids) - set(context.registry.models))
    if unknown:
        raise ValueError(f"Unknown model identifiers: {unknown}")
    return list(dict.fromkeys(model_ids))


def _resolve_historical_models(
    available_models: Sequence[str],
    args: argparse.Namespace,
) -> list[str]:
    if args.all_models and args.models:
        raise ValueError("Choose either --all-models or --model, not both.")
    model_ids = list(available_models) if not args.models else args.models
    unknown = sorted(set(model_ids) - set(available_models))
    if unknown:
        raise ValueError(
            f"Historical release has no model identifiers: {unknown}"
        )
    return list(dict.fromkeys(model_ids))


def run_doctor(args: argparse.Namespace) -> int:
    context = load_dpd_context(args.root, args.benchmark)
    credentials = configure_inspect_azure_environment()
    counts = {
        scope: int(len(load_dpd_cases(context, scope)))
        for scope in ("demo", "audit", "full")
    }
    inspect_version = _package_version("inspect-ai")
    openai_version = _package_version("openai")
    parity = build_parity_report(context)
    dependency_ready = (
        inspect_version == INSPECT_AI_VERSION
        and openai_version == OPENAI_VERSION
        and importlib.util.find_spec("openai") is not None
    )
    paid_run_ready = (
        dependency_ready
        and credentials["AZUREAI_OPENAI_API_KEY"]
        and credentials["AZUREAI_OPENAI_BASE_URL"]
    )
    report = {
        "status": "ready" if dependency_ready else "dependency_error",
        "inspect_ai": {
            "required_version": INSPECT_AI_VERSION,
            "installed_version": inspect_version,
        },
        "openai": {
            "required_version": OPENAI_VERSION,
            "installed_version": openai_version,
        },
        "benchmark_reference": args.benchmark,
        "benchmark_status": context.benchmark.status,
        "benchmark_fingerprint": context.fingerprint,
        "scope_case_counts": counts,
        "saved_output_replay_count": len(load_dpd_replay_records(context)),
        "scorer_parity": parity["status"],
        "credentials_present": credentials,
        "ready_for_paid_run": paid_run_ready,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        _write_json(args.output, report)
    if not dependency_ready or parity["status"] != "passed":
        return 1
    if args.require_credentials and not paid_run_ready:
        return 2
    return 0


def run_parity(args: argparse.Namespace) -> int:
    context = load_dpd_context(args.root, args.benchmark)
    report = build_parity_report(context)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        _write_json(args.output, report)
    return 0 if report["status"] == "passed" else 1


def _campaign_name(value: str | None) -> str:
    campaign = value or datetime.now(timezone.utc).strftime(
        "dpd-%Y%m%dT%H%M%SZ"
    )
    if not _CAMPAIGN_PATTERN.fullmatch(campaign):
        raise ValueError(
            "Campaign names may contain only letters, numbers, dots, "
            "underscores, and hyphens."
        )
    return campaign


def _run_plan(args: argparse.Namespace) -> tuple[Any, list[str], dict[str, Any]]:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if (
        args.requests_per_minute is not None
        and args.requests_per_minute <= 0
    ):
        raise ValueError("--requests-per-minute must be positive.")
    if args.max_connections <= 0:
        raise ValueError("--max-connections must be positive.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive.")
    if args.retry_on_error < 0:
        raise ValueError("--retry-on-error cannot be negative.")
    context = load_dpd_context(args.root, args.benchmark)
    models = _resolve_models(context, args)
    cases = load_dpd_cases(context, args.scope)
    case_count = min(len(cases), args.limit) if args.limit else len(cases)
    rpm = (
        args.requests_per_minute
        if args.requests_per_minute is not None
        else context.benchmark.runtime.max_requests_per_minute
    )
    plan_models = []
    for model_id in models:
        manifest = context.registry.resolve_model(model_id)
        plan_models.append(
            {
                "model_id": model_id,
                "inspect_route": inspect_model_route(
                    manifest.provider_route
                ),
                "request": manifest.request.model_dump(mode="json"),
            }
        )
    plan = {
        "schema_version": "1.0",
        "campaign": _campaign_name(args.campaign),
        "benchmark_reference": args.benchmark,
        "benchmark_fingerprint": context.fingerprint,
        "scope": args.scope,
        "case_count_per_model": case_count,
        "model_count": len(models),
        "total_requests_before_retries": case_count * len(models),
        "models": plan_models,
        "requests_per_minute_per_model": rpm,
        "minimum_request_window_minutes_per_model": (
            case_count / rpm if rpm else None
        ),
        "max_connections": args.max_connections,
        "responses_api": (
            context.benchmark.runtime.request_api == "responses"
        ),
        "execution": "sequential_models",
        "inspect_checkpointing": True,
    }
    return context, models, plan


def run_eval(args: argparse.Namespace) -> int:
    context, models, plan = _run_plan(args)
    output_root = Path(args.output_root).resolve() / plan["campaign"]
    log_dir = output_root / "logs"
    if log_dir.is_dir() and any(log_dir.rglob("*.eval")):
        raise RuntimeError(
            f"Campaign {plan['campaign']!r} already contains Inspect logs. "
            "Choose a new campaign name or use the retry command."
        )
    _write_json(output_root / "campaign_plan.json", plan)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.plan_only:
        return 0

    credentials = configure_inspect_azure_environment()
    if not credentials["AZUREAI_OPENAI_API_KEY"]:
        raise RuntimeError(
            "Missing AZURE_API_KEY or AZUREAI_OPENAI_API_KEY."
        )
    if not credentials["AZUREAI_OPENAI_BASE_URL"]:
        raise RuntimeError(
            "Missing AZURE_API_BASE or AZUREAI_OPENAI_BASE_URL."
        )
    try:
        from inspect_ai import eval as inspect_eval
    except ImportError as error:
        raise RuntimeError(
            "Inspect AI is not installed. Install requirements-inspect.txt."
        ) from error
    from integrations.inspect_ai.dpd_task import dpd
    from integrations.inspect_ai.reporting import write_campaign_report

    statuses: list[str] = []
    for model_id in models:
        manifest = context.registry.resolve_model(model_id)
        request = manifest.request
        generation: dict[str, Any] = {
            "max_connections": args.max_connections,
            "max_retries": request.max_retries,
            "timeout": args.timeout,
            "max_tokens": (
                request.max_completion_tokens
                or context.benchmark.runtime.max_completion_tokens
            ),
        }
        if request.temperature is not None:
            generation["temperature"] = request.temperature
        if request.reasoning_effort is not None:
            generation["reasoning_effort"] = request.reasoning_effort
        logs = inspect_eval(
            dpd(
                scope=args.scope,
                benchmark_reference=args.benchmark,
                root=str(context.root),
                requests_per_minute=args.requests_per_minute,
            ),
            model=inspect_model_route(manifest.provider_route),
            model_args={
                "responses_api": (
                    context.benchmark.runtime.request_api == "responses"
                )
            },
            metadata={
                "evalanche_model_id": model_id,
                "evalanche_campaign": plan["campaign"],
                "evalanche_benchmark_fingerprint": context.fingerprint[
                    "compatibility_sha256"
                ],
            },
            log_dir=str(log_dir),
            limit=args.limit,
            retry_on_error=args.retry_on_error,
            fail_on_error=False,
            checkpoint=True,
            display=args.display,
            **generation,
        )
        statuses.extend(str(log.status) for log in logs)
    summary = write_campaign_report(log_dir, output_root / "report")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if statuses and all(s == "success" for s in statuses) else 1


def run_retry(args: argparse.Namespace) -> int:
    credentials = configure_inspect_azure_environment()
    if not credentials["AZUREAI_OPENAI_API_KEY"]:
        raise RuntimeError(
            "Missing AZURE_API_KEY or AZUREAI_OPENAI_API_KEY."
        )
    if not credentials["AZUREAI_OPENAI_BASE_URL"]:
        raise RuntimeError(
            "Missing AZURE_API_BASE or AZUREAI_OPENAI_BASE_URL."
        )
    log_files = [Path(path).resolve() for path in args.log_files]
    missing = [str(path) for path in log_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Inspect log files not found: {missing}")
    log_directories = {path.parent for path in log_files}
    if len(log_directories) != 1:
        raise ValueError(
            "All retry logs must be in the same campaign log directory."
        )
    log_dir = next(iter(log_directories))
    try:
        from inspect_ai import eval_retry
    except ImportError as error:
        raise RuntimeError(
            "Inspect AI is not installed. Install requirements-inspect.txt."
        ) from error
    from integrations.inspect_ai.reporting import write_campaign_report

    logs = eval_retry(
        [str(path) for path in log_files],
        log_dir=str(log_dir),
        max_connections=args.max_connections,
        max_retries=args.max_retries,
        timeout=args.timeout,
        retry_on_error=args.retry_on_error,
        fail_on_error=False,
        checkpoint=True,
        display=args.display,
    )
    summary = write_campaign_report(log_dir, log_dir.parent / "report")
    result = {
        "status": (
            "passed"
            if all(str(log.status) == "success" for log in logs)
            else "failed"
        ),
        "retried_logs": [str(path) for path in log_files],
        "inspect_report": summary,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


def run_replay(args: argparse.Namespace) -> int:
    context = load_dpd_context(args.root, args.benchmark)
    parity = build_parity_report(context)
    if parity["status"] != "passed":
        print(json.dumps(parity, indent=2, ensure_ascii=False))
        return 1
    try:
        from inspect_ai import eval as inspect_eval
    except ImportError as error:
        raise RuntimeError(
            "Inspect AI is not installed. Install requirements-inspect.txt."
        ) from error
    from integrations.inspect_ai.dpd_task import dpd_replay
    from integrations.inspect_ai.reporting import write_campaign_report

    output_root = Path(args.output_dir).resolve()
    logs = inspect_eval(
        dpd_replay(
            benchmark_reference=args.benchmark,
            root=str(context.root),
        ),
        model="mockllm/model",
        log_dir=str(output_root / "logs"),
        display=args.display,
        fail_on_error=False,
    )
    summary = write_campaign_report(
        output_root / "logs", output_root / "report"
    )
    result = {
        "status": (
            "passed"
            if all(str(log.status) == "success" for log in logs)
            else "failed"
        ),
        "direct_parity": parity,
        "inspect_report": summary,
    }
    _write_json(output_root / "replay_result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


def run_import_results(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if args.max_connections <= 0:
        raise ValueError("--max-connections must be positive.")
    context = load_dpd_context(args.root, args.benchmark)
    historical = load_dpd_historical_import(context, args.source)
    models = _resolve_historical_models(historical.model_ids, args)
    case_count = (
        min(historical.case_count_per_model, args.limit)
        if args.limit
        else historical.case_count_per_model
    )
    campaign = _campaign_name(
        args.campaign
        or datetime.now(timezone.utc).strftime(
            "dpd-historical-%Y%m%dT%H%M%SZ"
        )
    )
    output_root = Path(args.output_root).resolve() / campaign
    log_dir = output_root / "logs"
    if log_dir.is_dir() and any(log_dir.rglob("*.eval")):
        raise RuntimeError(
            f"Campaign {campaign!r} already contains Inspect logs. "
            "Choose a new campaign name."
        )

    plan = {
        "schema_version": "1.0",
        "campaign": campaign,
        "benchmark_reference": args.benchmark,
        "benchmark_fingerprint": context.fingerprint,
        "execution_mode": "historical_import",
        "provider_requests": 0,
        "source": {
            "path": historical.source_path,
            "size_bytes": historical.source_size_bytes,
            "sha256": historical.source_sha256,
            "release_id": historical.release_id,
        },
        "case_count_per_model": case_count,
        "model_count": len(models),
        "saved_output_count": case_count * len(models),
        "models": [
            {
                "model_id": model_id,
                "historical_run_id": historical.published_runs[model_id][
                    "run_id"
                ],
                "published_case_scores_sha256": historical.published_runs[
                    model_id
                ]["case_scores_sha256"],
            }
            for model_id in models
        ],
        "inspect_model": "mockllm/model",
        "published_score_parity_required": True,
        "inspect_checkpointing": True,
    }
    _write_json(output_root / "import_plan.json", plan)
    print(json.dumps(json_safe(plan), indent=2, ensure_ascii=False))
    if args.plan_only:
        return 0

    try:
        from inspect_ai import eval as inspect_eval
    except ImportError as error:
        raise RuntimeError(
            "Inspect AI is not installed. Install requirements-inspect.txt."
        ) from error
    from integrations.inspect_ai.dpd_task import build_dpd_historical_task
    from integrations.inspect_ai.reporting import write_campaign_report

    statuses: list[str] = []
    for model_id in models:
        selected = historical.records.loc[
            historical.records["historical_model_id"].astype(str)
            == model_id
        ]
        if args.limit is not None:
            selected = selected.head(args.limit)
        task = build_dpd_historical_task(
            context=context,
            records=selected.to_dict(orient="records"),
            model_id=model_id,
            source_path=str(historical.source_path),
            source_sha256=historical.source_sha256,
            release_id=historical.release_id,
            run_info=historical.published_runs[model_id],
            benchmark_reference=args.benchmark,
        )
        logs = inspect_eval(
            task,
            model="mockllm/model",
            metadata={
                "evalanche_campaign": campaign,
                "evalanche_model_id": model_id,
                "evalanche_execution_mode": "historical_import",
                "evalanche_historical_release_id": historical.release_id,
                "evalanche_historical_run_id": historical.published_runs[
                    model_id
                ]["run_id"],
                "evalanche_historical_source_sha256": (
                    historical.source_sha256
                ),
            },
            log_dir=str(log_dir),
            max_connections=args.max_connections,
            fail_on_error=False,
            checkpoint=True,
            display=args.display,
        )
        statuses.extend(str(log.status) for log in logs)

    summary = write_campaign_report(log_dir, output_root / "report")
    expected_samples = case_count * len(models)
    passed = bool(statuses) and all(
        status == "success" for status in statuses
    )
    passed = passed and summary["status"] == "complete"
    passed = passed and summary["sample_count"] == expected_samples
    passed = passed and summary["parity"]["status"] == "passed"
    passed = (
        passed
        and summary["parity"]["checked_samples"] == expected_samples
    )
    passed = passed and summary["parity"]["disagreements"] == 0
    passed = passed and summary["execution_modes"] == [
        "historical_import"
    ]
    result = {
        "schema_version": "1.0",
        "status": "passed" if passed else "failed",
        "campaign": campaign,
        "provider_requests": 0,
        "imported_saved_outputs": summary["sample_count"],
        "expected_saved_outputs": expected_samples,
        "inspect_log_statuses": statuses,
        "inspect_report": summary,
    }
    _write_json(output_root / "import_result.json", result)
    print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))
    return 0 if passed else 1


def run_report(args: argparse.Namespace) -> int:
    from integrations.inspect_ai.reporting import write_campaign_report

    summary = write_campaign_report(args.log_dir, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m integrations.inspect_ai.cli",
        description=(
            "Run the frozen Evalanche DPD benchmark through optional "
            "Inspect AI infrastructure."
        ),
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=".")
    common.add_argument(
        "--benchmark", default=DEFAULT_BENCHMARK_REFERENCE
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", parents=[common])
    doctor.add_argument("--require-credentials", action="store_true")
    doctor.add_argument("--output")
    doctor.set_defaults(handler=run_doctor)

    parity = subparsers.add_parser("parity", parents=[common])
    parity.add_argument("--output")
    parity.set_defaults(handler=run_parity)

    replay = subparsers.add_parser("replay", parents=[common])
    replay.add_argument(
        "--output-dir", default="results/inspect_ai/dpd-replay"
    )
    replay.add_argument(
        "--display",
        choices=("full", "conversation", "rich", "plain", "log", "none"),
        default="plain",
    )
    replay.set_defaults(handler=run_replay)

    import_results = subparsers.add_parser(
        "import-results",
        parents=[common],
        help="Import saved full-census responses into Inspect AI.",
    )
    import_results.add_argument(
        "--source", default=DEFAULT_HISTORICAL_OUTPUTS_PATH
    )
    import_results.add_argument("--model", dest="models", action="append")
    import_results.add_argument("--all-models", action="store_true")
    import_results.add_argument("--campaign")
    import_results.add_argument(
        "--output-root", default="results/inspect_ai"
    )
    import_results.add_argument("--limit", type=int)
    import_results.add_argument("--max-connections", type=int, default=8)
    import_results.add_argument("--plan-only", action="store_true")
    import_results.add_argument(
        "--display",
        choices=("full", "conversation", "rich", "plain", "log", "none"),
        default="plain",
    )
    import_results.set_defaults(handler=run_import_results)

    run = subparsers.add_parser("run", parents=[common])
    run.add_argument("--scope", choices=("demo", "audit", "full"), default="demo")
    run.add_argument("--model", dest="models", action="append")
    run.add_argument("--all-models", action="store_true")
    run.add_argument("--campaign")
    run.add_argument("--output-root", default="results/inspect_ai")
    run.add_argument("--limit", type=int)
    run.add_argument("--requests-per-minute", type=float)
    run.add_argument("--max-connections", type=int, default=4)
    run.add_argument("--timeout", type=int, default=300)
    run.add_argument("--retry-on-error", type=int, default=1)
    run.add_argument("--plan-only", action="store_true")
    run.add_argument(
        "--display",
        choices=("full", "conversation", "rich", "plain", "log", "none"),
        default="plain",
    )
    run.set_defaults(handler=run_eval)

    retry = subparsers.add_parser("retry")
    retry.add_argument(
        "--log-file",
        dest="log_files",
        action="append",
        required=True,
    )
    retry.add_argument("--max-connections", type=int, default=4)
    retry.add_argument("--max-retries", type=int, default=3)
    retry.add_argument("--timeout", type=int, default=300)
    retry.add_argument("--retry-on-error", type=int, default=1)
    retry.add_argument(
        "--display",
        choices=("full", "conversation", "rich", "plain", "log", "none"),
        default="plain",
    )
    retry.set_defaults(handler=run_retry)

    report = subparsers.add_parser("report")
    report.add_argument("--log-dir", required=True)
    report.add_argument("--output-dir", required=True)
    report.set_defaults(handler=run_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
