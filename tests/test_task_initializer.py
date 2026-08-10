from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from evalanche.benchmark_campaign import build_tier_campaign_preflight
from evalanche.benchmark_runner import build_run_plan
from evalanche.config import load_evaluation_config
from evalanche.judges.protocol import (
    load_judge_protocol,
    resolve_judge_evidence,
)
from evalanche.registry import load_registry, validate_registry
from evalanche.task_initializer import (
    TaskCriterion,
    TaskModelRoute,
    initialize_task_bundle,
    parse_criterion,
    parse_model_route,
    task_execution_blocker,
    validate_task_bundle,
)


CREATED_ON = date(2026, 8, 10)
CURRENT_EXAMPLE_MODEL_IDS = {
    "gpt_5_4_mini",
    "gpt_5_6_luna",
    "gpt_5_6_sol",
    "gpt_5_6_terra",
}


def _family_options(family: str) -> dict[str, object]:
    if family == "structured-extraction":
        return {
            "required_output_fields": ["name", "status"],
            "sample_expected_output": '{"name":"Example","status":"ok"}',
        }
    if family == "open-ended-response":
        return {
            "criteria": [
                TaskCriterion(
                    name="grounding",
                    weight=1.0,
                    description="Every material claim is supported.",
                )
            ],
            "measured_construct": "Reference-grounded response quality.",
            "intended_use": "Exploratory local comparison only.",
            "sample_expected_output": "A reviewed reference response.",
        }
    if family == "classification":
        return {"sample_expected_output": "routine"}
    return {"sample_expected_output": "authoritative answer"}


def _initialize(
    tmp_path: Path,
    *,
    task_id: str,
    family: str,
    with_model: bool = True,
) -> dict[str, object]:
    models = (
        [
            TaskModelRoute(
                model_id="fictional_model_alpha",
                provider_route="test/fictional-alpha",
            )
        ]
        if with_model
        else []
    )
    return initialize_task_bundle(
        root_path=tmp_path,
        task_id=task_id,
        description="Evaluate a fictional task with representative cases.",
        family=family,
        languages=["en", "fr"],
        unacceptable_errors=["Returning a materially unsupported result."],
        models=models,
        sample_input="A fictional development input.",
        created_on=CREATED_ON,
        **_family_options(family),
    )


@pytest.mark.parametrize(
    ("family", "evaluation_type"),
    [
        ("classification", "exact"),
        ("structured-extraction", "json"),
        ("factual-response", "exact"),
        ("open-ended-response", "judge"),
    ],
)
def test_each_task_family_generates_a_schema_valid_isolated_bundle(
    tmp_path: Path,
    family: str,
    evaluation_type: str,
) -> None:
    result = _initialize(
        tmp_path,
        task_id=f"arbitrary_{family.replace('-', '_')}_task",
        family=family,
    )
    root = Path(result["task_root"])
    validation = result["validation"]

    assert validation["schema_valid"] is True
    assert validation["ready_for_plan"] is False
    assert validation["provider_calls"] == 0
    assert result["provider_calls"] == 0
    assert (root / "README.md").is_file()
    assert (root / "task.yaml").is_file()
    assert (root / "data/cases.csv").is_file()
    assert (root / "guidance/expected_answers.md").is_file()

    registry = load_registry(root)
    registry_validation = validate_registry(registry)
    assert registry_validation["valid"] is True
    benchmark = next(iter(registry.benchmarks.values()))
    assert benchmark.scoring.evaluation_type == evaluation_type
    assert set(registry.models) == {"fictional_model_alpha"}

    bundle_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.rglob("*")
        if path.is_file()
    )
    assert not (CURRENT_EXAMPLE_MODEL_IDS & set(bundle_text.split()))
    for model_id in CURRENT_EXAMPLE_MODEL_IDS:
        assert model_id not in bundle_text

    if evaluation_type == "judge":
        protocol = load_judge_protocol(
            root / "judge_validation/protocol.yaml"
        )
        assert protocol.target_level == "exploratory"
        assert protocol.synthetic_fixture is False
        assert (
            root / "judge_validation/human_annotations.csv"
        ).read_text(encoding="utf-8").count("\n") == 1


def test_no_models_creates_templates_without_copying_registry_models(
    tmp_path: Path,
) -> None:
    result = _initialize(
        tmp_path,
        task_id="fictional_task_without_routes",
        family="classification",
        with_model=False,
    )
    root = Path(result["task_root"])
    validation = result["validation"]

    assert validation["schema_valid"] is True
    assert any(
        "explicitly available model" in issue
        for issue in validation["unresolved_values"]
    )
    assert not list((root / "configs/models").glob("*.yaml"))
    assert not (root / "configs/access_sets").exists()
    assert (root / "templates/model.yaml").is_file()
    assert (root / "templates/access_set.yaml").is_file()


def test_generated_open_ended_protocol_matches_runtime_contract(
    tmp_path: Path,
) -> None:
    result = _initialize(
        tmp_path,
        task_id="fictional_open_quality",
        family="open-ended-response",
    )
    root = Path(result["task_root"])
    registry = load_registry(root)
    benchmark = registry.resolve_benchmark("fictional_open_quality@0.1.0")
    model = registry.resolve_model("fictional_model_alpha")

    plan = build_run_plan(
        registry=registry,
        benchmark=benchmark,
        model=model,
    )
    config = load_evaluation_config(
        root / plan["paths"]["evaluation_config"]
    )
    evidence = resolve_judge_evidence(config)

    assert evidence.level == "exploratory"
    assert evidence.protocol_id == "fictional_open_quality_judge"
    assert evidence.validation_report_path is None
    assert str(root) in str(config.run.input_path)


def test_task_becomes_plan_ready_only_after_cases_and_reviews(
    tmp_path: Path,
) -> None:
    result = _initialize(
        tmp_path,
        task_id="fictional_ready_task",
        family="classification",
    )
    root = Path(result["task_root"])
    cases_path = root / "data/cases.csv"
    cases = pd.read_csv(cases_path)
    cases.loc[1, "input"] = "A fictional held-out input."
    cases.loc[1, "expected_output"] = "escalate"
    cases.to_csv(cases_path, index=False, lineterminator="\n")

    profile_path = root / "task.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    for field in (
        "task_definition",
        "cases_and_expected_outputs",
        "prompt_and_scoring",
        "privacy_and_artifacts",
    ):
        profile["review"][field] = True
    profile_path.write_text(
        yaml.safe_dump(profile, sort_keys=False),
        encoding="utf-8",
    )

    validation = validate_task_bundle(root)

    assert validation["schema_valid"] is True
    assert validation["ready_for_plan"] is True
    assert validation["ready_for_cost_preflight"] is False
    assert validation["cost_blockers"]


def test_verified_pricing_makes_deterministic_bundle_cost_ready(
    tmp_path: Path,
) -> None:
    result = _initialize(
        tmp_path,
        task_id="fictional_priced_task",
        family="classification",
    )
    root = Path(result["task_root"])
    cases_path = root / "data/cases.csv"
    cases = pd.read_csv(cases_path)
    cases.loc[1, "input"] = "A fictional held-out input."
    cases.loc[1, "expected_output"] = "escalate"
    cases.to_csv(cases_path, index=False, lineterminator="\n")

    profile_path = root / "task.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    for field in (
        "task_definition",
        "cases_and_expected_outputs",
        "prompt_and_scoring",
        "privacy_and_artifacts",
    ):
        profile["review"][field] = True
    profile_path.write_text(
        yaml.safe_dump(profile, sort_keys=False),
        encoding="utf-8",
    )

    pricing_path = root / "configs/pricing.yaml"
    pricing_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "catalog_version": "fictional-1",
                "endpoints": [
                    {
                        "pricing_id": "fictional_price",
                        "model": "test/fictional-alpha",
                        "currency": "USD",
                        "input_per_million_tokens": 1.0,
                        "output_per_million_tokens": 2.0,
                        "effective_date": "2026-08-10",
                        "source": "Fictional unit-test price.",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    model_path = root / "configs/models/fictional_model_alpha.yaml"
    model = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    model["pricing_catalog_path"] = "configs/pricing.yaml"
    model["pricing_id"] = "fictional_price"
    model_path.write_text(
        yaml.safe_dump(model, sort_keys=False),
        encoding="utf-8",
    )

    validation = validate_task_bundle(root)

    assert validation["ready_for_cost_preflight"] is True
    assert task_execution_blocker(root) is None


def test_custom_root_tier_preflight_uses_bundle_paths_without_calls(
    tmp_path: Path,
) -> None:
    result = _initialize(
        tmp_path,
        task_id="fictional_preflight_task",
        family="classification",
    )
    root = Path(result["task_root"])
    cases_path = root / "data/cases.csv"
    cases = pd.read_csv(cases_path)
    cases.loc[1, "input"] = "A fictional held-out input."
    cases.loc[1, "expected_output"] = "escalate"
    cases.to_csv(cases_path, index=False, lineterminator="\n")

    profile_path = root / "task.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    for field in (
        "task_definition",
        "cases_and_expected_outputs",
        "prompt_and_scoring",
        "privacy_and_artifacts",
    ):
        profile["review"][field] = True
    profile_path.write_text(
        yaml.safe_dump(profile, sort_keys=False),
        encoding="utf-8",
    )

    pricing_path = root / "configs/pricing.yaml"
    pricing_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "catalog_version": "fictional-1",
                "endpoints": [
                    {
                        "pricing_id": "fictional_price",
                        "model": "test/fictional-alpha",
                        "currency": "USD",
                        "input_per_million_tokens": 1.0,
                        "output_per_million_tokens": 2.0,
                        "effective_date": "2026-08-10",
                        "source": "Fictional unit-test price.",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    model_path = root / "configs/models/fictional_model_alpha.yaml"
    model = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    model["pricing_catalog_path"] = "configs/pricing.yaml"
    model["pricing_id"] = "fictional_price"
    model_path.write_text(
        yaml.safe_dump(model, sort_keys=False),
        encoding="utf-8",
    )

    registry = load_registry(root)
    preflight = build_tier_campaign_preflight(
        registry=registry,
        benchmark=registry.resolve_benchmark(
            "fictional_preflight_task@0.1.0"
        ),
        tier_name="smoke",
        model_ids=["fictional_model_alpha"],
    )
    generation_config = yaml.safe_load(
        (
            root
            / preflight["model_plans"]["fictional_model_alpha"]["plan"]
            ["paths"]["generation_config"]
        ).read_text(encoding="utf-8")
    )

    assert preflight["pending_requests"] == 1
    assert preflight["projected_budgeted_total_usd"] > 0
    assert str(root) in generation_config["run"]["input_path"]
    assert str(root) in generation_config["generation"][
        "cost_preflight_sample_path"
    ]
    assert (
        task_execution_blocker(root, require_paid_execution=False) is None
    )


def test_initializer_refuses_overwrite_and_preserves_replaced_bundle(
    tmp_path: Path,
) -> None:
    first = _initialize(
        tmp_path,
        task_id="fictional_replace_task",
        family="classification",
    )
    root = Path(first["task_root"])
    (root / "user_marker.txt").write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        _initialize(
            tmp_path,
            task_id="fictional_replace_task",
            family="classification",
        )

    replaced = initialize_task_bundle(
        root_path=tmp_path,
        task_id="fictional_replace_task",
        description="Replacement fictional task.",
        family="classification",
        unacceptable_errors=["A material fictional error."],
        models=[
            TaskModelRoute(
                model_id="fictional_model_alpha",
                provider_route="test/fictional-alpha",
            )
        ],
        replace=True,
        created_on=CREATED_ON,
    )

    backup = Path(replaced["backup"])
    assert backup.is_dir()
    assert (backup / "user_marker.txt").read_text(encoding="utf-8") == (
        "preserve me"
    )
    assert not (Path(replaced["task_root"]) / "user_marker.txt").exists()


def test_sensitive_task_rejects_command_line_sample_values(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Do not pass sample values"):
        initialize_task_bundle(
            root_path=tmp_path,
            task_id="fictional_sensitive_task",
            description="A task completed only in an approved environment.",
            family="classification",
            contains_sensitive_data=True,
            sample_input="Potentially sensitive input.",
        )


def test_sensitive_task_cannot_start_with_shareable_artifacts(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must use local_only"):
        initialize_task_bundle(
            root_path=tmp_path,
            task_id="fictional_sensitive_share_task",
            description="A task requiring separate data review.",
            family="classification",
            contains_sensitive_data=True,
            artifact_policy="shareable",
        )


def test_output_directory_cannot_escape_repository(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must stay inside"):
        initialize_task_bundle(
            root_path=tmp_path,
            output_dir=tmp_path.parent / "outside",
            task_id="fictional_escape_task",
            description="A fictional path-safety test.",
            family="classification",
        )


def test_model_and_criterion_argument_parsers_are_provider_neutral() -> None:
    model = parse_model_route("made_up_model=self_hosted/made-up")
    criterion = parse_criterion(
        "grounding=0.75=Every claim must be supported by the source."
    )

    assert model.model_id == "made_up_model"
    assert model.provider_route == "self_hosted/made-up"
    assert criterion.name == "grounding"
    assert criterion.weight == 0.75


def test_generated_commands_are_single_line_and_include_named_models(
    tmp_path: Path,
) -> None:
    result = initialize_task_bundle(
        root_path=tmp_path,
        task_id="fictional_two_model_task",
        description="Compare two fictional routes.",
        family="classification",
        unacceptable_errors=["Returning the wrong fictional label."],
        models=[
            TaskModelRoute(
                model_id="fictional_alpha",
                provider_route="test/alpha",
            ),
            TaskModelRoute(
                model_id="fictional_beta",
                provider_route="test/beta",
            ),
        ],
        created_on=CREATED_ON,
    )
    readme = (Path(result["task_root"]) / "README.md").read_text(
        encoding="utf-8"
    )

    assert (
        "--model fictional_alpha --model fictional_beta --plan-only"
        in readme
    )
    assert "\\\n" not in readme
    assert "+  --model" not in readme


def test_open_ended_bundle_is_never_marked_paid_execution_ready(
    tmp_path: Path,
) -> None:
    result = _initialize(
        tmp_path,
        task_id="fictional_open_execution",
        family="open-ended-response",
    )
    root = Path(result["task_root"])

    assert result["validation"]["paid_execution_supported"] is False
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "No paid open-ended execution command is generated" in readme
    assert "validate-judge --root" in readme
