from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from evalanche.registry import BenchmarkManifest, LoadedRegistry
from evalanche.result_bundle import load_active_bundles

LEADERBOARD_SCHEMA_VERSION = "1.0"


def _format_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _bundle_rows(
    bundles: list[dict[str, Any]],
    benchmark: BenchmarkManifest,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        metadata = bundle["metadata"]
        summary = metadata["summary"]
        row: dict[str, Any] = {
            "model_id": metadata["model"]["model_id"],
            "model": metadata["model"]["display_name"],
            "run_id": metadata["run_id"],
            "cases": summary["cases"],
            "passed_cases": summary["passed_cases"],
            "failed_cases": summary["failed_cases"],
            "pass_rate": summary["pass_rate"],
            "average_field_score": summary["average_score"],
            "generation_failure_rate": summary[
                "generation_failure_rate"
            ],
            "total_cost_usd": summary["generation_total_cost_usd"],
            "average_cost_usd": summary[
                "generation_average_cost_usd"
            ],
            "median_latency_seconds": summary[
                "generation_median_seconds"
            ],
            "p95_latency_seconds": summary["generation_p95_seconds"],
        }
        slices = summary.get("slices", {})
        language_column = benchmark.language_column
        if language_column and language_column in slices:
            for language, values in slices[language_column].items():
                row[f"{language}_pass_rate"] = values["pass_rate"]
                row[f"{language}_cases"] = values["cases"]
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "rank",
                "model_id",
                "model",
                "run_id",
                "cases",
                "passed_cases",
                "failed_cases",
                "pass_rate",
                "average_field_score",
                "generation_failure_rate",
                "total_cost_usd",
                "average_cost_usd",
                "median_latency_seconds",
                "p95_latency_seconds",
            ]
        )

    leaderboard = pd.DataFrame(rows)
    leaderboard["rank"] = (
        leaderboard["pass_rate"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    ordered = [
        "rank",
        "model_id",
        "model",
        "run_id",
        "cases",
        "passed_cases",
        "failed_cases",
        "pass_rate",
        "average_field_score",
        "generation_failure_rate",
        "total_cost_usd",
        "average_cost_usd",
        "median_latency_seconds",
        "p95_latency_seconds",
    ]
    extra = [column for column in leaderboard if column not in ordered]
    return leaderboard[ordered + sorted(extra)].sort_values(
        ["rank", "model_id"],
        kind="stable",
    )


def _pairwise(
    bundles: list[dict[str, Any]],
    benchmark: BenchmarkManifest,
) -> pd.DataFrame:
    score_frames = [
        pd.read_csv(bundle["score_path"])[
            ["case_id", benchmark.group_key, "model_name", "final_passed"]
        ]
        for bundle in bundles
    ]
    columns = [
        "model_a",
        "model_b",
        "cases",
        "a_only_passed",
        "b_only_passed",
        "both_passed",
        "both_failed",
        "paired_net_wins_a",
    ]
    if len(score_frames) < 2:
        return pd.DataFrame(columns=columns)

    scores = pd.concat(score_frames, ignore_index=True)
    pivot = scores.pivot(
        index="case_id",
        columns="model_name",
        values="final_passed",
    )
    models = sorted(str(model) for model in pivot.columns)
    records: list[dict[str, Any]] = []
    for index, model_a in enumerate(models):
        for model_b in models[index + 1 :]:
            paired = pivot[[model_a, model_b]].dropna().astype(bool)
            a_only = int((paired[model_a] & ~paired[model_b]).sum())
            b_only = int((~paired[model_a] & paired[model_b]).sum())
            both_passed = int(
                (paired[model_a] & paired[model_b]).sum()
            )
            both_failed = int(
                (~paired[model_a] & ~paired[model_b]).sum()
            )
            records.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "cases": int(len(paired)),
                    "a_only_passed": a_only,
                    "b_only_passed": b_only,
                    "both_passed": both_passed,
                    "both_failed": both_failed,
                    "paired_net_wins_a": a_only - b_only,
                }
            )
    return pd.DataFrame(records, columns=columns)


def _slice_records(
    bundles: list[dict[str, Any]],
    benchmark: BenchmarkManifest,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for bundle in bundles:
        metadata = bundle["metadata"]
        for slice_name, values in metadata["summary"].get(
            "slices",
            {},
        ).items():
            for slice_value, metrics in values.items():
                records.append(
                    {
                        "model_id": metadata["model"]["model_id"],
                        "slice": slice_name,
                        "value": slice_value,
                        **metrics,
                    }
                )
    return sorted(
        records,
        key=lambda record: (
            record["slice"],
            record["value"],
            record["model_id"],
        ),
    )


def _markdown(
    benchmark: BenchmarkManifest,
    leaderboard: pd.DataFrame,
) -> str:
    lines = [
        f"# {benchmark.title}",
        "",
        f"Benchmark: `{benchmark.benchmark_id}@{benchmark.version}`",
        "",
    ]
    if leaderboard.empty:
        lines.extend(
            [
                "No compatible result bundles are registered yet.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "| Rank | Model | Passed | Pass rate | Field score | "
            "Generation failures | Cost (USD) | p95 latency (s) |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in leaderboard.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["rank"]),
                    str(row["model"]),
                    f"{int(row['passed_cases'])}/{int(row['cases'])}",
                    _format_percent(row["pass_rate"]),
                    _format_percent(row["average_field_score"]),
                    _format_percent(row["generation_failure_rate"]),
                    _format_number(row["total_cost_usd"], 4),
                    _format_number(row["p95_latency_seconds"], 3),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Ranks use strict pass rate. Cost and latency are displayed "
            "separately and are not collapsed into a recommendation score.",
            "",
        ]
    )
    return "\n".join(lines)


def _table_html(leaderboard: pd.DataFrame) -> str:
    if leaderboard.empty:
        return (
            '<div class="empty">No compatible result bundles are '
            "registered yet.</div>"
        )

    columns = [
        ("rank", "Rank"),
        ("model", "Model"),
        ("passed_cases", "Passed"),
        ("pass_rate", "Strict pass"),
        ("average_field_score", "Field score"),
        ("en_pass_rate", "English"),
        ("fr_pass_rate", "French"),
        ("generation_failure_rate", "Gen failures"),
        ("total_cost_usd", "Cost USD"),
        ("p95_latency_seconds", "p95 seconds"),
    ]
    columns = [
        item for item in columns if item[0] in leaderboard.columns
    ]
    header = "".join(
        f'<th data-key="{html.escape(key)}">{html.escape(label)}</th>'
        for key, label in columns
    )
    rows = []
    percent_columns = {
        "pass_rate",
        "average_field_score",
        "en_pass_rate",
        "fr_pass_rate",
        "generation_failure_rate",
    }
    for _, row in leaderboard.iterrows():
        cells = []
        for key, _ in columns:
            value = row[key]
            if key == "passed_cases":
                shown = f"{int(value):,}/{int(row['cases']):,}"
            elif key in percent_columns:
                shown = _format_percent(value)
            elif key in {"total_cost_usd", "p95_latency_seconds"}:
                shown = _format_number(value, 3)
            else:
                shown = "" if pd.isna(value) else str(value)
            sort_value = "" if pd.isna(value) else value
            cells.append(
                f'<td data-value="{html.escape(str(sort_value))}">'
                f"{html.escape(shown)}</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<table id="leaderboard"><thead><tr>'
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _html(
    benchmark: BenchmarkManifest,
    leaderboard: pd.DataFrame,
    slices: list[dict[str, Any]],
    pairwise: pd.DataFrame,
) -> str:
    payload = json.dumps(
        {
            "leaderboard": leaderboard.where(
                pd.notna(leaderboard),
                None,
            ).to_dict(orient="records"),
            "slices": slices,
            "pairwise": pairwise.to_dict(orient="records"),
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(benchmark.title)}</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f4f7fb;
      --card: #fff;
      --ink: #12233f;
      --muted: #5b6b82;
      --line: #dce4ef;
      --accent: #1768ac;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0e1726;
        --card: #172338;
        --ink: #eef5ff;
        --muted: #aebcd0;
        --line: #33445e;
        --accent: #77bdf2;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; background: var(--bg); color: var(--ink);
      font: 15px/1.5 system-ui, sans-serif;
    }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 48px 24px; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(30px, 5vw, 50px); }}
    .kicker {{ color: var(--accent); font-weight: 700; }}
    .intro {{ color: var(--muted); max-width: 850px; }}
    .card {{
      background: var(--card); border: 1px solid var(--line);
      border-radius: 16px; margin-top: 28px; padding: 20px;
      box-shadow: 0 12px 30px rgb(20 50 80 / 7%);
    }}
    .scroll {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 850px; }}
    th, td {{
      padding: 12px 10px; border-bottom: 1px solid var(--line);
      text-align: right; white-space: nowrap;
    }}
    th {{ cursor: pointer; color: var(--muted); font-size: 12px; }}
    th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    tbody tr:hover {{ background: color-mix(in srgb, var(--accent) 7%, transparent); }}
    .empty {{ color: var(--muted); padding: 24px 0; }}
    footer {{ color: var(--muted); margin-top: 24px; }}
    code {{ font-size: 0.95em; }}
  </style>
</head>
<body>
<main>
  <div class="kicker">Evalanche benchmark leaderboard</div>
  <h1>{html.escape(benchmark.title)}</h1>
  <p class="intro">{html.escape(benchmark.description)}</p>
  <p><code>{html.escape(benchmark.benchmark_id)}@{html.escape(benchmark.version)}</code></p>
  <section class="card">
    <h2>Results</h2>
    <div class="scroll">{_table_html(leaderboard)}</div>
  </section>
  <section class="card">
    <h2>How to read this</h2>
    <p class="intro">
      Rank uses strict case pass rate. Field score, language slices,
      reliability, cost, and latency remain separate so the table does
      not imply a universal model recommendation.
    </p>
  </section>
  <footer>Generated from compatible, versioned result bundles.</footer>
</main>
<script id="leaderboard-data" type="application/json">{payload}</script>
<script>
  const table = document.querySelector("#leaderboard");
  if (table) {{
    table.querySelectorAll("th").forEach((header, index) => {{
      let ascending = true;
      header.addEventListener("click", () => {{
        const body = table.tBodies[0];
        const rows = Array.from(body.rows);
        rows.sort((a, b) => {{
          const av = a.cells[index].dataset.value || "";
          const bv = b.cells[index].dataset.value || "";
          const an = Number(av), bn = Number(bv);
          const numeric = av !== "" && bv !== "" &&
            Number.isFinite(an) && Number.isFinite(bn);
          const result = numeric ? an - bn :
            av.localeCompare(bv, undefined, {{numeric: true}});
          return ascending ? result : -result;
        }});
        rows.forEach(row => body.appendChild(row));
        ascending = !ascending;
      }});
    }});
  }}
</script>
</body>
</html>
"""


def build_leaderboard(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    reports_root: str | Path = "reports/benchmarks",
) -> dict[str, Any]:
    bundles = load_active_bundles(
        registry=registry,
        benchmark=benchmark,
        reports_root=reports_root,
    )
    leaderboard = _bundle_rows(bundles, benchmark)
    pairwise = _pairwise(bundles, benchmark)
    slices = _slice_records(bundles, benchmark)
    output_dir = (
        registry.root
        / reports_root
        / benchmark.benchmark_id
        / benchmark.version
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "leaderboard.csv"
    json_path = output_dir / "leaderboard.json"
    markdown_path = output_dir / "leaderboard.md"
    html_path = output_dir / "leaderboard.html"
    pairwise_path = output_dir / "pairwise.csv"

    leaderboard.to_csv(csv_path, index=False, lineterminator="\n")
    pairwise.to_csv(pairwise_path, index=False, lineterminator="\n")
    document = {
        "schema_version": LEADERBOARD_SCHEMA_VERSION,
        "benchmark": {
            "benchmark_id": benchmark.benchmark_id,
            "version": benchmark.version,
            "title": benchmark.title,
            "status": benchmark.status,
        },
        "models": leaderboard.where(pd.notna(leaderboard), None).to_dict(
            orient="records"
        ),
        "slices": slices,
        "pairwise": pairwise.to_dict(orient="records"),
        "ranking_policy": (
            "Strict case pass rate only. Cost, latency, reliability, and "
            "slice metrics are descriptive."
        ),
    }
    json_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _markdown(benchmark, leaderboard),
        encoding="utf-8",
    )
    html_path.write_text(
        _html(benchmark, leaderboard, slices, pairwise),
        encoding="utf-8",
    )
    return {
        "benchmark_id": benchmark.benchmark_id,
        "benchmark_version": benchmark.version,
        "models": int(len(leaderboard)),
        "output_dir": output_dir,
        "leaderboard_path": csv_path,
        "json_path": json_path,
        "markdown_path": markdown_path,
        "html_path": html_path,
        "pairwise_path": pairwise_path,
    }


def build_benchmark_index(
    *,
    registry: LoadedRegistry,
    reports_root: str | Path = "reports/benchmarks",
) -> Path:
    root = registry.root / reports_root
    root.mkdir(parents=True, exist_ok=True)
    cards = []
    for key, benchmark in sorted(registry.benchmarks.items()):
        leaderboard_path = (
            root
            / benchmark.benchmark_id
            / benchmark.version
            / "leaderboard.json"
        )
        model_count = 0
        if leaderboard_path.is_file():
            document = json.loads(
                leaderboard_path.read_text(encoding="utf-8")
            )
            model_count = len(document.get("models", []))
        href = (
            f"{benchmark.benchmark_id}/{benchmark.version}/"
            "leaderboard.html"
        )
        cards.append(
            f'<a class="card" href="{html.escape(href)}">'
            f"<strong>{html.escape(benchmark.title)}</strong>"
            f"<span>{html.escape(key)}</span>"
            f"<span>{model_count} registered model"
            f"{'' if model_count == 1 else 's'}</span></a>"
        )
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Evalanche benchmark leaderboards</title>
  <style>
    body {{
      margin: 0; background: #f4f7fb; color: #12233f;
      font: 16px/1.5 system-ui, sans-serif;
    }}
    main {{ max-width: 1000px; margin: 0 auto; padding: 56px 24px; }}
    h1 {{ font-size: clamp(34px, 6vw, 58px); margin-bottom: 8px; }}
    p {{ color: #5b6b82; max-width: 700px; }}
    .grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px; margin-top: 36px;
    }}
    .card {{
      display: flex; flex-direction: column; gap: 8px; padding: 24px;
      color: inherit; text-decoration: none; background: white;
      border: 1px solid #dce4ef; border-radius: 16px;
      box-shadow: 0 12px 30px rgb(20 50 80 / 7%);
    }}
    .card:hover {{ border-color: #1768ac; transform: translateY(-2px); }}
    .card span {{ color: #5b6b82; font-size: 14px; }}
  </style>
</head>
<body>
<main>
  <h1>Benchmark leaderboards</h1>
  <p>
    Versioned, task-specific comparisons built from compatible result
    bundles. Each benchmark keeps quality, cost, latency, and reliability
    visible without declaring a universal winner.
  </p>
  <div class="grid">{''.join(cards)}</div>
</main>
</body>
</html>
"""
    path = root / "index.html"
    path.write_text(content, encoding="utf-8")
    return path
