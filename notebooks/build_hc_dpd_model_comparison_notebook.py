from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip() + "\n",
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


cells = [
    markdown(
        """
        <div style="padding:34px 38px;border-radius:18px;background:linear-gradient(135deg,#102a43 0%,#176b87 100%);color:white;">
          <div style="font-size:14px;letter-spacing:.12em;text-transform:uppercase;opacity:.82;">Evalanche demonstration</div>
          <h1 style="font-size:38px;line-height:1.1;margin:10px 0 12px;">Which GPT model should we use?</h1>
          <p style="font-size:20px;line-height:1.45;margin:0;max-width:900px;">
            The same Health Canada data extraction task, measured across quality, speed, and cost.
          </p>
        </div>

        **The question:** When several models can do the same job, how do we choose the one that gives us the right balance of accuracy and cost?
        """
    ),
    code(
        """
        from __future__ import annotations

        import ast
        import html
        import io
        import json
        import math
        import os
        import sys
        from contextlib import redirect_stdout
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import HTML, Markdown, display

        ROOT = next(
            (
                path
                for path in (Path.cwd(), *Path.cwd().parents)
                if (path / "evalanche").is_dir()
                and (path / "configs").is_dir()
            ),
            Path.cwd(),
        )
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        AUTO_REFRESH_LOCAL_RESULTS = True
        CENSUS_CASE_COUNT = 14_034
        MODEL_ORDER = [
            "gpt_5_4_mini",
            "gpt_5_6_luna",
            "gpt_5_6_terra",
            "gpt_5_6_sol",
        ]
        MODEL_INFO = {
            "gpt_5_4_mini": {
                "label": "GPT-5.4 mini",
                "role": "Full-census baseline",
                "color": "#2f80ed",
            },
            "gpt_5_6_luna": {
                "label": "GPT-5.6 Luna",
                "role": "Lowest list-price 5.6 tier",
                "color": "#27ae60",
            },
            "gpt_5_6_terra": {
                "label": "GPT-5.6 Terra",
                "role": "Middle list-price 5.6 tier",
                "color": "#f2994a",
            },
            "gpt_5_6_sol": {
                "label": "GPT-5.6 Sol",
                "role": "Highest list-price 5.6 tier",
                "color": "#eb5757",
            },
        }
        STRATUM_LABELS = {
            "single_ingredient": "Single ingredient",
            "multi_ingredient": "Multiple ingredients",
            "multi_variant": "Multiple variants",
            "multi_ingredient_multi_variant": "Multiple ingredients + variants",
        }
        FULL_MINI_RESULT = {
            "cases": 14_034,
            "passed": 10_827,
            "pass_rate": 10_827 / 14_034,
            "average_score": 0.923,
            "ci": "76.4% to 77.8%",
            "generation_failures": 0,
        }

        plt.rcParams.update(
            {
                "figure.figsize": (11, 5.5),
                "axes.facecolor": "#f7fafc",
                "figure.facecolor": "white",
                "axes.edgecolor": "#d9e2ec",
                "axes.labelcolor": "#334e68",
                "xtick.color": "#486581",
                "ytick.color": "#486581",
                "text.color": "#243b53",
                "font.size": 11,
                "axes.titleweight": "bold",
                "axes.titlesize": 13,
            }
        )

        display(
            HTML(
                '''
                <style>
                  .jp-Notebook { max-width: 1220px; margin: auto; }
                  .demo-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin:14px 0 20px; }
                  .demo-card { border:1px solid #d9e2ec; border-radius:14px; padding:18px; background:#fff; box-shadow:0 3px 12px rgba(16,42,67,.06); }
                  .demo-kicker { color:#627d98; text-transform:uppercase; letter-spacing:.08em; font-size:11px; font-weight:700; }
                  .demo-value { color:#102a43; font-size:30px; line-height:1.15; font-weight:800; margin:5px 0; }
                  .demo-note { color:#486581; font-size:13px; line-height:1.4; }
                  .task-grid { display:grid; grid-template-columns:1fr 58px 1fr; gap:12px; align-items:stretch; margin:16px 0; }
                  .task-panel { border-radius:14px; padding:18px; border:1px solid #d9e2ec; background:#f8fbfd; overflow:auto; }
                  .task-arrow { display:flex; align-items:center; justify-content:center; font-size:34px; color:#176b87; font-weight:800; }
                  .status-ready { border-left:6px solid #27ae60; background:#edf9f1; padding:14px 18px; border-radius:10px; }
                  .status-pending { border-left:6px solid #f2c94c; background:#fff9e6; padding:14px 18px; border-radius:10px; }
                  .small-muted { color:#627d98; font-size:12px; }
                  table.demo-table { border-collapse:collapse; width:100%; margin:12px 0 20px; font-size:13px; }
                  table.demo-table th { background:#eaf2f8; color:#243b53; padding:10px 12px; text-align:left; border-bottom:2px solid #bcccdc; }
                  table.demo-table td { padding:9px 12px; border-bottom:1px solid #d9e2ec; text-align:left; }
                  table.demo-table tr:nth-child(even) td { background:#f8fbfd; }
                  @media (max-width:760px) {
                    .task-grid { grid-template-columns:1fr; }
                    .task-arrow { transform:rotate(90deg); min-height:42px; }
                  }
                </style>
                '''
            )
        )

        def show_table(frame: pd.DataFrame) -> None:
            display(
                HTML(
                    frame.to_html(
                        index=False,
                        escape=True,
                        border=0,
                        classes="demo-table",
                    )
                )
            )
        """
    ),
    markdown(
        """
        ## 1. Start with the basics, what is DPD?

        The **Drug Product Database (DPD)** is Health Canada's public database of drugs authorized for sale in Canada. It includes identifiers and structured facts such as brand name, ingredients, strength, dosage form, route, schedule, status, and company.

        Health Canada updates the public database nightly. For a reproducible evaluation, this benchmark freezes one official snapshot instead of allowing the source to change during a run.

        - [Search the live Drug Product Database](https://health-products.canada.ca/dpd-bdpp/)
        - [Health Canada overview and access](https://www.canada.ca/en/health-canada/services/drugs-health-products/drug-products/drug-product-database.html)
        - [Download the official DPD open-data extract](https://open.canada.ca/data/en/dataset/bf55e42a-63cb-4556-bfd8-44f26e5a36fe)
        - [Open the exact 500-case comparison CSV](../data/hc/demos/dpd_model_comparison/1.0.0/cases.csv)
        """
    ),
    code(
        """
        CASES_PATH = ROOT / "data/hc/demos/dpd_model_comparison/1.0.0/cases.csv"
        REPORT_PATH = ROOT / "data/hc/demos/dpd_model_comparison/1.0.0/build_report.json"
        cases = pd.read_csv(CASES_PATH)
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

        cards = [
            ("7,017", "Product families", "Every representable marketed human-drug family in the frozen census"),
            ("14,034", "Bilingual cases", "One English and one French case per product"),
            ("500", "Comparison cases", "The same representative cases sent to every model"),
            ("8", "Required fields", "A strict pass requires the complete JSON contract"),
        ]
        display(
            HTML(
                '<div class="demo-grid">'
                + "".join(
                    f'<div class="demo-card"><div class="demo-kicker">{html.escape(title)}</div>'
                    f'<div class="demo-value">{value}</div><div class="demo-note">{html.escape(note)}</div></div>'
                    for value, title, note in cards
                )
                + "</div>"
            )
        )

        product_counts = (
            cases.drop_duplicates("product_id")["stratum"]
            .value_counts()
            .reindex(STRATUM_LABELS)
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.7, 1]})
        labels = [STRATUM_LABELS[value] for value in product_counts.index]
        bars = axes[0].barh(labels[::-1], product_counts.values[::-1], color="#176b87")
        axes[0].set_title("250 products reflect the census complexity mix")
        axes[0].set_xlabel("Product families")
        axes[0].grid(axis="x", alpha=.22)
        for bar, value in zip(bars, product_counts.values[::-1]):
            axes[0].text(value + 2, bar.get_y() + bar.get_height() / 2, f"{int(value)}", va="center", fontweight="bold")

        language_counts = cases["language"].value_counts().reindex(["en", "fr"])
        axes[1].pie(
            language_counts,
            labels=["English", "French"],
            autopct="%1.0f%%",
            startangle=90,
            colors=["#2f80ed", "#56ccf2"],
            wedgeprops={"width": .42, "edgecolor": "white"},
            textprops={"color": "#334e68", "weight": "bold"},
        )
        axes[1].set_title("Every product is tested bilingually")
        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 2. The task, one record in, one JSON object out

        The model is not asked for medical advice or an open-ended summary. It receives a rendered DPD source record and must copy the facts into one exact eight-field structure, without inventing anything.
        """
    ),
    code(
        """
        example = cases[
            (cases["language"] == "en")
            & (cases["stratum"] == "single_ingredient")
            & ~cases["input"].str.contains("VARIANT 2", regex=False)
        ].iloc[0]
        expected = json.loads(example["expected_output"])
        source_lines = example["input"].split("DPD SOURCE RECORD (ENGLISH)", 1)[-1].strip().splitlines()
        source_excerpt = "\\n".join(source_lines[:14])
        expected_pretty = json.dumps(expected, indent=2, ensure_ascii=False)

        display(
            HTML(
                f'''
                <div class="task-grid">
                  <div class="task-panel">
                    <div class="demo-kicker">Actual source record</div>
                    <div style="font-size:18px;font-weight:800;margin:5px 0 10px;">{html.escape(str(example["brand_name"]))}</div>
                    <pre style="white-space:pre-wrap;font-size:12px;line-height:1.45;margin:0;">{html.escape(source_excerpt)}</pre>
                  </div>
                  <div class="task-arrow">→</div>
                  <div class="task-panel" style="background:#f1fbf7;">
                    <div class="demo-kicker">Required model output</div>
                    <pre style="white-space:pre-wrap;font-size:12px;line-height:1.4;margin:10px 0 0;">{html.escape(expected_pretty)}</pre>
                  </div>
                </div>
                '''
            )
        )

        readable = pd.DataFrame(
            [
                ("DIN", ", ".join(expected["din"])),
                ("Brand name", expected["brand_name"]),
                (
                    "Active ingredient",
                    "; ".join(
                        f'{item["name"]} {item["strength"]} {item["unit"]}'
                        for item in expected["active_ingredients"]
                    ),
                ),
                ("Dosage form", ", ".join(expected["dosage_forms"])),
                ("Route", ", ".join(expected["routes"])),
                ("Schedule", ", ".join(expected["schedule"])),
                ("Status", expected["product_status"]),
                ("Company", expected["company"]),
            ],
            columns=["Required field", "Correct value"],
        )
        show_table(readable)
        """
    ),
    markdown(
        """
        ### What counts as correct?

        A **strict pass** means all eight required fields are present and correct in valid JSON.

        We also report an **average field score** so we can distinguish a nearly correct response from a completely wrong one. This is diagnostic information, it does not weaken the strict pass rule.

        The scorer allows only documented equivalences, including list order, letter case, an eight-digit DIN with leading zeroes, and numeric strengths such as `15`, `15.0`, or `"15"`. Missing facts, extra facts, wrong JSON types, and duplicated ingredients remain errors.
        """
    ),
    markdown(
        """
        ## 3. A fair model comparison

        Every model receives:

        1. The same 500 case IDs
        2. The same English and French source records
        3. The same instruction and output schema
        4. The same deterministic scorer
        5. No LLM judge calls

        GPT-5.4 mini has already completed the full 14,034-case census. The comparison reuses its matching 500 saved responses, so that work and cost are not repeated.
        """
    ),
    code(
        """
        RESULTS_PATH = ROOT / "results/evaluate_hc_dpd_comparison_500_results.csv"
        refresh_note = ""
        assembly = None

        if AUTO_REFRESH_LOCAL_RESULTS:
            previous_cwd = Path.cwd()
            try:
                os.chdir(ROOT)
                from evalanche.config import load_evaluation_config
                from evalanche.dpd_comparison import assemble_dpd_comparison_outputs
                from evalanche.evaluation import run_evaluation

                assembly = assemble_dpd_comparison_outputs(root_path=ROOT)
                evaluation_config_path = ROOT / "configs/evaluate_hc_dpd_comparison_500.yaml"
                evaluation_config = load_evaluation_config(evaluation_config_path)
                with redirect_stdout(io.StringIO()):
                    run_evaluation(
                        evaluation_config,
                        config_path=evaluation_config_path,
                    )
                refresh_note = "Local comparison artifacts were refreshed. No model calls were made."
            except (FileNotFoundError, ValueError, OSError) as error:
                refresh_note = f"Comparison refresh is pending: {error}"
            finally:
                os.chdir(previous_cwd)

        results = pd.read_csv(RESULTS_PATH) if RESULTS_PATH.exists() else pd.DataFrame()
        available_models = [
            model for model in MODEL_ORDER
            if not results.empty and model in set(results["model_name"].astype(str))
        ]

        if available_models:
            status_html = (
                f'<div class="status-ready"><strong>Comparison ready:</strong> '
                f'{len(available_models)} model(s), 500 identical cases per model.<br>'
                f'<span class="small-muted">{html.escape(refresh_note)}</span></div>'
            )
        else:
            status_html = (
                '<div class="status-pending"><strong>Full Mini baseline is ready.</strong> '
                'Run Terra first to unlock the cross-model comparison.<br>'
                f'<span class="small-muted">{html.escape(refresh_note)}</span></div>'
            )
        display(HTML(status_html))

        model_cards = []
        for model in MODEL_ORDER:
            info = MODEL_INFO[model]
            if model in available_models:
                status = "500 cases loaded"
                status_color = "#27ae60"
            elif model == "gpt_5_4_mini":
                status = "14,034-case census complete"
                status_color = "#2f80ed"
            else:
                status = "Not run yet"
                status_color = "#b8c2cc"
            model_cards.append(
                f'<div class="demo-card" style="border-top:5px solid {info["color"]};">'
                f'<div class="demo-kicker">{html.escape(info["role"])}</div>'
                f'<div style="font-size:20px;font-weight:800;margin:5px 0;">{html.escape(info["label"])}</div>'
                f'<div class="demo-note"><span style="color:{status_color};font-weight:800;">●</span> {status}</div>'
                '</div>'
            )
        display(HTML('<div class="demo-grid">' + "".join(model_cards) + "</div>"))

        full_cards = [
            ("77.1%", "Strict pass rate", FULL_MINI_RESULT["ci"]),
            ("0.923", "Average field score", "Across all 14,034 cases"),
            ("10,827", "Complete passes", "Every one satisfied the full contract"),
            ("0", "Generation failures", "All requests returned an output"),
        ]
        display(
            HTML(
                '<h3 style="margin-top:26px;">Full-census GPT-5.4 mini baseline</h3>'
                '<div class="demo-grid">'
                + "".join(
                    f'<div class="demo-card"><div class="demo-kicker">{title}</div>'
                    f'<div class="demo-value">{value}</div><div class="demo-note">{note}</div></div>'
                    for value, title, note in full_cards
                )
                + "</div>"
            )
        )
        """
    ),
    markdown(
        """
        ## 4. Performance, speed, and cost

        The full-census Mini result above establishes scale. The charts below use only the paired 500-case comparison, because model-to-model conclusions must come from identical cases.
        """
    ),
    code(
        """
        def wilson_interval(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
            if total == 0:
                return (float("nan"), float("nan"))
            p = passed / total
            denominator = 1 + z * z / total
            centre = (p + z * z / (2 * total)) / denominator
            margin = (
                z
                * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
                / denominator
            )
            return centre - margin, centre + margin


        def summarize_models(frame: pd.DataFrame) -> pd.DataFrame:
            records = []
            for model_name, group in frame.groupby("model_name", sort=False):
                scored = group[group["final_passed"].notna()].copy()
                passed = int(scored["final_passed"].astype(bool).sum())
                total = len(scored)
                ci_low, ci_high = wilson_interval(passed, total)
                costs = pd.to_numeric(group["generation_cost_usd"], errors="coerce")
                seconds = pd.to_numeric(group["generation_seconds"], errors="coerce")
                total_cost = costs.sum(min_count=1)
                records.append(
                    {
                        "model_name": str(model_name),
                        "model": MODEL_INFO[str(model_name)]["label"],
                        "cases": total,
                        "passed": passed,
                        "pass_rate": passed / total if total else np.nan,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "average_field_score": pd.to_numeric(scored["final_score"], errors="coerce").mean(),
                        "generation_failures": int((group["generation_status"].astype(str).str.lower() != "success").sum()),
                        "p50_latency_seconds": seconds.median(),
                        "p95_latency_seconds": seconds.quantile(.95),
                        "total_cost_usd": total_cost,
                        "cost_per_1000_usd": total_cost / len(group) * 1000 if pd.notna(total_cost) else np.nan,
                        "cost_per_strict_pass_usd": total_cost / passed if passed and pd.notna(total_cost) else np.nan,
                        "projected_census_cost_usd": total_cost / len(group) * CENSUS_CASE_COUNT if pd.notna(total_cost) else np.nan,
                    }
                )
            summary = pd.DataFrame(records)
            if summary.empty:
                return summary
            summary["_order"] = summary["model_name"].map({name: i for i, name in enumerate(MODEL_ORDER)})
            return summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)


        summary = summarize_models(results) if not results.empty else pd.DataFrame()

        if summary.empty:
            display(
                HTML(
                    '<div class="status-pending"><strong>Cross-model charts are waiting for results.</strong> '
                    'Complete Terra, then run this notebook again.</div>'
                )
            )
        else:
            table = summary[
                [
                    "model",
                    "cases",
                    "passed",
                    "pass_rate",
                    "average_field_score",
                    "total_cost_usd",
                    "cost_per_1000_usd",
                    "p95_latency_seconds",
                ]
            ].rename(
                columns={
                    "model": "Model",
                    "cases": "Cases",
                    "passed": "Strict passes",
                    "pass_rate": "Pass rate",
                    "average_field_score": "Avg. field score",
                    "total_cost_usd": "500-case cost",
                    "cost_per_1000_usd": "Cost per 1,000",
                    "p95_latency_seconds": "p95 latency",
                }
            )
            formatted_table = table.copy()
            formatted_table["Pass rate"] = formatted_table["Pass rate"].map(
                lambda value: f"{value:.1%}" if pd.notna(value) else "N/A"
            )
            formatted_table["Avg. field score"] = formatted_table["Avg. field score"].map(
                lambda value: f"{value:.3f}" if pd.notna(value) else "N/A"
            )
            formatted_table["500-case cost"] = formatted_table["500-case cost"].map(
                lambda value: f"${value:.2f}" if pd.notna(value) else "N/A"
            )
            formatted_table["Cost per 1,000"] = formatted_table["Cost per 1,000"].map(
                lambda value: f"${value:.2f}" if pd.notna(value) else "N/A"
            )
            formatted_table["p95 latency"] = formatted_table["p95 latency"].map(
                lambda value: f"{value:.2f} s" if pd.notna(value) else "N/A"
            )
            show_table(formatted_table)

            colors = [MODEL_INFO[name]["color"] for name in summary["model_name"]]
            labels = summary["model"].tolist()
            x = np.arange(len(summary))
            fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))

            rates = summary["pass_rate"].to_numpy()
            errors = np.vstack(
                [
                    rates - summary["ci_low"].to_numpy(),
                    summary["ci_high"].to_numpy() - rates,
                ]
            )
            bars = axes[0, 0].bar(x, rates, color=colors, yerr=errors, capsize=5)
            axes[0, 0].set_title("Strict pass rate")
            axes[0, 0].set_ylim(0, 1.02)
            axes[0, 0].set_xticks(x, labels, rotation=18, ha="right")
            axes[0, 0].set_ylabel("Share of 500 cases")
            axes[0, 0].grid(axis="y", alpha=.22)
            for bar, value in zip(bars, rates):
                axes[0, 0].text(bar.get_x() + bar.get_width() / 2, value + .025, f"{value:.1%}", ha="center", fontweight="bold")

            field_scores = summary["average_field_score"].to_numpy()
            bars = axes[0, 1].bar(x, field_scores, color=colors)
            axes[0, 1].set_title("Average field score")
            axes[0, 1].set_ylim(0, 1.02)
            axes[0, 1].set_xticks(x, labels, rotation=18, ha="right")
            axes[0, 1].grid(axis="y", alpha=.22)
            for bar, value in zip(bars, field_scores):
                axes[0, 1].text(bar.get_x() + bar.get_width() / 2, value + .02, f"{value:.3f}", ha="center", fontweight="bold")

            costs = summary["total_cost_usd"].to_numpy()
            bars = axes[1, 0].bar(x, costs, color=colors)
            axes[1, 0].set_title("Estimated cost for these 500 cases")
            axes[1, 0].set_xticks(x, labels, rotation=18, ha="right")
            axes[1, 0].set_ylabel("USD")
            axes[1, 0].grid(axis="y", alpha=.22)
            for bar, value in zip(bars, costs):
                if pd.notna(value):
                    axes[1, 0].text(bar.get_x() + bar.get_width() / 2, value, f"${value:.2f}", ha="center", va="bottom", fontweight="bold")

            latency = summary["p95_latency_seconds"].to_numpy()
            bars = axes[1, 1].bar(x, latency, color=colors)
            axes[1, 1].set_title("p95 response time")
            axes[1, 1].set_xticks(x, labels, rotation=18, ha="right")
            axes[1, 1].set_ylabel("Seconds")
            axes[1, 1].grid(axis="y", alpha=.22)
            for bar, value in zip(bars, latency):
                if pd.notna(value):
                    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}s", ha="center", va="bottom", fontweight="bold")

            plt.tight_layout()
            plt.show()

            display(
                HTML(
                    '<div class="small-muted">Error bars show 95% Wilson intervals. '
                    'Cost is estimated from recorded tokens and the versioned public Global Standard rate card, '
                    'not from a reconciled Azure invoice.</div>'
                )
            )
        """
    ),
    code(
        """
        if not summary.empty:
            fig, ax = plt.subplots(figsize=(9.5, 5.5))
            for _, row in summary.iterrows():
                info = MODEL_INFO[row["model_name"]]
                ax.scatter(
                    row["cost_per_1000_usd"],
                    row["pass_rate"],
                    s=180,
                    color=info["color"],
                    edgecolor="white",
                    linewidth=1.5,
                    zorder=3,
                )
                ax.annotate(
                    info["label"],
                    (row["cost_per_1000_usd"], row["pass_rate"]),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontweight="bold",
                )
            ax.set_title("The decision view: quality versus cost")
            ax.set_xlabel("Estimated cost per 1,000 cases, USD")
            ax.set_ylabel("Strict pass rate")
            ax.set_ylim(max(0, summary["pass_rate"].min() - .12), min(1.02, summary["pass_rate"].max() + .12))
            ax.grid(alpha=.22)
            plt.tight_layout()
            plt.show()

            projection = summary[
                ["model", "projected_census_cost_usd", "cost_per_strict_pass_usd"]
            ].rename(
                columns={
                    "model": "Model",
                    "projected_census_cost_usd": "Projected 14,034-case cost",
                    "cost_per_strict_pass_usd": "Cost per strict pass",
                }
            )
            projection["Projected 14,034-case cost"] = projection[
                "Projected 14,034-case cost"
            ].map(lambda value: f"${value:.2f}" if pd.notna(value) else "N/A")
            projection["Cost per strict pass"] = projection[
                "Cost per strict pass"
            ].map(lambda value: f"${value:.4f}" if pd.notna(value) else "N/A")
            show_table(projection)
            display(
                HTML(
                    '<div class="small-muted">The census figures are simple projections from the observed '
                    '500-case token use. They are planning estimates, not completed full-model runs.</div>'
                )
            )
        """
    ),
    markdown(
        """
        ## 5. Where do the models struggle?

        One headline number is not enough. We also check whether performance changes by language, product complexity, or output field.
        """
    ),
    code(
        """
        def parse_field_list(value) -> list[str]:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return []
            text = str(value).strip()
            if not text or text in {"[]", "nan"}:
                return []
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed]
                except Exception:
                    pass
            return [text]


        if not results.empty:
            language_rates = (
                results.groupby(["model_name", "language"])["final_passed"]
                .mean()
                .unstack("language")
                .reindex(available_models)
            )
            stratum_rates = (
                results.groupby(["model_name", "stratum"])["final_passed"]
                .mean()
                .unstack("stratum")
                .reindex(index=available_models, columns=STRATUM_LABELS)
            )

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.8]})
            language_rates.rename(columns={"en": "English", "fr": "French"}).plot(
                kind="bar",
                ax=axes[0],
                color=["#2f80ed", "#56ccf2"],
                width=.74,
            )
            axes[0].set_title("Pass rate by language")
            axes[0].set_xlabel("")
            axes[0].set_ylabel("Strict pass rate")
            axes[0].set_ylim(0, 1.02)
            axes[0].set_xticklabels([MODEL_INFO[name]["label"] for name in language_rates.index], rotation=18, ha="right")
            axes[0].legend(frameon=False)
            axes[0].grid(axis="y", alpha=.2)

            heat = stratum_rates.to_numpy(dtype=float)
            image = axes[1].imshow(heat, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
            axes[1].set_title("Pass rate by product complexity")
            axes[1].set_xticks(range(len(STRATUM_LABELS)), [STRATUM_LABELS[name] for name in STRATUM_LABELS], rotation=25, ha="right")
            axes[1].set_yticks(range(len(stratum_rates.index)), [MODEL_INFO[name]["label"] for name in stratum_rates.index])
            for row_index in range(heat.shape[0]):
                for column_index in range(heat.shape[1]):
                    value = heat[row_index, column_index]
                    if not np.isnan(value):
                        axes[1].text(column_index, row_index, f"{value:.0%}", ha="center", va="center", color="white" if value > .62 else "#102a43", fontweight="bold")
            fig.colorbar(image, ax=axes[1], fraction=.035, pad=.03)
            plt.tight_layout()
            plt.show()

            field_rows = []
            for _, row in results[~results["final_passed"].fillna(False)].iterrows():
                fields = parse_field_list(row.get("json_missing_fields"))
                fields += parse_field_list(row.get("json_extra_fields"))
                fields += parse_field_list(row.get("json_mismatched_fields"))
                if not fields and str(row.get("generation_status", "")).lower() != "success":
                    fields = ["generation failure"]
                for field in set(fields):
                    field_rows.append({"model_name": row["model_name"], "field": field})

            if field_rows:
                field_counts = (
                    pd.DataFrame(field_rows)
                    .groupby(["model_name", "field"])
                    .size()
                    .rename("failed_cases")
                    .reset_index()
                )
                top_failures = (
                    field_counts.sort_values(
                        ["model_name", "failed_cases", "field"],
                        ascending=[True, False, True],
                    )
                    .groupby("model_name")
                    .head(4)
                    .copy()
                )
                top_failures["Model"] = top_failures["model_name"].map(
                    lambda value: MODEL_INFO[value]["label"]
                )
                show_table(
                    top_failures[["Model", "field", "failed_cases"]]
                    .rename(columns={"field": "Field", "failed_cases": "Failed cases"})
                )
            else:
                display(HTML('<div class="status-ready">No strict failures were recorded.</div>'))
        else:
            display(HTML('<div class="status-pending">Breakdowns will appear when a 500-case comparison result is available.</div>'))
        """
    ),
    code(
        """
        if not results.empty and (~results["final_passed"].fillna(False)).any():
            failures = results[~results["final_passed"].fillna(False)].copy()
            failures["_score"] = pd.to_numeric(failures["final_score"], errors="coerce")
            near_miss = failures.sort_values("_score", ascending=False).iloc[0]
            mismatch = ", ".join(parse_field_list(near_miss.get("json_mismatched_fields"))) or "output contract"
            display(
                HTML(
                    f'''
                    <div class="demo-card" style="border-left:6px solid #f2994a;">
                      <div class="demo-kicker">Example near miss</div>
                      <div style="font-size:20px;font-weight:800;margin:6px 0;">
                        {html.escape(MODEL_INFO[str(near_miss["model_name"])]["label"])} on {html.escape(str(near_miss["brand_name"]))}
                      </div>
                      <div class="demo-note">
                        Field score: <strong>{float(near_miss["_score"]):.3f}</strong><br>
                        Problem field: <strong>{html.escape(mismatch)}</strong><br>
                        Why strict scoring failed: {html.escape(str(near_miss["evaluation_reason"]))}
                      </div>
                    </div>
                    '''
                )
            )
        """
    ),
    markdown(
        """
        ## 6. What should we conclude?

        The goal is not to declare one universally best model. It is to make the tradeoff visible, then choose according to the workload.
        """
    ),
    code(
        """
        if len(summary) >= 2:
            best_quality = summary.loc[summary["pass_rate"].idxmax()]
            lowest_cost = summary.loc[summary["cost_per_1000_usd"].idxmin()]
            best_value = summary.loc[summary["cost_per_strict_pass_usd"].idxmin()]
            conclusion_cards = [
                (
                    "Best quality in this sample",
                    best_quality["model"],
                    f'{best_quality["pass_rate"]:.1%} strict pass rate',
                    MODEL_INFO[best_quality["model_name"]]["color"],
                ),
                (
                    "Lowest operating cost",
                    lowest_cost["model"],
                    f'${lowest_cost["cost_per_1000_usd"]:.2f} per 1,000 cases',
                    MODEL_INFO[lowest_cost["model_name"]]["color"],
                ),
                (
                    "Lowest cost per complete result",
                    best_value["model"],
                    f'${best_value["cost_per_strict_pass_usd"]:.4f} per strict pass',
                    MODEL_INFO[best_value["model_name"]]["color"],
                ),
            ]
            display(
                HTML(
                    '<div class="demo-grid">'
                    + "".join(
                        f'<div class="demo-card" style="border-top:5px solid {color};">'
                        f'<div class="demo-kicker">{html.escape(title)}</div>'
                        f'<div style="font-size:22px;font-weight:800;margin:6px 0;">{html.escape(str(value))}</div>'
                        f'<div class="demo-note">{html.escape(detail)}</div></div>'
                        for title, value, detail, color in conclusion_cards
                    )
                    + "</div>"
                )
            )
            display(
                Markdown(
                    "**Decision rule:** Use the lowest-cost model that clears the quality requirement for the real application. "
                    "If strict schema compliance is mandatory, the pass-rate gap matters. If failed records can be automatically "
                    "retried or reviewed, the cost gap may matter more."
                )
            )
        elif len(summary) == 1:
            display(
                HTML(
                    '<div class="status-pending"><strong>The baseline is loaded, but one model cannot form a comparison.</strong> '
                    'Run GPT-5.6 Terra first, then rerun the notebook.</div>'
                )
            )
        else:
            display(
                HTML(
                    '<div class="status-pending"><strong>Next action:</strong> Run the 500-case Terra comparison. '
                    'The completed Mini census will be reused automatically.</div>'
                )
            )
        """
    ),
    markdown(
        """
        ---

        ### Method and boundaries

        - **Full baseline:** GPT-5.4 mini, 14,034 cases, 10,827 strict passes, 77.1% pass rate, 0.923 average field score.
        - **Cross-model design:** 500 paired cases from 250 product families, 250 English and 250 French.
        - **Sampling:** deterministic proportional stratification, seed `20260728`.
        - **Model versions:** GPT-5.4 mini `2026-03-17`; GPT-5.6 Luna, Terra, and Sol `2026-07-09`.
        - **Deployment class:** Global Standard, Canada East resource.
        - **Reasoning:** GPT-5.6 models explicitly use `reasoning_effort: none` for a comparable extraction baseline.
        - **Cost:** calculated from recorded input and output tokens using versioned public USD rate cards. It is an estimate, not a reconciled invoice.
        - **Scope:** this is extraction from rendered DPD records. It is not yet extraction from Product Monograph documents.

        Sources: [Health Canada DPD](https://www.canada.ca/en/health-canada/services/drugs-health-products/drug-products/drug-product-database.html), [DPD open data](https://open.canada.ca/data/en/dataset/bf55e42a-63cb-4556-bfd8-44f26e5a36fe), [Microsoft GPT-5.6 pricing](https://azure.microsoft.com/en-us/blog/gpt-5-6-now-available-in-microsoft-foundry/), [Azure OpenAI pricing](https://azure.microsoft.com/en-us/pricing/details/azure-openai/).
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


output = Path(__file__).with_name(
    "HC_DPD_Model_Comparison_Demo.ipynb"
)
output.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(output)
