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
          <div style="font-size:14px;letter-spacing:.12em;text-transform:uppercase;opacity:.82;">Evalanche validation analysis</div>
          <h1 style="font-size:38px;line-height:1.1;margin:10px 0 12px;">What do the full DPD benchmark results actually mean?</h1>
          <p style="font-size:19px;line-height:1.45;margin:0;max-width:940px;">
            Four models, 14,034 bilingual cases each, analyzed by field, language, complexity, product family, failure mechanism, cost, and paired disagreements.
          </p>
        </div>

        This notebook reads the compact, versioned analysis release. It makes
        no Azure or model calls. The strict deterministic leaderboard remains
        authoritative while diagnostic sensitivity analyses expose what is
        driving each failure.
        """
    ),
    code(
        """
        from __future__ import annotations

        import json
        import sys
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

        ANALYSIS_DIR = (
            ROOT / "reports/hc_dpd_census/0.2.0/analysis"
        )
        RAW_RESULTS = (
            ROOT
            / "results/evaluate_hc_dpd_census_all_models_results.csv"
        )
        AUTO_REFRESH_LOCAL_ANALYSIS = False

        MODEL_ORDER = [
            "gpt_5_4_mini",
            "gpt_5_6_luna",
            "gpt_5_6_terra",
            "gpt_5_6_sol",
        ]
        MODEL_LABELS = {
            "gpt_5_4_mini": "GPT-5.4 mini",
            "gpt_5_6_luna": "GPT-5.6 Luna",
            "gpt_5_6_terra": "GPT-5.6 Terra",
            "gpt_5_6_sol": "GPT-5.6 Sol",
        }
        MODEL_COLORS = {
            "gpt_5_4_mini": "#2f80ed",
            "gpt_5_6_luna": "#27ae60",
            "gpt_5_6_terra": "#f2994a",
            "gpt_5_6_sol": "#eb5757",
        }
        STRATUM_LABELS = {
            "single_ingredient": "Single ingredient",
            "multi_ingredient": "Multiple ingredients",
            "multi_variant": "Multiple variants",
            "multi_ingredient_multi_variant": (
                "Multiple ingredients + variants"
            ),
        }
        ERROR_LABELS = {
            "schema_type_only": "Scalar/list type only",
            "duplicate_items_only": "Duplicate list items only",
            "compound_label_split_only": "Compound label split only",
            "schema_and_duplicates": "Schema + duplicates",
            "schema_and_label_split": "Schema + label split",
            "multiple_repairable": "Multiple structural",
            "text_or_value_error": "Text or value",
            "missing_or_extra_fields": "Missing or extra fields",
            "invalid_json": "Invalid JSON",
            "generation_error": "Generation error",
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
                  .analysis-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; margin:14px 0 20px; }
                  .analysis-card { border:1px solid #d9e2ec; border-radius:14px; padding:18px; background:#fff; box-shadow:0 3px 12px rgba(16,42,67,.06); }
                  .analysis-kicker { color:#627d98; text-transform:uppercase; letter-spacing:.08em; font-size:11px; font-weight:700; }
                  .analysis-value { color:#102a43; font-size:29px; line-height:1.15; font-weight:800; margin:5px 0; }
                  .analysis-note { color:#486581; font-size:13px; line-height:1.4; }
                  .status-ready { border-left:6px solid #27ae60; background:#edf9f1; padding:14px 18px; border-radius:10px; }
                  .status-pending { border-left:6px solid #f2c94c; background:#fff9e6; padding:14px 18px; border-radius:10px; }
                  table.analysis-table { border-collapse:collapse; width:100%; margin:12px 0 20px; font-size:13px; }
                  table.analysis-table th { background:#eaf2f8; color:#243b53; padding:10px 12px; text-align:left; border-bottom:2px solid #bcccdc; }
                  table.analysis-table td { padding:9px 12px; border-bottom:1px solid #d9e2ec; text-align:left; }
                  table.analysis-table tr:nth-child(even) td { background:#f8fbfd; }
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
                        classes="analysis-table",
                    )
                )
            )
        """
    ),
    code(
        """
        if AUTO_REFRESH_LOCAL_ANALYSIS:
            if not RAW_RESULTS.is_file():
                raise FileNotFoundError(
                    "AUTO_REFRESH_LOCAL_ANALYSIS requires the ignored "
                    f"case-level result file: {RAW_RESULTS}"
                )
            from evalanche.dpd_analysis import build_dpd_census_analysis

            build_dpd_census_analysis(root_path=ROOT)

        manifest = json.loads(
            (ANALYSIS_DIR / "analysis_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        overview = pd.read_csv(ANALYSIS_DIR / "model_overview.csv")
        slices = pd.read_csv(ANALYSIS_DIR / "slice_performance.csv")
        fields = pd.read_csv(ANALYSIS_DIR / "field_accuracy.csv")
        taxonomy = pd.read_csv(ANALYSIS_DIR / "error_taxonomy.csv")
        pairwise = pd.read_csv(
            ANALYSIS_DIR / "pairwise_tradeoffs.csv"
        )
        frontier = pd.read_csv(ANALYSIS_DIR / "frontier_cases.csv")
        selected = pd.read_csv(ANALYSIS_DIR / "selected_cases.csv")
        audit = pd.read_csv(ANALYSIS_DIR / "evidence_audit.csv")

        overview["model_label"] = overview["model_name"].map(
            MODEL_LABELS
        )
        overview = overview.set_index("model_name").loc[
            MODEL_ORDER
        ].reset_index()

        expected_audit_cases = (
            manifest["audit_set"]["primary_model_failures"]
            + manifest["audit_set"]["comparison_only_failures"]
            + manifest["audit_set"][
                "shared_lower_tier_failure_sample"
            ]
            + manifest["audit_set"]["all_model_pass_sample"]
        )
        assert manifest["benchmark"]["case_count"] == 14_034
        assert manifest["source"]["rows"] == 56_136
        assert overview["cases"].eq(14_034).all()
        assert len(frontier) == 55
        assert len(selected) == expected_audit_cases == 105
        assert len(audit) == len(selected)
        assert audit["case_id"].is_unique
        assert audit["audit_status"].eq("complete").all()
        assert audit["strict_score_correct"].eq("yes").all()
        assert audit["adjudication"].eq("retain_result").all()
        assert audit["human_signoff_status"].eq("not_claimed").all()

        cards = [
            ("56,136", "Evaluated outputs", "Four models on identical cases"),
            ("14,034", "Cases per model", "7,017 products in English and French"),
            ("55", "Frontier edge cases", "Failed by Sol, Terra, or both"),
            ("105", "Audited cases", "Source, labels, and scores rechecked"),
        ]
        display(
            HTML(
                '<div class="analysis-grid">'
                + "".join(
                    f'<div class="analysis-card"><div class="analysis-kicker">{title}</div>'
                    f'<div class="analysis-value">{value}</div><div class="analysis-note">{note}</div></div>'
                    for value, title, note in cards
                )
                + "</div>"
                + '<div class="status-ready"><b>Automated evidence audit complete.</b> '
                + 'All 105 selected cases retained their strict results. '
                + 'Independent human sign-off is not claimed.</div>'
            )
        )
        """
    ),
    markdown(
        """
        ## 1. The strict result is clear, the operational choice is not

        Sol is the strict-score leader. Terra costs roughly half as much and
        misses only 35 additional complete records across 14,034 cases. Before
        choosing between them, we need to understand those specific records.
        """
    ),
    code(
        """
        headline = overview[
            [
                "model_label",
                "passed_cases",
                "pass_rate",
                "average_field_score",
                "bilingual_product_pass_rate",
                "p95_latency_seconds",
                "total_cost_usd",
            ]
        ].copy()
        headline.columns = [
            "Model",
            "Strict passes",
            "Case pass rate",
            "Average field score",
            "Both-language product rate",
            "p95 latency (s)",
            "Run cost (USD)",
        ]
        headline["Case pass rate"] = headline[
            "Case pass rate"
        ].map(lambda value: f"{value:.2%}")
        headline["Average field score"] = headline[
            "Average field score"
        ].map(lambda value: f"{value:.5f}")
        headline["Both-language product rate"] = headline[
            "Both-language product rate"
        ].map(lambda value: f"{value:.2%}")
        headline["p95 latency (s)"] = headline[
            "p95 latency (s)"
        ].map(lambda value: f"{value:.2f}")
        headline["Run cost (USD)"] = headline[
            "Run cost (USD)"
        ].map(lambda value: f"${value:.2f}")
        show_table(headline)

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 4.8),
            gridspec_kw={"width_ratios": [1.35, 1]},
        )
        bars = axes[0].bar(
            overview["model_label"],
            overview["pass_rate"] * 100,
            color=[
                MODEL_COLORS[model]
                for model in overview["model_name"]
            ],
        )
        axes[0].set_title("Strict complete-record accuracy")
        axes[0].set_ylabel("Pass rate (%)")
        axes[0].set_ylim(65, 101.5)
        axes[0].tick_params(axis="x", rotation=18)
        axes[0].grid(axis="y", alpha=.22)
        for bar, value in zip(bars, overview["pass_rate"]):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value * 100 + .7,
                f"{value:.2%}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        axes[1].scatter(
            overview["total_cost_usd"],
            overview["pass_rate"] * 100,
            s=160,
            c=[
                MODEL_COLORS[model]
                for model in overview["model_name"]
            ],
            edgecolor="white",
            linewidth=1.5,
        )
        for row in overview.itertuples():
            axes[1].annotate(
                row.model_label,
                (row.total_cost_usd, row.pass_rate * 100),
                xytext=(6, 7),
                textcoords="offset points",
                fontsize=9,
            )
        axes[1].set_title("Observed quality versus run cost")
        axes[1].set_xlabel("Estimated run cost (USD)")
        axes[1].set_ylabel("Pass rate (%)")
        axes[1].set_ylim(65, 101.5)
        axes[1].grid(alpha=.22)
        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 2. Bilingual dependency changes the interpretation

        Each product contributes one English and one French case. The
        both-language product pass rate asks a practical question: would the
        model extract that product correctly in both official languages?
        """
    ),
    code(
        """
        language = slices[slices["slice_type"] == "language"].copy()
        language["model_label"] = language["model_name"].map(
            MODEL_LABELS
        )
        language_pivot = (
            language.pivot(
                index="model_name",
                columns="slice_value",
                values="pass_rate",
            )
            .loc[MODEL_ORDER]
        )

        x = np.arange(len(MODEL_ORDER))
        width = .34
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        axes[0].bar(
            x - width / 2,
            language_pivot["en"] * 100,
            width,
            label="English",
            color="#2f80ed",
        )
        axes[0].bar(
            x + width / 2,
            language_pivot["fr"] * 100,
            width,
            label="French",
            color="#56ccf2",
        )
        axes[0].set_xticks(
            x,
            [MODEL_LABELS[model] for model in MODEL_ORDER],
            rotation=18,
        )
        axes[0].set_ylim(55, 101.5)
        axes[0].set_ylabel("Pass rate (%)")
        axes[0].set_title("English and French strict accuracy")
        axes[0].legend(frameon=False)
        axes[0].grid(axis="y", alpha=.22)

        product_rates = overview[
            [
                "model_label",
                "bilingual_product_pass_rate",
            ]
        ]
        bars = axes[1].barh(
            product_rates["model_label"][::-1],
            product_rates["bilingual_product_pass_rate"][::-1]
            * 100,
            color=[
                MODEL_COLORS[model]
                for model in overview["model_name"][::-1]
            ],
        )
        axes[1].set_xlim(50, 101.5)
        axes[1].set_xlabel("Products passing in both languages (%)")
        axes[1].set_title("Bilingual product-family success")
        axes[1].grid(axis="x", alpha=.22)
        for bar, value in zip(
            bars,
            product_rates[
                "bilingual_product_pass_rate"
            ][::-1],
        ):
            axes[1].text(
                value * 100 + .6,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2%}",
                va="center",
                fontweight="bold",
                fontsize=9,
            )
        plt.tight_layout()
        plt.show()

        gap_table = pd.DataFrame(
            {
                "Model": [
                    MODEL_LABELS[model] for model in MODEL_ORDER
                ],
                "French minus English": [
                    f"{(language_pivot.loc[model, 'fr'] - language_pivot.loc[model, 'en']):+.2%}"
                    for model in MODEL_ORDER
                ],
                "Both-language products": [
                    f"{int(row.both_languages_passed_products):,} / {int(row.product_families):,}"
                    for row in overview.itertuples()
                ],
            }
        )
        show_table(gap_table)
        """
    ),
    markdown(
        """
        ## 3. Mini and Luna fail for different reasons

        Mini is strongest on multi-variant products and weaker on simple
        products because it often turns scalar fields into singleton lists.
        Luna shows the opposite complexity pattern and a large French penalty.
        This is primarily schema following, not uniformly poor extraction.
        """
    ),
    code(
        """
        stratum = slices[slices["slice_type"] == "stratum"].copy()
        stratum_pivot = (
            stratum.pivot(
                index="slice_value",
                columns="model_name",
                values="pass_rate",
            )
            .reindex(STRATUM_LABELS)
        )
        fig, ax = plt.subplots(figsize=(12, 5.2))
        x = np.arange(len(stratum_pivot))
        width = .19
        for index, model in enumerate(MODEL_ORDER):
            ax.bar(
                x + (index - 1.5) * width,
                stratum_pivot[model] * 100,
                width,
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
            )
        ax.set_xticks(
            x,
            [
                STRATUM_LABELS[value]
                for value in stratum_pivot.index
            ],
            rotation=13,
        )
        ax.set_ylim(25, 101.5)
        ax.set_ylabel("Strict pass rate (%)")
        ax.set_title("Performance changes sharply with product structure")
        ax.legend(frameon=False, ncol=4, loc="lower center")
        ax.grid(axis="y", alpha=.22)
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        overall_taxonomy = taxonomy[
            taxonomy["language"] == "all"
        ].copy()
        tax_pivot = (
            overall_taxonomy.pivot(
                index="model_name",
                columns="error_type",
                values="count",
            )
            .fillna(0)
            .reindex(MODEL_ORDER)
        )
        category_order = [
            category
            for category in [
                "schema_type_only",
                "schema_and_duplicates",
                "schema_and_label_split",
                "duplicate_items_only",
                "compound_label_split_only",
                "text_or_value_error",
                "missing_or_extra_fields",
                "invalid_json",
                "generation_error",
            ]
            if category in tax_pivot
        ]
        taxonomy_colors = {
            "schema_type_only": "#4e79a7",
            "schema_and_duplicates": "#76b7b2",
            "schema_and_label_split": "#59a14f",
            "duplicate_items_only": "#f28e2b",
            "compound_label_split_only": "#edc949",
            "text_or_value_error": "#e15759",
            "missing_or_extra_fields": "#b07aa1",
            "invalid_json": "#9c755f",
            "generation_error": "#79706e",
        }

        fig, ax = plt.subplots(figsize=(11.5, 5.2))
        bottom = np.zeros(len(tax_pivot))
        for category in category_order:
            values = tax_pivot[category].to_numpy()
            ax.bar(
                [MODEL_LABELS[model] for model in MODEL_ORDER],
                values,
                bottom=bottom,
                label=ERROR_LABELS.get(category, category),
                color=taxonomy_colors[category],
            )
            bottom += values
        ax.set_yscale("symlog", linthresh=20)
        ax.set_ylabel("Failed cases, symlog scale")
        ax.set_title("Failure mechanism, strict failures only")
        ax.legend(
            frameon=False,
            ncol=2,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        ax.grid(axis="y", alpha=.22)
        plt.tight_layout()
        plt.show()

        sensitivity = overview[
            [
                "model_label",
                "pass_rate",
                "schema_type_repair_sensitivity_rate",
                "structural_repair_sensitivity_rate",
                "remaining_text_or_value_failures",
            ]
        ].copy()
        sensitivity.columns = [
            "Model",
            "Strict",
            "Type-shape diagnostic",
            "All structural diagnostic",
            "Text/value failures remain",
        ]
        for column in (
            "Strict",
            "Type-shape diagnostic",
            "All structural diagnostic",
        ):
            sensitivity[column] = sensitivity[column].map(
                lambda value: f"{value:.2%}"
            )
        show_table(sensitivity)
        display(
            Markdown(
                "**Important:** diagnostic transformations are sensitivity "
                "analyses only. They do not change the published scores and "
                "do not predict an API-enforced structured-output rerun."
            )
        )
        """
    ),
    markdown(
        """
        ## 4. Field accuracy explains the score gap

        A complete record fails if any one field fails. Mini and Luna can
        therefore have similar strict accuracy while Luna retains a higher
        average field score.
        """
    ),
    code(
        """
        field_order = [
            "din",
            "brand_name",
            "active_ingredients",
            "dosage_forms",
            "routes",
            "schedule",
            "product_status",
            "company",
        ]
        overall_fields = fields[
            (fields["language"] == "all")
            & fields["field_name"].isin(field_order)
        ]
        field_pivot = (
            overall_fields.pivot(
                index="model_name",
                columns="field_name",
                values="field_accuracy",
            )
            .reindex(index=MODEL_ORDER, columns=field_order)
        )
        fig, ax = plt.subplots(figsize=(12, 4.2))
        image = ax.imshow(
            field_pivot.to_numpy() * 100,
            vmin=75,
            vmax=100,
            cmap="Blues",
            aspect="auto",
        )
        ax.set_xticks(
            np.arange(len(field_order)),
            [value.replace("_", " ").title() for value in field_order],
            rotation=24,
            ha="right",
        )
        ax.set_yticks(
            np.arange(len(MODEL_ORDER)),
            [MODEL_LABELS[model] for model in MODEL_ORDER],
        )
        for row in range(len(MODEL_ORDER)):
            for column in range(len(field_order)):
                value = field_pivot.iloc[row, column] * 100
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 88 else "#102a43",
                    fontsize=9,
                    fontweight="bold",
                )
        ax.set_title("Field-level accuracy (%)")
        fig.colorbar(image, ax=ax, fraction=.022, pad=.02)
        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 5. Sol and Terra differ on only 49 directional cases

        Their errors are not nested. Sol uniquely passes 42 cases, Terra
        uniquely passes 7, and both fail 6. The content of those cases matters
        more than another aggregate confidence interval.
        """
    ),
    code(
        """
        frontier_pair = pairwise[
            (
                (pairwise["model_a"] == "gpt_5_6_sol")
                & (pairwise["model_b"] == "gpt_5_6_terra")
            )
            | (
                (pairwise["model_a"] == "gpt_5_6_terra")
                & (pairwise["model_b"] == "gpt_5_6_sol")
            )
        ].iloc[0]
        if frontier_pair["model_a"] == "gpt_5_6_sol":
            sol_only = int(frontier_pair["model_a_only_passed"])
            terra_only = int(frontier_pair["model_b_only_passed"])
        else:
            sol_only = int(frontier_pair["model_b_only_passed"])
            terra_only = int(frontier_pair["model_a_only_passed"])
        both_failed = int(frontier_pair["both_failed"])

        cards = [
            (str(sol_only), "Sol-only passes", "Terra failed"),
            (str(terra_only), "Terra-only passes", "Sol failed"),
            (str(both_failed), "Shared failures", "Both failed"),
            (
                f"${float(frontier_pair['incremental_cost_per_net_pass_usd']):.2f}",
                "Increment per net pass",
                "Sol run cost minus Terra, divided by 35 net passes",
            ),
        ]
        display(
            HTML(
                '<div class="analysis-grid">'
                + "".join(
                    f'<div class="analysis-card"><div class="analysis-kicker">{title}</div>'
                    f'<div class="analysis-value">{value}</div><div class="analysis-note">{note}</div></div>'
                    for value, title, note in cards
                )
                + "</div>"
            )
        )

        frontier_counts = (
            frontier["pair_outcome"]
            .value_counts()
            .rename_axis("Outcome")
            .reset_index(name="Cases")
        )
        frontier_counts["Outcome"] = frontier_counts[
            "Outcome"
        ].str.replace("_", " ").str.replace(
            "gpt 5 6", "GPT-5.6", regex=False
        ).str.title()
        show_table(frontier_counts)
        """
    ),
    code(
        """
        sol_failures = frontier[
            ~frontier["gpt_5_6_sol_passed"]
        ][
            [
                "case_id",
                "language",
                "stratum",
                "gpt_5_6_sol_mismatched_fields",
                "gpt_5_6_sol_error_type",
                "gpt_5_6_terra_passed",
                "diagnostic_focus",
            ]
        ].copy()
        sol_failures.columns = [
            "Case",
            "Language",
            "Stratum",
            "Mismatched fields",
            "Diagnostic type",
            "Terra passed",
            "Diagnostic focus",
        ]
        sol_failures["Stratum"] = sol_failures["Stratum"].map(
            STRATUM_LABELS
        )
        sol_failures["Diagnostic type"] = sol_failures[
            "Diagnostic type"
        ].map(ERROR_LABELS)
        sol_failures["Terra passed"] = sol_failures[
            "Terra passed"
        ].map(lambda value: "Yes" if value else "No")
        show_table(sol_failures)

        compound = taxonomy[
            (taxonomy["language"] == "all")
            & (
                taxonomy["error_type"]
                == "compound_label_split_only"
            )
        ].set_index("model_name")["count"]
        strict_gap = int(
            overview.set_index("model_name").loc[
                "gpt_5_6_sol", "passed_cases"
            ]
            - overview.set_index("model_name").loc[
                "gpt_5_6_terra", "passed_cases"
            ]
        )
        adjusted_gap = strict_gap - int(
            compound.get("gpt_5_6_terra", 0)
        ) + int(compound.get("gpt_5_6_sol", 0))
        display(
            HTML(
                '<div class="status-ready"><b>Compound-label audit:</b> '
                f'Terra has {int(compound.get("gpt_5_6_terra", 0))} '
                'compound-label split-only failures and Sol has '
                f'{int(compound.get("gpt_5_6_sol", 0))}. '
                'The independent parser confirmed that rendered multi-values '
                'use <code> | </code> while commas remain part of official '
                'labels. These failures were retained, so the strict gap '
                f'remains {strict_gap} cases. A hypothetical permissive '
                f'rescore would have reduced it to {adjusted_gap}.</div>'
            )
        )
        """
    ),
    markdown(
        """
        ## 6. The selected evidence audit is complete

        The selected audit set is deliberately not a random accuracy sample. It
        contains every frontier edge case, then adds stratified checks for
        shared lower-tier failures and apparent all-model successes. The
        automated audit rebuilt the cases from the frozen DPD archive,
        independently parsed the expected answers, and rescored all selected
        model outputs.
        """
    ),
    code(
        """
        selection_counts = (
            selected.groupby(
                ["selection_priority", "selection_group"],
                as_index=False,
            )
            .size()
            .sort_values("selection_priority")
        )
        selection_counts["selection_group"] = (
            selection_counts["selection_group"]
            .str.replace("_", " ")
            .str.title()
        )
        selection_counts.columns = [
            "Priority",
            "Selection group",
            "Cases",
        ]
        show_table(selection_counts)

        audit_summary = (
            audit.groupby(
                ["operational_severity", "error_owner"],
                as_index=False,
            )
            .size()
            .sort_values(
                ["operational_severity", "error_owner"]
            )
        )
        audit_summary.columns = [
            "Operational severity",
            "Error owner",
            "Cases",
        ]
        show_table(audit_summary)
        display(
            HTML(
                '<div class="status-ready"><b>Audit result:</b> '
                f'{len(audit)} of {len(selected)} cases retained, with '
                'no label corrections, exclusions, or scoring-rule changes. '
                f'<code>{ANALYSIS_DIR / "evidence_audit.csv"}</code><br>'
                'Independent human sign-off is not claimed.</div>'
            )
        )
        """
    ),
    markdown(
        """
        ## 7. Leaderboard interpretation

        - **Strict-score leader:** GPT-5.6 Sol.
        - **Cost-quality tradeoff:** GPT-5.6 Terra costs roughly half as much
          and trails by 35 complete records on this benchmark.
        - **Important production experiment:** rerun a controlled subset using
          API-enforced structured outputs before dismissing lower-cost models,
          because Mini and Luna are dominated by type-shape failures.
        - **Benchmark conclusion:** the structured DPD task is saturated for
          frontier models. The next benchmark should move to bilingual Product
          Monograph extraction, where the input is genuinely unstructured.

        This is a versioned benchmark leaderboard, not a formal Health Canada
        model recommendation. Independent human sign-off would be required
        only to describe the release as human-validated.
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
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


output_path = Path(__file__).with_name(
    "HC_DPD_Census_Analysis.ipynb"
)
output_path.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(output_path)
