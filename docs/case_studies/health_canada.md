# Health Canada case studies

Evalanche includes three related Health Canada benchmark paths. They are useful
public examples of bilingual structured evaluation and remain benchmark assets
in their own right. They are not the main model-selection handbook and they do
not define which models a new user can access.

## At a glance

| Benchmark | Input | Cases | Status | Public result |
| --- | --- | ---: | --- | --- |
| DPD structured extraction 0.2.0 | Deterministic bilingual rendering of Drug Product Database records | 14,034 | Frozen | Four-model compact release and leaderboard |
| Product Monograph evidence-window extraction 0.1.0 | Label-selected text windows from official monographs | 80 | Frozen | Four-model leaderboard |
| Product Monograph native-PDF extraction 0.1.0 | Complete official English and French PDFs | 80 | Draft | Local provisional experiments only |

Each result applies only to the named input contract, prompt, dataset, model
route, and scoring policy.

## Drug Product Database census

### What it measures

The DPD task asks a model to reproduce selected structured fields from a
deterministic bilingual case rendering. Strict pass requires the complete JSON
record to match after symmetric canonicalization. It is primarily a structured
copying and schema-following task, not an unstructured document benchmark.

### Frozen result

| Model route in the release | Strict passes | Pass rate | Average field score | Observed cost |
| --- | ---: | ---: | ---: | ---: |
| GPT-5.6 Sol | 14,021 / 14,034 | 99.91% | 99.99% | US$75.52 |
| GPT-5.6 Terra | 13,986 / 14,034 | 99.66% | 99.95% | US$37.76 |
| GPT-5.4 Mini | 10,827 / 14,034 | 77.15% | 92.34% | US$11.40 |
| GPT-5.6 Luna | 10,662 / 14,034 | 75.97% | 95.92% | US$15.12 |

Cost and quality are deliberately separate. The table is not a product
recommendation.

The automated analysis includes product-family dependence, English and French
slices, field accuracy, failure mechanisms, every frontier disagreement, and
a deterministic 105-case evidence set. The evidence audit rebuilt sources,
parsed expected answers independently, and rescored 420 selected model outputs.
Independent human sign-off is not claimed.

Open:

- [release overview](../../reports/hc_dpd_census/0.2.0/README.md);
- [analysis and validation](../../reports/hc_dpd_census/0.2.0/analysis/README.md);
- [selected-case evidence audit](../../reports/hc_dpd_census/0.2.0/analysis/EVIDENCE_AUDIT.md);
- [sortable leaderboard](../../reports/benchmarks/hc_dpd_structured_extraction/0.2.0/leaderboard.html).

## Product Monograph evidence window

### What it measures

This task supplies only the text windows selected to contain the expected
label evidence. It isolates extraction and JSON fidelity after relevant pages
have already been found.

It does not measure:

- finding evidence in the full monograph;
- native PDF transport;
- page layout and visual interpretation;
- long-document retrieval;
- complete-PDF context behavior.

### Frozen result

| Model route in the release | Strict passes | Pass rate | Average field score | Observed cost | Run status |
| --- | ---: | ---: | ---: | ---: | --- |
| GPT-5.6 Sol | 64 / 80 | 80.00% | 93.75% | US$0.4365 | Eligible |
| GPT-5.6 Luna | 56 / 80 | 70.00% | 91.25% | US$0.0863 | Eligible |
| GPT-5.6 Terra | 56 / 80 | 70.00% | 90.00% | US$0.2154 | Eligible |
| GPT-5.4 Mini | 0 / 80 | 0.00% | 0.00% | Unknown | Ineligible because every generation failed |

The failed route remains visible rather than being silently removed. It is not
ranked alongside complete runs.

Open:

- [benchmark card and runbook](../hc_benchmark/HC_PRODUCT_MONOGRAPH_BENCHMARK.md);
- [Markdown leaderboard](../../reports/benchmarks/hc_product_monograph_structured_extraction/0.1.0/leaderboard.md);
- [sortable leaderboard](../../reports/benchmarks/hc_product_monograph_structured_extraction/0.1.0/leaderboard.html).

## Product Monograph native PDF

### What it measures

This separate draft sends each complete official PDF through a native
file-input route. It adds retrieval, layout, context, file transport, and
provider PDF processing to the extraction problem.

Scores from the evidence-window and native-PDF tasks must not be mixed into one
ranking. A difference between them can help diagnose whether the bottleneck is
retrieval and document handling or extraction and schema fidelity.

### Current status

- 80 PDFs representing 40 English and French product pairs.
- Complete source lock with URL, file hash, byte size, page count, and language
  checks.
- Deterministic JSON scoring for brand, ingredients, strengths, dosage forms,
  and routes.
- Smoke, screen, and standard tiers of 2, 20, and 80 cumulative PDFs.
- Project records describe completed local experiments, but no native-PDF
  leaderboard is tracked as a public release.
- Human reference review remains 0 of 390 scored facts in the release state.

The 390 items are facts across 80 PDFs, not 390 documents. They include
individual ingredients, strengths, dosage forms, and routes.

The expected labels inherit automated alignment, page-evidence checks, and
targeted overrides from the evidence-window work. This is useful provisional
evidence, but it is not independent human labeling.

Until the review is complete:

- results are local, provisional, and unranked;
- the benchmark remains `draft`;
- normal registered execution and publication are blocked;
- models can still be tested locally without repeating prior outputs;
- label review is not required for the main Evalanche handbook or software.

Open the [native-PDF benchmark plan](../hc_benchmark/HC_PRODUCT_MONOGRAPH_NATIVE_PDF_BENCHMARK.md)
for acquisition, source verification, review, tier, and publication details.

## Compare another model you can access

Do not run every registry model. Add the available route to a dated access set
first.

Preflight a native-PDF smoke run:

```bash
docker compose run --rm evalanche \
  python -m evalanche.cli run-benchmark \
  --benchmark hc_product_monograph_native_pdf_extraction@0.1.0 \
  --tier smoke \
  --all-compatible \
  --access-set configs/access_sets/my_available_models.yaml \
  --preflight-only
```

After reviewing the projected cost, repeat with:

```text
--experiment --max-cost-usd <reviewed_cap>
```

Summarize access-confirmed standard results:

```bash
docker compose run --rm evalanche \
  python -m evalanche.cli summarize-benchmark \
  --benchmark hc_product_monograph_native_pdf_extraction@0.1.0 \
  --tier standard \
  --access-set configs/access_sets/my_available_models.yaml
```

Use `--all-results` only for a historical evidence summary. That mode does not
assert that any included route remains available.

## Publication boundaries

### DPD

The compact release is published with automated validation and an explicit
statement that independent human sign-off is not claimed.

### Evidence window

The benchmark and current leaderboard are frozen. Interpret the task as an
oracle-window extraction test, not an end-to-end PDF result.

### Native PDF

Promotion requires, at minimum:

- review and approval of all 390 reference facts;
- a new dataset version for any label correction;
- complete PDF integrity checks;
- locked prompt before held-out evaluation;
- documented token and cost coverage;
- frozen dataset, benchmark, scorer, and model setup;
- identical cases for every public leaderboard entry;
- publication of limitations and retained raw-artifact hashes.

Expanding the PDF corpus is deliberately deferred. Adding one model to the
fixed 80-case task costs only that model's 80 calls. Expanding the benchmark
creates a new release and may require comparable new runs across every model.

## Detailed documentation

- [Health Canada dataset inventory](../hc_benchmark/HC_DATASET_INVENTORY.md)
- [DPD full-census runbook](../hc_benchmark/HC_DPD_CENSUS_RUNBOOK.md)
- [DPD analysis and validation](../hc_benchmark/HC_DPD_CENSUS_ANALYSIS.md)
- [DPD and Product Monograph specification](../hc_benchmark/HC_DPD_PM_BENCHMARK_SPEC.md)
- [Product Monograph evidence-window card](../hc_benchmark/HC_PRODUCT_MONOGRAPH_BENCHMARK.md)
- [Product Monograph native-PDF plan](../hc_benchmark/HC_PRODUCT_MONOGRAPH_NATIVE_PDF_BENCHMARK.md)
- [Architecture review](../hc_benchmark/PRODUCT_MONOGRAPH_ARCHITECTURE_REVIEW.md)
