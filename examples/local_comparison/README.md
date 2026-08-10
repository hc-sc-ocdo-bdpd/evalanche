# Local comparison input templates

These small synthetic files show the minimum case shape for four common task
families. Replace every example with representative cases from the real task.

| Template | Evaluation route | Intended use |
| --- | --- | --- |
| `classification_cases.csv` | `exact` | Closed-label classification |
| `structured_extraction_cases.csv` | `json` | JSON extraction with required fields |
| `factual_cases.csv` | `exact` | Short answers with one accepted reference |
| `open_ended_cases.csv` | `judge` | Exploratory rubric-based review |

All templates include:

```text
case_id,input,expected_output,evaluation_type
```

The extra `group_id`, `language`, `split`, and `risk` columns demonstrate
metadata that can be preserved for grouped sampling and slice analysis.

The open-ended template is not decision-grade by itself. Build human-reviewed
calibration examples and follow
[`docs/llm_judges.md`](../../docs/llm_judges.md) before using an LLM judge for
an important conclusion. The complete provider-free fixture is in
[`examples/judge_validation/`](../judge_validation/README.md).

See [`docs/local_comparison.md`](../../docs/local_comparison.md) for the full
workflow.
