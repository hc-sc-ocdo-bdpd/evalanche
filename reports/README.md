# Published results

This directory contains compact, versioned evaluation releases that are
appropriate for source control and review.

Large model outputs, checkpoints, and case-level result tables remain under
the ignored `data/generated/` and `results/` directories. A published release
records their hashes and sizes so the summarized result remains tied to the
original evidence without placing hundreds of megabytes in Git.

Each release should include:

- a concise scope and interpretation document;
- aggregate model metrics;
- paired comparisons for shared cases;
- evaluation and generation metadata;
- a manifest for both published and retained raw artifacts;
- an explicit validation status.

Publication does not imply that labels or conclusions have received human
validation. A release must state whether manual review, grouped analysis, and
error adjudication are complete.

Analysis subdirectories may add compact field, slice, taxonomy, and
case-review tables derived from retained case-level evidence. Derived files
must record their source hash, method version, deterministic sample seed, and
their own integrity hashes. Diagnostic transformations must remain separate
from the primary score unless the benchmark is explicitly versioned.

The generated registry surface is `reports/benchmarks/index.html`. Each
benchmark version contains active-run pointers, immutable compact bundles,
paired comparisons, and equivalent CSV, JSON, Markdown, and sortable HTML
leaderboards. Compatibility hashes prevent results from a changed dataset,
prompt, or scorer from entering the same table.
