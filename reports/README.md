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
