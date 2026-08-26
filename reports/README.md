# Published results

This directory contains compact, versioned evaluation evidence suitable for
source control and review. Large provider outputs, checkpoints, and detailed
local result tables remain ignored under `data/generated/` and `results/`.
Compact bundles retain hashes that tie reports back to their source evidence.

A benchmark report may be:

- **ranked**, when its contract supports an official ordering among complete
  compatible runs;
- **descriptive**, when measurements are useful but an official ordering is not
  justified;
- **ineligible** for a particular run when generation or scoring is incomplete.

Descriptive reporting keeps pass rate, field score, slices, cost, latency,
reliability, and paired comparisons visible. `frozen` describes reproducibility
of the benchmark contract, not the strength of every possible claim.

The generated registry surface is `reports/benchmarks/index.html`. Each
benchmark version contains active-run pointers, immutable compact bundles,
paired comparisons, and equivalent CSV, JSON, Markdown, and sortable HTML
reports. Compatibility fingerprints prevent results from a changed dataset,
prompt, or scorer from silently entering the same comparison.

Case-study analysis directories may add field, slice, taxonomy, and selected
case evidence. Derived artifacts should record their source and method so they
remain distinguishable from the benchmark's primary scoring contract.
