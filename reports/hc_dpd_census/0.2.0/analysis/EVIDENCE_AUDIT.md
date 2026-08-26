# DPD selected-case evidence audit

**Status:** Automated evidence audit complete. Independent human sign-off is
not claimed.

The 105 selected cases were audited without model calls. The audit:

- rebuilt every selected input, expected answer, and provenance record from
  the frozen marketed DPD archive;
- independently parsed every rendered source record into the expected
  eight-field JSON contract;
- rescored all 420 selected model outputs using an independent implementation
  of the configured canonical comparison rules; and
- assigned error ownership, operational severity, and an adjudication while
  preserving the original selected-case evidence.

## Result

| Check or outcome | Result |
| --- | ---: |
| Selected cases rebuilt from source | 105 / 105 |
| Expected answers independently reproduced | 105 / 105 |
| Model outputs rescored | 420 / 420 |
| Strict-score disagreements | 0 |
| Results retained | 105 / 105 |
| Expected answers corrected | 0 |
| Cases excluded | 0 |
| Scoring rules revised | 0 |

The strict leaderboard is unchanged.

| Severity | Cases |
| --- | ---: |
| None, all-model pass controls | 25 |
| Low, structural, duplicate, label-split, or non-material text issues | 63 |
| Medium, changed source text or values in operationally material fields | 17 |
| High | 0 |

## Compound labels

All 26 selected comma-bearing label cases were retained as valid strict
failures. The rendered record uses ` | ` between multiple source values, while
the comma remains part of an official label such as
`SPRAY, BAG-ON-VALVE`. Splitting the label at the comma changes the source
value and violates the instruction not to infer values.

This resolves the earlier sensitivity question for the frozen benchmark. Sol
remains ahead of Terra by 35 complete records, not 10.

## Administrative source text

The case containing the ingredient label prefix `(FR)` was also retained. This
benchmark measures fidelity to frozen source labels. A task that intentionally normalizes administrative markers would use a
different scoring contract; this leaderboard preserves the frozen source labels.

## Provenance

Detailed case decisions are in `evidence_audit.csv`. Machine-readable hashes,
verification counts, and aggregate outcomes are in
`evidence_audit_summary.json`. Running `audit-dpd-evidence` with the default
paths refreshes both the analysis manifest and the enclosing release manifest.

The audit is sufficient to support the versioned benchmark leaderboard as an
automatically validated artifact. Independent human validation is not claimed; the completed claim is automated
evidence validation against the frozen source and scoring contract.
