# Product Monograph architecture review

Reviewed: 2026-08-04

## Verdict

Keep the progressed Product Monograph work. It is a strong, auditable
evidence-window extraction benchmark and a useful addition to Evalanche. Do
not present it as a native-PDF or end-to-end document-understanding benchmark.

The strongest architecture keeps that diagnostic intact and adds a separate
native-PDF benchmark. The two contracts answer different questions and must
have separate compatibility fingerprints, execution gates, and leaderboards.

## What the progressed work got right

- It moved Evalanche from benchmark-specific scripts toward reusable model,
  dataset, benchmark, result-bundle, and leaderboard contracts.
- It froze a bilingual 40-product cohort with 80 distinct source-document
  hashes, explicit exclusions, development and held-out splits, and source
  lineage.
- It attached one or more source pages to every scored field item and retained
  deterministic JSON scoring, operational data, and compact result bundles.
- It kept the raw Product Monograph PDFs out of Git while preserving official
  URLs, SHA-256 values, byte sizes, and page counts.
- It produced a useful extraction diagnostic with real English and French
  evidence, rather than extending the earlier DPD-rendered task and calling it
  document extraction.

This work should not be discarded or rewritten around PDFs. It isolates a
valuable layer of the system and provides the paired baseline needed to
interpret native-PDF results.

## The measurement boundary that needed correction

The evidence-window builder uses the expected labels to locate supporting
pages, then sends text from those pages to the model. This is valid when the
research question is:

> Can the model extract and serialize the right facts after an oracle supplies
> the relevant evidence pages?

It is not valid evidence for:

> Can the model find and interpret the right facts in a complete Product
> Monograph?

The second question includes retrieval, layout, page rendering, long-context
behavior, and visual reading. Native PDF input matters because the OpenAI
Responses file-input contract supplies both extracted text and rendered page
images to vision-capable models. Azure's Responses documentation likewise
documents complete PDF input by Base64 data or file ID. See the official
[OpenAI file-input guide](https://developers.openai.com/api/docs/guides/file-inputs)
and [Azure Responses API guide](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/responses).

The progressed benchmark was therefore narrower than the repo's most valuable
target, but it was not a wrong turn. The incorrect move would have been to
silently replace its text windows with PDFs under the same benchmark identity
or to publish its scores as document-understanding results.

## Defects found and corrected

| Finding | Why it mattered | Resolution |
| --- | --- | --- |
| Dependency merge conflict and detached development requirements | A clean install and CI run were not reliable | Resolved the conflict and restored application dependencies through `requirements-dev.txt` |
| GPT-5.4 Mini received unsupported `temperature` sampling | All 80 calls failed before producing model outputs | Removed temperature for reasoning-model routes; the old failed run remains visible but unranked |
| Failed and partial runs could receive leaderboard ranks | Infrastructure failure looked like model quality | Ranking now requires zero generation failures and a score for every case |
| Evidence windows could be mistaken for full-document input | Claims would exceed what the benchmark measured | Renamed and documented the release as an evidence-window diagnostic |
| Input modality and model compatibility were implicit | An incompatible route could be treated like a bad model | Added benchmark-required capability gates for PDF input, Responses, and vision |
| No native file-input path existed | Complete-PDF evaluation could not run through the registry | Added hash-verified Base64 PDF input through the Responses adapter |
| A draft benchmark could create publishable-looking output | Provisional labels and untested transport could leak into a leaderboard | Registered execution remains blocked for `draft`; explicit local experiments are unregistered and cannot change leaderboards |
| The benchmark index linked to missing draft leaderboards | The generated report surface contained a broken link | Drafts without leaderboards now render as non-clickable status cards |

## Options considered

| Option | Strength | Limitation | Decision |
| --- | --- | --- | --- |
| Keep only evidence windows | Cheap, stable, easy to diagnose | Cannot measure retrieval, layout, or native PDF handling | Retain as a diagnostic, not the primary end-to-end benchmark |
| Replace evidence windows with PDFs under the same ID | Superficially simple | Invalidates score comparability and hides a major contract change | Reject |
| Use only native PDFs | Measures the most valuable end-to-end behavior | Harder to diagnose failures and more provider-dependent | Add, but keep the window baseline |
| Send full extracted text | Broad provider compatibility and lower visual complexity | Does not measure page images or native PDF processing | Possible future benchmark with a separate ID |
| Render every page to images locally | Explicit and provider-neutral visual contract | High token cost and different from provider-native PDF processing | Possible future benchmark with a separate ID |
| Keep evidence-window and native-PDF benchmarks side by side | Separates retrieval from extraction and preserves prior work | Requires paired reporting and careful label governance | Chosen |

## Implemented target architecture

The evidence-window benchmark remains frozen:

```text
hc_product_monograph_structured_extraction@0.1.0
```

The native-PDF benchmark is a separate draft:

```text
hc_product_monograph_native_pdf_extraction@0.1.0
```

Each native case contains a task prompt plus one repository-relative file
descriptor. The request path rejects absolute or escaping paths, accepts only
PDF media types, enforces file-count and 50 MB limits, verifies the frozen
SHA-256, and sends the complete file through the Responses API. The PDFs are
downloaded locally from the source lock and are not committed.

The provisional expected JSON is inherited from the evidence-window release,
but no selected page text, page number, expected output, or source metadata is
placed in the model prompt. A 390-item review queue records the source page and
human approval state for every scored item.

## How to interpret paired results

| Evidence window | Native PDF | Most useful interpretation |
| --- | --- | --- |
| Strong | Strong | Retrieval and extraction both work well |
| Strong | Weak | Investigate retrieval, layout, context, rendering, or PDF transport |
| Weak | Weak | Investigate extraction, schema compliance, canonicalization, or labels |
| Weak | Strong | Investigate prompt sensitivity, oracle-page selection effects, or run validity |

Do not average the two scores or merge them into one rank. Input processing can
differ between providers, so comparisons are valid only within a declared and
versioned input contract.

## Required next sequence

1. Review and approve or correct all 390 native-PDF label items against their
   exact hash-locked source pages.
2. Acquire and verify all 80 PDFs locally.
3. Smoke-test one development English case and one development French case on
   every intended provider route. Confirm output text, token usage, cost
   reporting, retry behavior, and the provider's accepted PDF request shape.
4. Lock the prompt and release a promoted dataset and benchmark version.
5. Rerun GPT-5.4 Mini on the evidence-window benchmark with the corrected
   request configuration.
6. Run the promoted native-PDF benchmark one model at a time, with checkpoints
   and conservative concurrency.
7. Compare paired case and field outcomes against the evidence-window
   diagnostic. Publish each leaderboard separately.

The label review and provider smoke test are real release gates, not paperwork.
No native-PDF model ranking should be published before they are complete.
