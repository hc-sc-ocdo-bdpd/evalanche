# Model decision record template

Copy this file for a real decision. Complete only the sections the stakes
justify, but do not omit access, hard requirements, evidence limitations, or
the review trigger.

## Decision

- Decision owner:
- Date:
- Intended task:
- Intended environment:
- Selected model or retained shortlist:
- Decision status: exploratory, internal decision, or publishable claim

## Task contract

- Representative inputs:
- Required outputs:
- Required languages and modalities:
- Unacceptable errors:
- Human review or fallback:
- Stakes and reversibility:

## Access-confirmed candidates

| Model deployment ID | Provider or runtime | Exact version | Access basis | Confirmed on |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

Models mentioned in public evidence but not available remain background only.

## Hard requirements

| Requirement | Threshold or rule | Owner | Evidence | Candidate outcomes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

## Public evidence

| Source | Reviewed on | What it measures | Relevant finding | Strength | Important limit |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

## Local comparison, if run

- Benchmark ID and version:
- Dataset and split:
- Case count and groups:
- Prompt version:
- Scorer and canonicalization version:
- Judge protocol and calibration, if any:
- Model and provider versions:
- Run date and environment:
- Access-set ID and date:
- Artifact paths and hashes:

## Observed evidence

| Candidate | Quality and uncertainty | Important failures | Cost | Latency | Reliability | Missing evidence |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |

## Tradeoff

- What evidence materially changed the decision?
- Which difference is practically important?
- Which uncertainty remains?
- Which risk or disadvantage is being accepted?
- Why is more testing not justified now?

## Decision rationale

Record the user's or stakeholder's reasoning. If an explicit weighted policy
was used, preserve the approved thresholds, weights, access profiles, minimum
margin, and any sensitivity analysis. Do not present those preferences as
empirical facts.

## Review trigger

Revisit the decision when any selected trigger occurs:

- model version or provider alias changes;
- price, quota, latency, or reliability changes materially;
- access, policy, region, or privacy rules change;
- task distribution or output contract changes;
- a new access-confirmed candidate has relevant evidence;
- monitoring detects a specified failure level;
- scheduled review date:

## Approval

- Decision owner:
- Risk or domain reviewer, if required:
- Approval date:
- Notes:
