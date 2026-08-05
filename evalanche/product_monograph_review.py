from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from evalanche.product_monograph import (
    PM_NATIVE_PDF_CASES_PATH,
    PM_NATIVE_PDF_REVIEW_PATH,
    PM_NATIVE_PDF_REVIEW_STATUSES,
    PM_OUTPUT_DIR,
    PM_SOURCES_PATH,
    _build_native_pdf_review_queue,
)


PM_FIELD_EVIDENCE_PATH = PM_OUTPUT_DIR / "field_evidence.csv.gz"
REVIEWED_STATUSES = {
    "approved",
    "rejected",
    "needs_correction",
}


def _is_utc_timestamp(value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return False
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.utcoffset() == timedelta(0)


def check_product_monograph_label_review(
    *,
    root_path: str | Path = ".",
    review_path: str | Path = PM_NATIVE_PDF_REVIEW_PATH,
    cases_path: str | Path = PM_NATIVE_PDF_CASES_PATH,
    evidence_path: str | Path = PM_FIELD_EVIDENCE_PATH,
    sources_path: str | Path = PM_SOURCES_PATH,
) -> dict[str, Any]:
    """Validate review integrity, metadata, progress, and promotion status."""

    root = Path(root_path).resolve()
    resolved_review_path = root / review_path
    if not resolved_review_path.is_file():
        raise FileNotFoundError(
            f"Product Monograph label review not found: {resolved_review_path}"
        )

    native_cases = pd.read_csv(root / cases_path, dtype=str)
    evidence = pd.read_csv(root / evidence_path, dtype=str)
    sources = pd.read_csv(root / sources_path, dtype=str)
    queue = _build_native_pdf_review_queue(
        root=root,
        native_cases=native_cases,
        evidence=evidence,
        sources=sources,
        review_path=review_path,
    )

    queue = queue.fillna("").copy()
    queue["human_review_status"] = (
        queue["human_review_status"].astype(str).str.strip().str.casefold()
    )
    invalid_statuses = sorted(
        set(queue["human_review_status"]) - PM_NATIVE_PDF_REVIEW_STATUSES
    )
    if invalid_statuses:
        raise ValueError(f"Invalid label review statuses: {invalid_statuses}")

    issues: list[str] = []
    for row in queue.to_dict(orient="records"):
        status = str(row["human_review_status"])
        reviewer = str(row["reviewer"]).strip()
        reviewed_at = str(row["reviewed_at_utc"]).strip()
        notes = str(row["notes"]).strip()
        item_id = str(row["review_item_id"])

        if status in REVIEWED_STATUSES:
            if not reviewer:
                issues.append(f"{item_id}: reviewed item has no reviewer")
            if not _is_utc_timestamp(reviewed_at):
                issues.append(
                    f"{item_id}: reviewed_at_utc must be an ISO 8601 UTC "
                    "timestamp"
                )
            if status in {"rejected", "needs_correction"} and not notes:
                issues.append(
                    f"{item_id}: {status} requires an explanatory note"
                )
        elif reviewer or reviewed_at:
            issues.append(
                f"{item_id}: pending item cannot have reviewer metadata"
            )

    status_counts = {
        status: int(count)
        for status, count in queue["human_review_status"]
        .value_counts()
        .sort_index()
        .items()
    }
    for status in sorted(PM_NATIVE_PDF_REVIEW_STATUSES):
        status_counts.setdefault(status, 0)
    approved = status_counts["approved"]
    reviewed = sum(status_counts[status] for status in REVIEWED_STATUSES)
    total = int(len(queue))
    cases_approved = 0
    for _, group in queue.groupby("native_case_id", sort=False):
        if set(group["human_review_status"]) == {"approved"}:
            cases_approved += 1

    by_field = (
        queue.groupby(["field", "human_review_status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .to_dict(orient="records")
    )
    by_language = (
        queue.groupby(["language", "human_review_status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .to_dict(orient="records")
    )
    by_split = (
        queue.groupby(["split", "human_review_status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .to_dict(orient="records")
    )
    promotion_ready = approved == total and not issues
    return {
        "review_path": resolved_review_path,
        "items": total,
        "reviewed": reviewed,
        "approved": approved,
        "remaining": total - approved,
        "cases": int(queue["native_case_id"].nunique()),
        "cases_approved": cases_approved,
        "status_counts": status_counts,
        "by_field": by_field,
        "by_language": by_language,
        "by_split": by_split,
        "issues": issues,
        "valid": not issues,
        "promotion_ready": promotion_ready,
    }
