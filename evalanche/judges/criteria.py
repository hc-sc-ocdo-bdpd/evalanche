from __future__ import annotations

import json
from typing import Any

import pandas as pd

from evalanche.config import EvalConfig
from evalanche.judges.validation import validate_judge_response
from evalanche.llm import LLMClient


class CriteriaJudge:
    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.client = LLMClient(
            model=config.judge.model,
            temperature=config.judge.temperature,
            max_retries=config.judge.max_retries,
        )

    def judge_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        records: list[dict[str, Any]] = []

        for row in df.to_dict(orient="records"):
            records.append(self.judge_case(row))

        return pd.DataFrame(records)

    def judge_case(self, row: dict[str, Any]) -> dict[str, Any]:
        messages = self._build_messages(row)
        result = self.client.complete_json(
            messages,
            validator=self._validate_response,
        )

        # Validate again at the consumer boundary. The LLM client validates
        # inside its retry loop, while this check protects against any future
        # client implementation that does not apply the callback.
        self._validate_response(result)

        usage = result.pop("_usage", {})

        criterion_scores = result.get("criteria", {})
        normalized_score = self._weighted_normalized_score(criterion_scores)
        passed = normalized_score >= self.config.scoring.pass_threshold

        output = {
            "case_id": row["case_id"],
            "model_name": row["model_name"],
            "weighted_score": round(normalized_score, 4),
            "passed": passed,
            "overall_reason": result.get("overall_reason", ""),
            "raw_judge_result": json.dumps(result, ensure_ascii=False),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

        for criterion in self.config.criteria:
            score_obj = criterion_scores.get(criterion.name, {})
            output[f"{criterion.name}_score"] = score_obj.get("score")
            output[f"{criterion.name}_reason"] = score_obj.get("reason", "")

        return output

    def _weighted_normalized_score(self, criterion_scores: dict[str, Any]) -> float:
        score_min = self.config.scoring.score_min
        score_max = self.config.scoring.score_max
        score_range = score_max - score_min

        if score_range <= 0:
            raise ValueError("score_max must be greater than score_min")

        weighted_total = 0.0
        weight_total = 0.0

        for criterion in self.config.criteria:
            score_obj = criterion_scores.get(criterion.name, {})
            raw_score = score_obj.get("score")

            if raw_score is None:
                continue

            raw_score = float(raw_score)
            normalized = (raw_score - score_min) / score_range
            normalized = max(0.0, min(1.0, normalized))

            weighted_total += normalized * criterion.weight
            weight_total += criterion.weight

        if weight_total == 0:
            return 0.0

        return weighted_total / weight_total
    
    def _validate_response(self, response: dict[str, Any]) -> None:
        validate_judge_response(response, self.config)   

    def _build_messages(self, row: dict[str, Any]) -> list[dict[str, str]]:
        criteria_text = "\n".join(
            [
                f"- {c.name} (weight {c.weight}): {c.description}"
                for c in self.config.criteria
            ]
        )
        response_example = {
            "criteria": {
                criterion.name: {
                    "score": self.config.scoring.score_min,
                    "reason": "brief reason",
                }
                for criterion in self.config.criteria
            },
            "overall_reason": "brief overall reason",
        }

        system = f"""
You are an expert evaluator of language model outputs.

Your job is to evaluate a candidate model output for the task below.

Task:
{self.config.task.description}

Scoring:
- Score each criterion from {self.config.scoring.score_min} to {self.config.scoring.score_max}.
- {self.config.scoring.score_min} means completely unacceptable.
- {self.config.scoring.score_max} means excellent.
- Use the expected output as the reference answer.
- Penalize unsupported claims, contradictions, and missing important information.
- Be strict but fair.
- Include every configured criterion exactly once.
- Do not add or rename criteria.
- Return only valid JSON.

Criteria:
{criteria_text}

Return JSON in this exact structure:
{json.dumps(response_example, indent=2)}
""".strip()

        user = f"""
Input:
{row["input"]}

Expected output:
{row["expected_output"]}

Candidate model output:
{row["model_output"]}
""".strip()

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]