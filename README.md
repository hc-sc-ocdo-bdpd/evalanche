# Mission Evalanche

Mission Evalanche is a lightweight model evaluation toolkit for task-grounded LLM evaluation.

The goal is to help answer:

> What model should I use for this task?

The core principle is that there is no single best model in the abstract. Model choice depends on the task, data, risk tolerance, cost, latency, language needs, and deployment constraints.

## First working slice

This first slice implements criteria-based LLM-as-judge evaluation over pre-generated model outputs.

Input:

- test case
- expected output
- model name
- model output
- scoring rubric

Output:

- criterion scores
- weighted score
- pass/fail
- judge rationale
- CSV report

## Setup

Copy the environment template:

```bash
cp .env.example .env