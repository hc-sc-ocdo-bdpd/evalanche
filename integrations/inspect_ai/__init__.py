"""Optional Inspect AI execution path for Evalanche benchmarks."""

INSPECT_AI_VERSION = "0.3.257"
OPENAI_VERSION = "2.54.0"
DEFAULT_BENCHMARK_REFERENCE = "hc_dpd_structured_extraction@0.2.0"
DEFAULT_MODEL_ID = "gpt_5_6_sol"
DEFAULT_HISTORICAL_OUTPUTS_PATH = (
    "data/generated/hc_dpd_census_all_models_outputs.csv"
)
DPD_REPLAY_MODEL_IDS = (
    "gpt_5_4_mini",
    "gpt_5_6_luna",
    "gpt_5_6_terra",
    "gpt_5_6_sol",
)

__all__ = [
    "DEFAULT_BENCHMARK_REFERENCE",
    "DEFAULT_HISTORICAL_OUTPUTS_PATH",
    "DEFAULT_MODEL_ID",
    "DPD_REPLAY_MODEL_IDS",
    "INSPECT_AI_VERSION",
    "OPENAI_VERSION",
]
