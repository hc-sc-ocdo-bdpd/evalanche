# Judge protocol records

Store real, versioned task-specific judge protocols here. Begin with
[`examples/judge_validation/protocol.yaml`](../../examples/judge_validation/protocol.yaml),
then replace every synthetic input and declaration.

A protocol identifies one exact task, measured construct, rubric, prompt,
judge deployment and version, settings, human-review process, validation data,
controls, thresholds, and intended evidence level. Changing any material part
requires a new protocol version and a new validation report.

Do not commit private human-review content or provider outputs merely to place
a protocol here. The YAML may reference approved local paths, while public
artifacts retain hashes and appropriately redacted summary evidence.

See [`docs/llm_judges.md`](../../docs/llm_judges.md) for the schema, command,
statistics, and claim limits.
