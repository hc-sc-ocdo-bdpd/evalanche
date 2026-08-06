# Access sets

Store dated access-confirmation YAML files here. Access sets contain model IDs
and confirmation notes, never credentials or endpoint secrets.

Copy the template from
[`examples/local_comparison/access_set.example.yaml`](../../examples/local_comparison/access_set.example.yaml)
and replace every placeholder before use.

An access set is validated when it is used. `registry-validate` also checks
every YAML file in this directory against the current model registry.
