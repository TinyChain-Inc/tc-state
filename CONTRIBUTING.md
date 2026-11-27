# Contributing to tc-state

Read this repository's [invariants](AGENTS.md).

Keep changes within universal State, native routing/resolution, Class/Object
semantics, views, and codecs. Filesystem policy, application bootstrap,
transports, and collection implementations belong to their respective owners.

Before review, select the affected library or named integration targets and
strict Clippy. Public/shared changes include their boundary regressions. The
complete owner suite, required before release, is:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

When a wire representation changes, update both streaming directions and the
shared fixtures in the same change.

## Module ownership

`state` owns universal values and conversions; `class` owns Classes and instances;
`route` binds native operations; `resolve` resolves scalar and reference values.
`gateway` defines transaction-bound host access. Operation scheduling and the
executor implementation belong to the host. `codec` and `view` retain separate
serialization and projection responsibilities. Private unit tests under
`tests/unit/` follow these implementation owners.

Direct Tensor arithmetic and standalone Tensor codec assertions belong to
Collection tests. State retains tagging, context isolation, projection, and
partial-decode cancellation coverage.
