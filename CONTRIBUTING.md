# Contributing to tc-state

Read this repository's [invariants](AGENTS.md). The parent workspace
[contributor guide](https://github.com/TinyChain-Inc/tcv2/blob/main/CONTRIBUTING.md)
is non-normative integration context for contributors working in a superproject
checkout.

Keep changes within universal State, native routing/resolution, Class/Object
semantics, views, and codecs. Filesystem policy, application bootstrap,
transports, and collection implementations belong to their respective owners.

Before opening a pull request, run:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

When a wire representation changes, update both streaming directions and the
shared fixtures in the same change.
