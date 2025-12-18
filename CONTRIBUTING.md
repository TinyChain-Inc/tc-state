# Contributing to tc-state

`tc-state` prototypes TinyChain’s state subsystem (collections, `/state/media`,
bootstrap helpers). Keep contributions focused on refining the shared txfs
layout and serialization contracts so every adapter can hydrate the same tree.

## Before you start

- Read the workspace `ARCHITECTURE.md`, `ROADMAP.md`, and this crate’s
  `README.md`/`AGENTS.md` to understand the current scope (control-plane
  bootstrap, media experiments, retention defaults).
- Follow the repo-wide `CODE_STYLE.md`: grouped imports, rustfmt, and clippy
  cleanliness apply here.

## Development checklist

1. **Plan schema changes.** Update `ROADMAP.md` (or crate-local docs) when
   adding new `/state/...` prefixes, media layouts, or bootstrap files so
   hosts/clients can prepare for the new contract.
2. **Preserve URI ↔ txfs mirroring.** Any code that touches disk must keep the
   `<data-dir>/<segment>` layout perfectly aligned with the URI helper
   constants. Add tests if you introduce new helpers.
3. **Serialization symmetry.** Whenever you edit `destream::IntoStream` or
   `FromStream` implementations, add/refresh round-trip tests. Tensors, maps,
   and media descriptors all need deterministic shapes.
4. **Testing.**
   - Run `cargo test -p tc-state --all-features` locally.
   - Add focused unit tests for new media chunks, retention policies, or
     bootstrap loaders.
5. **Docs.** Reflect behavioral changes in `README.md`, `AGENTS.md`, and any
   relevant workspace doc. Call out migration steps (e.g., required txfs
   migrations) so hosts can upgrade safely.
6. **No fallback flows.** Remove legacy pathways instead of adding new ones; the
   crate should model the single canonical state layout.

## Pre-submit

- `cargo fmt`
- `cargo clippy --all-targets --all-features`
- `cargo test -p tc-state --all-features`
- Update docs + roadmap notes, and summarize rollout considerations in your PR
  description (data migrations, queue updates, compliance implications, etc.).

## Rights and licensing

By contributing to this crate you represent that (a) you authored the work (or
otherwise have the rights to contribute it) and (b) you transfer and assign all
right, title, and interest in the contribution to the TinyChain Open-Source
Project for distribution under the TinyChain open-source license (Apache 2.0,
see the root `LICENSE`). Contributions must be free of third-party claims or
encumbrances.
