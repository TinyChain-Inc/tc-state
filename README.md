# tc-state

`tc-state` prototypes the TinyChain state subsystem that every host and client
relies on: collection storage, `/state/media` primitives, and helpers that keep
the txfs layout aligned with canonical TinyChain URIs. Treat this crate as the
design sandpit for migrating v1 state semantics into the control-plane-aware v2
model.

## Role in the project

- Defines control-plane bootstrap helpers so hosts can honor publisher policies
  (via `control_plane.toml` and `authorized_publishers.toml`) before the ledger
  shards are online.
- Provides the canonical txfs layout contract for state, library, and media
  paths so adapters and queues can hydrate the same tree without bespoke glue.
- Hosts the recorded-media experiments (`/state/media/...`) and temporal cursor
  APIs that `tc-server` will expose over HTTP/WebSocket/WebTransport once the
  `media` feature lands.

See `ROADMAP.md` for the phase breakdown (control-plane readiness, media storage,
temporal access patterns) and cross-references to `tc-server`, `tc-chain`, and
client deliverables.

## Txfs layout invariant

TinyChain persists state via a single transactional directory (`data_dir`) per
host. Every state path mirrors its URI segments exactly:

- `/state/foo/bar` ↔ `<data_dir>/state/foo/bar`
- `/lib/publisher/library/1.0.0` ↔ `<data_dir>/lib/publisher/library/1.0.0`
- `/state/media/publisher/assets/<id>` ↔ `<data_dir>/state/media/publisher/assets/<id>`

There are no alternate roots or adapter-specific mount points. Queue services,
PyO3, HTTP, and future transports all talk to the same layout, so any code in
`tc-state` that writes to disk must preserve this invariant. The integration
tests planned in the roadmap will enforce it automatically.

## Control-plane bootstrap helpers

Until the governance shards ship, `tc-state` components load the following files
from disk so installs and requests can be authorized:

- `control_plane.toml` – per-library/service metadata (publisher IDs, allowed
  capabilities, rollout policy) installed alongside the payload under txfs.
- `authorized_publishers.toml` – bootstrap registry mapping publisher IDs to
  public keys, capability scopes, rotation cadence, and references to the
  private-key material managed out-of-band.

Helper tooling (planned `tc-state/bin/seed_registry.rs`) reads these files,
validates key references, and primes the on-disk store so `tc-server` can start
with a consistent view of the control plane even before `tc-chain` is live. Once
the BlockChain shard exists, the same data migrates into the ledger without
rewriting the local layout.

## Media storage experiments

The `ROADMAP.md` phases introduce chunked storage and temporal cursor APIs for
`/state/media`. Key expectations:

- Recorded media lives in ordinary TinyChain collections; large blobs are chunked
  and deduplicated under txfs with metadata rows describing content type,
  duration, custody tags, and temporal offsets.
- Temporal access (seek/iterate/window) is expressed as state traits so
  `tc-server` can implement transports behind a feature flag without changing
  how data is stored.
- MetricsChain counters (bytes stored/fetched, queue retries) and BlockChain
  custody events must be emitted whenever media state changes, aligning with the
  control-plane roadmap.

Client helpers (`client/py`, `client/js`) will call into these APIs once the
chunk manager stabilizes.

## PII-aware retention defaults

- `tc-state` assumes every write is PII unless the publisher explicitly marks it
  as non-PII in the manifest. Collections therefore default to the shortest
  retention window (`retention = "minimal"`, scrub-on-delete) and tenant-scoped
  access. Publishers can override this (e.g., to keep financial ledgers for 7
  years) by including a `compliance` block in the manifest that references the
  applicable regulation.
- LogChain ingestion (run by the host, described in `ROADMAP.md`) hashes or
  drops any field flagged as `PII` before persisting it, so the state subsystem
  does not have to guard against accidental log leakage.
- Media blobs under `/state/media/...` inherit the same custody tags. Delete
  queues must respect `forget_after` timestamps to satisfy GDPR/CCPA erasure
  requests.

## Serialization contract

- `destream::FromStream` and `IntoStream` implementations must always agree on the
  exact shape of the payload they exchange. If you touch one, audit the other.
- Tensors use the tuple schema from the reference host:
  `[(ValueType::Number path, shape), values]`. This keeps our transitional crate
  compatible with PyO3/HTTP adapters today and the full TinyChain host tomorrow.
- Reuse the path-prefix constants defined beside each class/type when constructing
  new URIs so `/state/...` segments stay canonical across adapters.

## Status & next steps

- Track open work in `ROADMAP.md`.
- Update this README whenever the txfs contract or bootstrap helpers evolve so
  future contributors do not have to reverse-engineer the layout.
- When integrating with other crates, keep the code surface minimal and reuse
  shared primitives (`tc-ir`, `tc-value`) rather than introducing new data
  representations.
