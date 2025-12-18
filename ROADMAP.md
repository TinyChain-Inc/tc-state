# tc-state Roadmap

This roadmap tracks the evolution of the TinyChain state subsystem—collections,
media storage primitives, and the reference implementation that migrates v1
state semantics to the new control-plane model. The work ties directly into
`tc-server` (kernel, installers) and the client roadmaps (`client/py`, `client/js`).

---

## Phase 1 – Control-plane readiness

Goal: make the reference state implementation compatible with the control-plane
bootstrap (minimal control/auth blocks, authorized publishers config).

1. **Txfs layout audit.** Ensure all state paths (collections, `/state/media/...`,
   `/lib`, etc.) mirror URI segments exactly. Document the invariant in
   `tc-state/README.md` and enforce it via integration tests.
2. **Control block ingestion.** Teach the state subsystem to load the
   `control_plane.toml` metadata (publisher IDs, capability scopes) from disk so
   new installs honor control-plane policies even before the ledger is online.
3. **Authorized publishers seed.** Provide helper tooling (`tc-state/bin/seed_registry.rs`)
   that reads `authorized_publishers.toml`, validates key references, and primes
   the state store for `tc-server` to consume.
4. **CI cross-reference.** Add tests that run `tc-server` with the seeded state
   store, install a WASM library, and confirm state snapshots match expectations.
   Cross-link these tests from the `tc-server` and control-plane roadmaps.

## Phase 2 – Media storage primitives (recorded media)

Goal: add block storage for images/audio/video backed by state collections.

1. **Recorded media schema.** Define state schemas for immutable media objects
   (images, audio tracks, videos) stored under `/state/media/...` with metadata
   rows (content-type, duration, checksums, temporal index offsets, custody
   tags). Temporal addressing (e.g., “give me frame at 1:23”) is supported via
   auxiliary index tables, not separate `Chain`s.
2. **Chunked storage engine.** Implement a chunk manager that stores fixed-size
   blocks under txfs, deduplicates via checksums, and exposes a simple trait for
   `tc-server` handlers.
3. **Metadata APIs.** Add state routes (`/state/media/...`) that allow listing,
   fetching metadata, and retrieving individual chunks, all guarded by capability
   tokens.
4. **Client helpers.** Coordinate with `client/py` and `client/js` to expose
   `media` helper methods that call these routes (see their roadmaps for details).
5. **Metrics hooks.** Emit MetricsChain counters (bytes stored, bytes fetched,
   temporal-index lookups) whenever state operations touch media collections;
   record custody events in BlockChain as specified in the control-plane roadmap.

## Phase 3 – Time-dependent access patterns

Goal: add a clear, unitary API for time-dependent access (progressive reads,
temporal offsets) while `tc-server` owns the actual transport.

1. **Temporal cursor API.** Define state traits/helpers for “seek to offset,”
   “iterate forward/backward,” and “bounded window” operations on recorded media.
   These APIs return chunk IDs / byte ranges without assuming a transport.
2. **Queue-friendly ingest.** Provide utilities for recording progressive uploads
   into state while tracking temporal indexes so `tc-server` can resume ingestion
   after queue retries.
3. **Server handoff.** Document how `tc-server` (guarded by a `media` feature flag)
   consumes the temporal cursor API to implement HTTP range responses or
   WebSocket/WebTransport chunking. `tc-state` must not implement transports;
   it only exposes the access patterns.
4. **Client coordination.** Ensure the temporal APIs align with the helper
   methods planned in `client/py` and `client/js` (e.g., `media.stream(offset=...)`).
5. **Testing.** Add unit/integration tests that exercise temporal seeks, range
   calculations, and queue-resumed ingestion without involving network transports.
