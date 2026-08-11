## tc-state Agent Notes

- `tc-state` is the universal value and conversion boundary, not a collection
  runtime, router, storage owner, or transport codec. Delegate collection
  behavior to `tc-collection` and transport projection to adapters; remove any
  duplicate helper instead of maintaining a compatibility path.
- Maintain `destream` parity: every `FromStream` in this crate must parse exactly
  what the corresponding `IntoStream` emits so adapters never guess at schema
  drift. When you tweak serialization, update both directions in the same PR.
- `tc-state` owns the one recursive traversal of universal state structure.
  Recurse through maps, tuples, and scalar forms here; delegate a recognized
  collection leaf to `tc-collection`. Do not add collection-type dispatch,
  JSON re-encoding, or a second structural decoder in kernel or adapters.
- `From` and `TryCastFrom` implementations on `State` are the canonical
  construction and extraction paths. Cross-crate capability traits such as
  `CollectionState` must delegate to these conversions rather than repeat State
  variant matching or define a parallel coercion vocabulary.
- `State` is the native routing and graph-execution value. It does not implement
  transaction-dependent serialization directly. Its recursive `IntoView`
  implementation is the sole terminal projection walk and delegates collection
  leaves to `tc-collection`; local routes and `OpDef`s exchange `State` without
  constructing a view.
- Every `State<Txn>` signature must name its transaction capability explicitly.
  Do not add a default transaction type or use `State<()>` as a general runtime value;
  transaction-free scalar structure belongs in `tc_ir::Scalar`, not a partially
  instantiated universal State.
- `IntoView` is a native acquisition contract, not an encoding trait. Keep the
  recursive transaction-consistent view walk in `view`, and keep `IntoStream`
  implementations in `codec`; neither module may reach into the other's owner.
- Tensor payloads follow the canonical tuple schema from
  `tinychain/host/collection/tensor`: the encoded form is
  `[(ValueType::Number path, shape), values]`. Do not fall back to map-based
  payloads—PyO3, HTTP, and future adapters all rely on this tuple to round-trip
  dense tensors without special cases.
- Define and compare class paths with `Label`/`PathLabel` segments beside the
  types they describe. Avoid string-based helpers (e.g., `is_tensor_path`) so
  every caller enforces the same TinyChain `Id` validation rules.
- Encode/decode state exclusively with `destream` unless a protocol forces a
  tiny, bounded payload (e.g., query parameters). If you must reach for `serde`
  in those edge cases, document the reason inline.
- When encoding or decoding collection state, keep BTree payload handling
  stream-first end-to-end. Do not call APIs that materialize full key vectors
  (`Vec`) in production serialization/deserialization paths.
- Recursive state decoding and `IntoView` projection must preserve pull-based
  backpressure. Recurse one item at a time or within an explicit finite window;
  never turn an unknown-length map, tuple, collection, or body into an eager
  intermediate buffer merely to simplify traversal.
- Do not construct `freqfs::Cache` in production `tc-state` module code. State
  decoding receives only a kernel-delegated collection allocation context; it does
  not receive filesystem paths, construct roots, or retain a transaction ID/handle.
  Named collections are URI-derived by host workspace delegation and literal BTree/
  Table values receive unique transaction children. Any cache construction in this
  crate must be limited to `#[cfg(test)]` helpers.
- State and collection leaves use only directories delegated from the host's
  workspace cache. The separate `data_dir` cache belongs exclusively to the
  server's library/artifact store; see [`../docs/storage.md`](../docs/storage.md).
