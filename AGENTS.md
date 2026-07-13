## tc-state Agent Notes

- Maintain `destream` parity: every `FromStream` in this crate must parse exactly
  what the corresponding `IntoStream` emits so adapters never guess at schema
  drift. When you tweak serialization, update both directions in the same PR.
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
- Do not construct `freqfs::Cache` in production `tc-state` module code.
  BTree decode paths must receive preloaded roots via bootstrap-owned context
  injection (`StateContext::with_btree_roots(...)`). Any cache construction in
  this crate must be limited to `#[cfg(test)]` helpers.
