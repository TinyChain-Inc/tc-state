## tc-state Agent Notes

- `tc-state` is the universal value, resolution, and native routing boundary,
  not a collection runtime, storage owner, or transport codec. Delegate collection
  behavior to `tc-collection` and transport projection to adapters; remove any
  duplicate helper instead of maintaining a compatibility path.
- Delegate recognized Value leaf inspection and behavior to `tc-value`.
  Keep universal State traversal, native route selection, request diagnostics,
  method binding, and transaction-dependent execution here.
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
- `Gateway` owns the trusted host interface for resource access, Class lookup,
  and the inherited write-scope check; resource owners enforce permissions.
  `OpExecutor: Gateway` adds only server-owned operation execution. Neither
  interface independently grants access. A declaring-Class origin is supplied only by canonical Class/member routing and
  is never decoded from State or IR. Host implementations remain responsible
  for deadlines, execution budgets, scheduling, and outbound authorization.
  Resolution checks the inherited write capability before mutating an existing
  receiver; pure calculations and fresh literal construction remain available.
- Every `State<Txn>` signature must name its transaction capability explicitly.
  Do not add a default transaction type or use `State<()>` as a general runtime value;
  transaction-free scalar structure belongs in `tc_ir::Scalar`, not a partially
  instantiated universal State.
- `IntoView` is a native acquisition contract, not an encoding trait. Keep the
  recursive transaction-consistent view walk in `view`, and keep `IntoStream`
  implementations in `codec`; neither module may reach into the other's owner.
- Delegate the logical Tensor wire schema and streams to `tc-collection`'s codecs.
  State and adapters must not maintain a separate schema or dense-only decoder.
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
  Literal and transaction-local BTree/Table values receive delegated allocation
  children. Persistent named members retain their Chain owner in State; State
  does not host standalone collections. Any cache construction in this crate must be limited to
  `#[cfg(test)]` helpers.
- Every application mutation of persistent mutable collection state targets its
  owning Chain. Returned collection views are read-only, including full reads
  and identity transforms. State must preserve Chain ownership when routing
  members; it must not unwrap a subject to execute a write. Selection writes
  carry their bounds to the original owner through ordinary PUT/DELETE.
- State and collection leaves use only allocation contexts delegated by their
  caller. They do not know or select a host data or workspace root.

- Match Value, Scalar, TCRef, State, and StateView variants directly. Codecs
  enforce 16 structural levels; trusted native composition must respect that
  bound. Conversions into Value check its 64-tuple limit, including embedded
  Values, before construction. Keep execution stack safety separate from data
  ownership.

- Value extraction accepts nested State/Scalar tuples through the canonical
  conversions. Preflight their combined depth with embedded Values before
  construction; do not reset the allowance at a layer boundary. Preserve
  structural-tuple versus embedded-Value sequence shapes.
