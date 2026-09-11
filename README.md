# tc-state

`tc-state` owns TinyChain's universal native `State<Txn>` value. It combines
scalar IR and collection values behind one recursive routing, resolution, view,
and codec boundary without owning host storage or transports.

## Responsibilities

- Symmetric format-neutral State encoding and decoding.
- Recursive resolution of `Scalar`, `TCRef`, operation parameters, control flow,
  and concrete `$self` values.
- Native routing for scalar values, tuples, maps, objects, Classes, and delegated
  collections.
- Canonical Class construction, digest verification, inheritance, cycle/depth
  validation, and declaring-Class method ownership.
- Recursive transaction-consistent view acquisition before adapter encoding.

Collection behavior belongs to `tc-collection`. `tc-ir` reports syntax-level
requirements but owns no execution plan; graph scheduling and `OpDef` execution
belong to `tc-server`. `StateExecutor` is trusted in-process host SPI for external
dispatch, Class lookup, and server-owned `OpDef` execution. Its declaring-Class
origin is supplied only by canonical Class routing and is not encodable in State
or IR.

## Non-responsibilities

`tc-state` does not own filesystem roots, caches, `txfs` layouts, application
bootstrap, persistent named collections, media storage, HTTP, PyO3, or WASM
projection. Decoding receives only a delegated collection allocation context.

Tensor remains the single `tc-collection` Tensor variant exposed through State;
its future storage backend must not create another State variant or route family.

## Testing

```bash
cargo test --all-targets --all-features
```

Changes to recursive State or Class behavior require symmetric codec tests,
native routing tests, and mocked `StateExecutor` tests proving transaction,
deadline, and concrete-subject preservation.

See [the crate invariants](AGENTS.md). For non-normative integration context,
see the TinyChain
[workspace architecture](https://github.com/TinyChain-Inc/tcv2/blob/main/ARCHITECTURE.md)
and the collection
[transaction contract](https://github.com/TinyChain-Inc/tc-collection/blob/main/TRANSACTIONAL_COLLECTION_CONTRACT.md).
