# tc-state

`tc-state` owns TinyChain's universal native `State<Txn>` value. It combines
scalar IR and collection values behind one recursive routing, resolution, view,
and codec boundary without owning host storage or transports.

`State` and its terminal `StateView` are ordinary enums. Match or move their
variants directly; no detachable contents or custom destruction is required.
Cloning, conversion, and destruction use ordinary recursive ownership.

Typed decoding delegates maps, tuples, and object fields directly, with a shared
16-level structural allowance passed into IR, Value, and Class decoding. Boxed
field helpers and large child futures keep polling frames small; boxing alone
does not bound recursive depth. Native constructors and runtime
composition are trusted to respect this contract. Codec checks alone do not
bound program-generated nesting; violating the native contract can overflow
recursive operations or destruction.

Resolution and streaming projection retain their existing execution mechanisms.
Encoding delegates each view variant directly to typed codec APIs, consuming
map and tuple siblings lazily. Object fields use ordinary tagged maps and tuples.
Collection leaves retain their native guards and streams until consumption or
drop. Child contexts preserve tuple-index and map-key storage isolation; only
collection leaves request storage from those contexts.
Conversions into Value retain its checked tuple-depth contract.

## Responsibilities

- Symmetric format-neutral State encoding and decoding.
- Recursive resolution of `Scalar`, `TCRef`, operation parameters, control flow,
  and concrete `$self` values.
- Native routing for scalar values, tuples, maps, objects, Classes, and delegated
  collections and Chain-owned members.
- Canonical Class construction, digest verification, inheritance, cycle/depth
  validation, and declaring-Class method ownership.
- Recursive transaction-consistent view acquisition before adapter encoding.

`State::Chain` preserves the WAL boundary during native method execution. Its
terminal view delegates to the collection view; no additional wire envelope is
introduced. `BoundMethod` executes an existing `OpDef` with its native `$self`;
maps recursively delegate member routes without discarding a Chain owner.

Numeric handlers delegate arithmetic to `number-general`'s checked methods and
return ordinary request errors on failure. Tuple indices and slice bounds use
checked integral conversion; fractional, nonfinite, and out-of-range arguments
are rejected. Valid negative slice bounds keep their existing meaning. Tagged
numeric decoding delegates to `tc-value`'s checked conversion contract.

Value-only inspection and text conversion delegate to `tc-value`. Public `eq`
uses Value equality: Links and strings are distinct; numeric representations
use canonical numeric equality. Number inspection of State unwraps its embedded
Value and reuses the
canonical borrowed Value conversion. String template parameters borrow existing
strings during replacement; returned text remains owned. State retains request
validation, route selection, and traversal of universal tuples/maps.

Every application mutation of a persistent mutable collection must pass through
that Chain. Returned collection views are read-only; State does not unwrap their
mutable sources or restore mutation authority through identity transforms.
`route_scalar(&scalar, || subject)` selects application scalar handlers and
constructs the subject only for a method. Its operation-reference GET evaluates
the reference; native map member observations continue returning reference values.

Collection behavior belongs to `tc-collection`. `tc-ir` reports syntax-level
requirements but owns no execution plan; graph scheduling and `OpDef` execution
belong to `tc-server`. `Gateway` provides transaction-bound resource access, Class
lookup, and the inherited write-scope check. `OpExecutor: Gateway` adds only
server-owned `OpDef` execution. Both are trusted host interfaces; resource owners
still enforce permissions, and `check_write` is not complete authorization. The
declaring-Class origin is supplied only by canonical Class routing and is not
encodable in State or IR.

## Non-responsibilities

`tc-state` does not own filesystem roots, caches, `txfs` layouts, application
bootstrap, persistent named collections, media storage, HTTP, PyO3, or WASM
projection. Decoding receives only a delegated collection allocation context.

Tensor remains the single `tc-collection` Tensor variant exposed through State;
its future storage backend must not create another State variant or route family.

Borrowed `Number::opt_cast_from(&state)` recognizes only an embedded numeric
Value and copies its `Number`. `CollectionState::as_number` delegates to this
conversion; strings, containers, and other State variants are not coerced.

Value conversion accepts embedded Values, eligible Scalar tuples, and State
tuples recursively. All three share Value's tuple-depth allowance, checked before
conversion constructs containers. Other State variants remain rejected, including
bare `State::None` (an embedded `Value::None` remains valid). Structural tuples
become element sequences in `Vec<Value>` conversion; an embedded Value, including
a Value tuple, remains one sequence element. Leaf semantics belong to `tc-value`,
and Scalar eligibility belongs to `tc-ir`.

## Testing

```bash
cargo test --all-targets --all-features
```

Changes to recursive State or Class behavior require symmetric codec tests,
native routing tests, and mocked `OpExecutor` tests proving transaction,
deadline, and concrete-subject preservation.

Private storage fixtures return a directory owner alongside their transaction.
Retain that owner until decoded values, views, streams, and transaction handles
have been released, including when decoding consumes the transaction argument.

See [the crate invariants](AGENTS.md) and the collection
[transaction contract](https://github.com/TinyChain-Inc/tc-collection/blob/main/TRANSACTIONAL_COLLECTION_CONTRACT.md).

Unit-test implementations live under `tests/unit/`, linked from their owning
modules to preserve private access and test names. Shared test fixtures belong
under `tests/common/`; production types must not acquire fixture fields.

Static async delegation uses concrete futures. Box futures at object-safe or
heterogeneous ownership boundaries, to make recursive future types finite, or
where normal-stack measurements require smaller frames. Local pinning alone
does not reduce future size or recursive polling depth. Decoder field and tagged
payload boundaries retain their measured boxes; ordinary tuple delegation uses
the already-boxed recursive child decoder.
