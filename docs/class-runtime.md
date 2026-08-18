# Class runtime parity and contract status

This document inventories and tracks the v1 object behavior requested by
[`tc-state#9`](https://github.com/TinyChain-Inc/tc-state/issues/9). It is the
design checkpoint for the native runtime implementation. Only the wire-level
portions are intentionally deferred to the canonical fixtures owned by `tcv2#68` (see
[Contract dependency](#contract-dependency)).

## v1 → v2 parity table

| Concern | v1 behavior | v2 disposition | Implementation or acceptance test |
| --- | --- | --- | --- |
| Class parent | `InstanceClass::extends` is a `Link`; the implicit base is `/state/class` | **Preserve the capability, replace the representation.** Native and user-defined parents must be distinct canonical variants from `tcv2#68`; there is no implicit sentinel link. | Blocked: golden native-parent and class-parent fixtures |
| Prototype | Ordered `Map<Scalar>` | **Preserve.** Members remain deferred IR scalars/references; lookup must not evaluate the whole prototype. | `ClassDef::prototype`; no-projection lookup test |
| Class identity | Equality compares parent and prototype; hashing omits the implicit base parent | **Replace.** The authoritative manifest layer supplies a generic bound identity; definition equality remains structural and hashing uses that identity without duplicating manifest serialization. | `ClassDef`; identity hash test |
| Empty prototype coercion | An empty prototype can collapse to its parent `Link`/native `StateType` | **Drop.** A Class remains a Class, including with an empty prototype. | Blocked: empty-prototype golden fixture |
| Class encoding | A one-entry map from parent link to prototype | **Replace.** Consume the language-neutral `tcv2#68` fixture exactly; do not perpetuate the ambiguous map-key schema. | Blocked: malformed and golden codec tests |
| Instance data | `InstanceExt` retains boxed parent state, class, and member map | **Preserve.** `State::Object(Object::Instance)` owns concrete parent `State<Txn>`, immutable class definition/identity, and instance members. | Native object/state tests; wire round trip remains blocked |
| Parent coercion | Scalar/value conversions delegate through the boxed parent | **Preserve narrowly.** Canonical `State` conversions remain the only coercion vocabulary. | Planned conversion tests |
| Inheritance | Parent behavior is largely supplied by Rust `Deref`; user-class traversal is not explicit | **Replace.** Traverse explicit class parents with cycle detection and a fixed public maximum depth. | `ClassResolver`; nested inheritance, cycle, and depth tests |
| Member dispatch | v1 object/public routing combines prototype and native dispatch | **Replace.** One state-level resolver uses the order below; authorization and routing stay in the server. | `resolve_member`; precedence and missing-member tests |
| Bound methods | v1 public handlers arrange instance context | **Preserve behavior, replace mechanism.** Return a state-level bound member carrying `self`, never a transaction handle. | `ResolvedMember::BoundMethod`; bound-self test |
| Collection parents | Parent state may be a collection | **Preserve laziness.** Inspect class/member metadata only; never call `IntoView` or enumerate/materialize the collection during lookup. | Planned instrumented no-materialization test |
| Transaction lifecycle | `InstanceExt` carries a phantom transaction type and v1 public operations use a transaction | **Replace.** `State<Txn>` names the capability, but Class/instance values never mint, commit, retain, or expose a transaction handle. | Architecture source test |
| Instance hashing | v1 rejects instance hashing | **Defer unless `tcv2#68` specifies it.** Class identity must be stable; instance identity policy is not invented here. | Contract-dependent |
| Authorization | v1 dispatch lives near public routing | **Defer to `tc-server`.** Prototype membership grants no authority. | Architecture/source test |

`ClassDef` and `ClassInstance` live in the runtime implementation because they
contain native `State<Txn>` values and implement lookup. They are not detached
from universal state: `State::Object` carries the public `Object::{Class,
Instance}` sum type, and `State`/`Object` implement `tc_ir::Route`. A Class root
route constructs an instance; an Instance delegates concrete member routes and
native fallback to its parent. Prototype methods route as bound handlers through
the explicit `ClassExecutor` host capability. The handler supplies the complete
instance/self context and borrowed transaction, while `tc-state` never evaluates
the `OpDef` or assumes transaction lifecycle ownership.

## Member resolution contract

The state-level resolver will use this deterministic order once the canonical
identity schema is available:

1. instance members;
2. the concrete Class prototype;
3. each user-defined parent Class prototype, nearest parent first;
4. the declared native parent/type behavior;
5. the native behavior of the concrete parent state.

The first match wins. A method-like member is returned bound to the complete
instance as `self`; binding does not evaluate unrelated members and does not
expose `Txn`. Duplicate names are therefore ordinary, deterministic overrides.
An override whose member kind cannot be bound or invoked in the inherited slot
fails with a typed `UnsupportedOverride` error rather than silently changing
dispatch semantics.

Traversal must track visited class identities and enforce a public, fixed depth
limit. Repeated identity produces `InheritanceCycle`; exceeding the limit
produces `InheritanceDepthExceeded`. Malformed definitions, invalid parent
kinds, unsupported overrides, and absent names remain distinct typed failures.

## Intended public Rust interface for `tc-server#34`

The native runtime now provides:

- an immutable `ClassDef` exposing its canonical identity, parent, and
  prototype;
- a `ClassParent` enum distinguishing a native `StateType` parent from a
  user-defined Class identity/definition;
- a `ClassInstance<Txn, I>` retaining `State<Txn>` parent, `ClassDef<I>`, and
  `Map<State<Txn>>` members;
- a lookup operation returning `ResolvedMember`, including a bound method with
  instance `self` context;
- `tc_ir::Route` implementations plus the `ClassExecutor` capability used by
  the host/kernel to execute a routed bound `OpDef`; and
- a non-exhaustive typed `ClassError` covering malformed definitions, invalid
  parents, unsupported overrides, missing members, cycles, and depth exhaustion.

The generic identity parameter is intentionally supplied by the authoritative
manifest layer, so this interface does not commit to its digest shape or an encoding.
Those details must match the canonical fixtures before codec support is added.
The interface deliberately contains no server transport routing,
authorization, network, replication, persistence, or transaction-lifecycle
operation.

## Materialization and transaction ownership

Resolution is metadata-only. It walks instance maps and Class prototype maps
one member at a time and delegates native behavior without projecting the parent
through `IntoView`. In particular, a collection parent remains an opaque
`tc-collection` leaf during member lookup. Only a later, explicit terminal view
operation may materialize it under the host-provided transaction capability.

Instances parameterize their state with the host's explicit `Txn` capability,
as all `State<Txn>` values do, but do not store a second transaction handle or
perform lifecycle operations. The kernel remains the sole owner of acquisition,
commit, and rollback.

## Intentional v1 incompatibilities and deferred behavior

- The `/state/class` sentinel-parent convention and empty-Class-to-Link
  coercion will not be retained.
- Rust `Deref` is not an inheritance or dispatch mechanism in v2.
- Prototype membership never implies route authority.
- Instance hashing remains deferred until the shared contract specifies it.
- Server routing, persistence/replication, authorization, and language APIs are
  outside this state-level issue.

## Contract dependency

As of 2026-08-18, `tcv2#68` is explicitly blocked on its own prerequisites and
does not yet provide canonical Class fixtures. The narrow outstanding question is:
**what exact tagged
destream shapes and identity digest inputs distinguish a native parent, a
user-defined parent, a Class definition, and an instance?**

Implementing encode/decode or identity digest derivation before that answer
would invent a competing wire format, contrary to `tc-state#9`.
Consequently this crate implements the format-independent native runtime and
tests its inheritance, override, cycle, depth, bound-method, identity-hash, and
no-projection behavior. `destream` implementations plus golden, malformed, and
unknown-version fixtures remain deferred until the authoritative corpus exists.
