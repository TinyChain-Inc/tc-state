# Class and Instance Runtime Contract

`tc-state` owns Class validation and ordinary recursive instance behavior. A
Class is an immutable value; an instance is a self-contained `State`, not a
service, registry entry, or storage handle.

## Canonical values

- `ClassBody` contains only the declared parent and prototype.
- `ClassDef` adds an ordinary identity `Link` and the semantic digest derived
  from the immutable body.
- `Object::Class` and `Object::Instance` are carried by universal `State<Txn>`
  and have symmetric codecs.
- Decoding recomputes and validates the Class digest before exposing a value.

At an application boundary the host passes the literal key as an ordinary
identity `Link` and the bare body to `Class::from_definition`. The host resolves
authority and dependencies and persists the resulting definition. `tc-state`
owns no application directory, URI taxonomy, or storage.

## Member resolution

Instance routing performs one bounded lookup in this order:

1. instance members;
2. the concrete Class prototype;
3. user-defined parent prototypes, nearest first;
4. declared native-parent behavior;
5. behavior of the concrete parent state.

The first match wins. Ordinary values delegate through their own `Route<State>`;
an `OpDef` becomes a bound handler carrying the complete concrete instance as
`$self`. Missing native members return `NotFound` and never fall through to an
external transport.

Class validation tracks visited identities and enforces the public inheritance
depth bound. Invalid parents, invalid prototypes, cycles, depth exhaustion,
unsupported overrides, and digest mismatches remain distinct structured errors.
`validate_classes` validates a staged-plus-committed Class set in one memoized
traversal. Application dependency policy is derived and enforced by the host,
not by `tc-state`. An inherited method executes under the policy of the Class
which declared it.

## Runtime boundary

`StateExecutor` is the only host capability required by recursive State
execution. It provides canonical Class lookup, outbound dispatch, and the
callback for server-owned `OpDef` execution. The state layer preserves the
transaction, deadline, method call, declaring-Class origin, and concrete
subject; it does not aggregate application policy, schedule graphs, or choose
transaction outcomes.

Instances may be stored as ordinary collection values and participate in the
owning collection's lifecycle. Class storage owns definitions only. Neither a
Class nor an instance retains a transaction handle, filesystem path, cache,
instance ID, or registry membership.

## Verification

Changes require deterministic Class and instance round trips plus tests for
member precedence, bound `$self`, native-parent delegation, declaring-Class
ownership, malformed definitions, cycles, and depth limits. A mocked
`StateExecutor` must demonstrate that nested dispatch preserves transaction and
subject identity.
