//! Universal State values and their canonical conversions.

use number_general::Number;
use safecast::TryCastFrom;
use tc_collection::Collection;
use tc_collection::table::Table;
use tc_collection::tensor::Tensor;
use tc_ir::{Map, Scalar};
use tc_value::Value;

use crate::Object;

/// TinyChain runtime state with ordinary recursive ownership.
/// Native construction must respect the codec's structural nesting contract.
#[derive(Clone)]
pub enum State<Txn: tc_collection::StorageContext> {
    None,
    Scalar(Scalar),
    Map(Map<State<Txn>>),
    Tuple(Vec<State<Txn>>),
    Collection(Collection<Txn>),
    Chain(tc_chain::SyncChain<Txn, Txn::File>),
    Object(Box<Object<Txn>>),
}

impl<Txn: tc_collection::StorageContext + std::fmt::Debug> std::fmt::Debug for State<Txn> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => f.write_str("None"),
            Self::Scalar(value) => f.debug_tuple("Scalar").field(value).finish(),
            Self::Map(value) => write!(f, "Map({:?})", **value),
            Self::Tuple(value) => f.debug_tuple("Tuple").field(value).finish(),
            Self::Collection(value) => f.debug_tuple("Collection").field(value).finish(),
            Self::Chain(value) => f.debug_tuple("Chain").field(value).finish(),
            Self::Object(value) => match value.as_ref() {
                Object::Class(class) => write!(f, "Object(Class({class:?}))"),
                Object::Instance(instance) => write!(
                    f,
                    "Object(Instance {{ parent: {:?}, class: {:?}, members: Map({:?}) }})",
                    instance.parent(),
                    instance.class(),
                    **instance.members()
                ),
            },
        }
    }
}

impl<Txn: tc_collection::StorageContext> State<Txn> {
    /// Check native composition against the codec's structural allowance.
    /// Collections and Chains retain their separately bounded native owners.
    pub fn within_depth(&self, remaining: usize) -> bool {
        match self {
            Self::None | Self::Collection(_) | Self::Chain(_) => true,
            Self::Scalar(value) => value.within_depth(remaining),
            Self::Map(values) => remaining
                .checked_sub(1)
                .is_some_and(|n| values.values().all(|value| value.within_depth(n))),
            Self::Tuple(values) => remaining
                .checked_sub(1)
                .is_some_and(|n| values.iter().all(|value| value.within_depth(n))),
            Self::Object(object) => {
                remaining
                    .checked_sub(1)
                    .is_some_and(|n| match object.as_ref() {
                        Object::Class(class) => class.within_depth(n),
                        Object::Instance(instance) => {
                            instance.parent().within_depth(n)
                                && instance.class().within_depth(n)
                                && n.checked_sub(1).is_some_and(|n| {
                                    instance
                                        .members()
                                        .values()
                                        .all(|value| value.within_depth(n))
                                })
                        }
                    })
            }
        }
    }

    fn within_value_depth(&self, remaining: usize) -> bool {
        match self {
            Self::Scalar(scalar) => scalar.within_value_depth(remaining),
            Self::Tuple(items) => {
                remaining > 0
                    && items
                        .iter()
                        .all(|item| item.within_value_depth(remaining - 1))
            }
            _ => false,
        }
    }

    pub fn is_none(&self) -> bool {
        match &self {
            State::None => true,
            State::Scalar(scalar) => {
                matches!(&scalar, tc_ir::Scalar::Value(value) if matches!(&value, tc_value::Value::None))
            }
            State::Tuple(items) => items.is_empty(),
            _ => false,
        }
    }
}

impl<Txn: tc_collection::StorageContext> Default for State<Txn> {
    fn default() -> Self {
        State::Scalar(Scalar::default())
    }
}

impl<Txn: tc_collection::StorageContext> From<Value> for State<Txn> {
    fn from(value: Value) -> Self {
        State::Scalar(Scalar::from(value))
    }
}

impl<Txn: tc_collection::StorageContext> From<Scalar> for State<Txn> {
    /// Lift an IR scalar into native structural State through its container variants.
    fn from(scalar: Scalar) -> Self {
        match scalar {
            Scalar::Tuple(items) => Self::Tuple(items.into_iter().map(Self::from).collect()),
            Scalar::Map(items) => Self::Map(
                items
                    .into_iter()
                    .map(|(id, value)| (id, Self::from(value)))
                    .collect(),
            ),
            scalar => Self::Scalar(scalar),
        }
    }
}

impl<Txn: tc_collection::StorageContext> From<Map<State<Txn>>> for State<Txn> {
    fn from(map: Map<State<Txn>>) -> Self {
        State::Map(map)
    }
}

impl<Txn: tc_collection::StorageContext> From<Vec<State<Txn>>> for State<Txn> {
    fn from(tuple: Vec<State<Txn>>) -> Self {
        State::Tuple(tuple)
    }
}

impl<Txn: tc_collection::StorageContext> From<Map<bool>> for State<Txn> {
    fn from(map: Map<bool>) -> Self {
        State::Map(
            map.into_iter()
                .map(|(name, value)| (name, Self::from(Number::from(value))))
                .collect(),
        )
    }
}

// Inspection recognizes the existing numeric variant without coercion or ownership copies.
impl<Txn: tc_collection::StorageContext> TryCastFrom<&State<Txn>> for Number {
    fn can_cast_from(state: &&State<Txn>) -> bool {
        Self::opt_cast_from(*state).is_some()
    }

    fn opt_cast_from(state: &State<Txn>) -> Option<Self> {
        match state {
            State::Scalar(Scalar::Value(value)) => Self::opt_cast_from(value),
            _ => None,
        }
    }
}

impl<Txn: tc_collection::StorageContext> TryCastFrom<State<Txn>> for Value {
    fn can_cast_from(state: &State<Txn>) -> bool {
        state.within_value_depth(tc_value::MAX_VALUE_DEPTH)
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        if !Self::can_cast_from(&state) {
            return None;
        }

        state_into_value(state)
    }
}

// The complete State/Scalar/Value shape has already passed the shared depth checks.
fn state_into_value<Txn: tc_collection::StorageContext>(state: State<Txn>) -> Option<Value> {
    match state {
        State::Scalar(scalar) => Value::opt_cast_from(scalar),
        State::Tuple(items) => items
            .into_iter()
            .map(state_into_value)
            .collect::<Option<Vec<_>>>()
            .map(Value::Tuple),
        _ => None,
    }
}

impl<Txn: tc_collection::StorageContext> TryCastFrom<State<Txn>> for Vec<Value> {
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Tuple(items) => items.iter().all(Value::can_cast_from),
            State::Scalar(Scalar::Tuple(items)) => items.iter().all(Value::can_cast_from),
            _ => Value::can_cast_from(state),
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Tuple(items) => items.into_iter().map(Value::opt_cast_from).collect(),
            State::Scalar(Scalar::Tuple(items)) => {
                items.into_iter().map(Value::opt_cast_from).collect()
            }
            state => Value::opt_cast_from(state).map(|value| vec![value]),
        }
    }
}

impl<Txn: tc_collection::StorageContext> TryCastFrom<State<Txn>> for Vec<State<Txn>> {
    fn can_cast_from(state: &State<Txn>) -> bool {
        match &state {
            State::Tuple(_) => true,
            State::Scalar(scalar) => match &scalar {
                tc_ir::Scalar::Tuple(_) => true,
                tc_ir::Scalar::Value(value) => {
                    matches!(&value, tc_value::Value::Tuple(_))
                }
                _ => false,
            },
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Tuple(items) => Some(items),
            State::Scalar(scalar) => match scalar {
                tc_ir::Scalar::Tuple(items) => Some(
                    items
                        .into_iter()
                        .map(|scalar| State::Scalar(scalar))
                        .collect(),
                ),
                tc_ir::Scalar::Value(tc_value::Value::Tuple(items)) => {
                    Some(items.into_iter().map(State::from).collect())
                }
                _ => None,
            },
            _ => None,
        }
    }
}

impl<Txn: tc_collection::StorageContext> TryCastFrom<State<Txn>> for Map<State<Txn>> {
    fn can_cast_from(state: &State<Txn>) -> bool {
        matches!(&state, State::Map(_))
            || matches!(&state, State::Scalar(scalar) if matches!(&scalar, tc_ir::Scalar::Map(_)))
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Map(map) => Some(map),
            State::Scalar(Scalar::Map(map)) => Some(
                map.into_iter()
                    .map(|(id, scalar)| (id, State::Scalar(scalar)))
                    .collect(),
            ),
            _ => None,
        }
    }
}

impl<Txn: tc_collection::StorageContext> TryCastFrom<State<Txn>> for Tensor<Txn> {
    fn can_cast_from(state: &State<Txn>) -> bool {
        matches!(&state, State::Collection(Collection::Tensor(_)))
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Collection(Collection::Tensor(tensor)) => Some(tensor),
            _ => None,
        }
    }
}

impl<Txn: tc_collection::StorageContext> From<Collection<Txn>> for State<Txn> {
    fn from(collection: Collection<Txn>) -> Self {
        State::Collection(collection)
    }
}

impl<Txn: tc_collection::StorageContext> From<tc_chain::SyncChain<Txn, Txn::File>> for State<Txn> {
    fn from(chain: tc_chain::SyncChain<Txn, Txn::File>) -> Self {
        State::Chain(chain)
    }
}

impl<Txn: tc_collection::StorageContext> TryCastFrom<State<Txn>> for Collection<Txn> {
    fn can_cast_from(state: &State<Txn>) -> bool {
        matches!(&state, State::Collection(_))
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Collection(collection) => Some(collection),
            _ => None,
        }
    }
}

impl<Txn: tc_collection::StorageContext> From<Object<Txn>> for State<Txn> {
    fn from(object: Object<Txn>) -> Self {
        State::Object(Box::new(object))
    }
}

impl<Txn: tc_collection::StorageContext> From<Table<Txn>> for State<Txn> {
    fn from(table: Table<Txn>) -> Self {
        Self::from(Collection::from(table))
    }
}

impl<Txn: tc_collection::StorageContext> TryCastFrom<State<Txn>> for Scalar {
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::None | State::Scalar(_) => true,
            State::Tuple(items) => items.iter().all(Self::can_cast_from),
            State::Map(items) => items.values().all(Self::can_cast_from),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::None => Some(Self::default()),
            State::Scalar(scalar) => Some(scalar),
            State::Tuple(items) => items
                .into_iter()
                .map(Self::opt_cast_from)
                .collect::<Option<Vec<_>>>()
                .map(Self::Tuple),
            State::Map(items) => items
                .into_iter()
                .map(|(id, value)| Self::opt_cast_from(value).map(|value| (id, value)))
                .collect::<Option<Map<_>>>()
                .map(Self::Map),
            _ => None,
        }
    }
}

impl<Txn: tc_collection::StorageContext + 'static> tc_collection::CollectionState for State<Txn> {
    type Txn = Txn;

    fn into_scalar(self) -> tc_error::TCResult<Scalar> {
        Scalar::try_cast_from(self, |_| {
            tc_error::TCError::bad_request("expected scalar state")
        })
    }

    fn into_value(self) -> tc_error::TCResult<Value> {
        Value::try_cast_from(self, |_| {
            tc_error::TCError::bad_request("expected scalar value")
        })
    }

    fn into_tuple(self) -> tc_error::TCResult<Vec<Self>> {
        Vec::try_cast_from(self, |_| {
            tc_error::TCError::bad_request("expected tuple state")
        })
    }

    fn into_map(self) -> tc_error::TCResult<Map<Self>> {
        Map::try_cast_from(self, |_| {
            tc_error::TCError::bad_request("expected map state")
        })
    }

    fn into_tensor(self) -> tc_error::TCResult<Tensor<Txn>> {
        Tensor::try_cast_from(self, |_| {
            tc_error::TCError::bad_request("expected tensor state")
        })
    }

    fn as_number(&self) -> Option<Number> {
        Number::opt_cast_from(self)
    }

    fn is_none(&self) -> bool {
        State::is_none(self)
    }
}

impl<Txn: tc_collection::StorageContext + 'static> tc_ir::StateInstance for State<Txn> {
    type Transaction = Txn;
}

impl<Txn: tc_collection::StorageContext> From<Number> for State<Txn> {
    fn from(number: Number) -> Self {
        State::from(Value::from(number))
    }
}

impl<Txn: tc_collection::StorageContext> From<u64> for State<Txn> {
    fn from(number: u64) -> Self {
        State::from(Number::from(number))
    }
}

#[cfg(test)]
#[path = "../tests/unit/state/tests.rs"]
pub(crate) mod tests;
