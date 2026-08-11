//! TinyChain state runtime primitives.
//!
//! BTree decode consumes transaction-local roots delegated by the host. This
//! crate does not construct `freqfs::Cache` or transaction directories.

use number_general::Number;
use safecast::TryCastFrom;
use tc_collection::table::Table;
use tc_ir::{Map, Scalar};
use tc_value::Value;

pub use crate::codec::{BTreeType, CollectionType, StateType, TableType, TensorType};
pub use tc_collection::tensor::{AxisRange, Range, Tensor, TensorReduceResult};
pub use tc_collection::Collection;
pub use tc_ir::{Class, NativeClass};

/// TinyChain runtime state.
#[derive(Clone, Debug)]
pub enum State<Txn> {
    None,
    Scalar(Scalar),
    Map(Map<State<Txn>>),
    Tuple(Vec<State<Txn>>),
    Collection(Collection<Txn>),
}

impl<Txn> State<Txn> {
    /// Lift a transaction-free IR scalar into its structural State form.
    pub fn from_scalar(scalar: Scalar) -> Self {
        match scalar {
            Scalar::Map(map) => State::Map(
                map.into_iter()
                    .map(|(id, value)| (id, Self::from_scalar(value)))
                    .collect(),
            ),
            Scalar::Tuple(items) => {
                State::Tuple(items.into_iter().map(Self::from_scalar).collect())
            }
            scalar => State::Scalar(scalar),
        }
    }

    pub fn is_none(&self) -> bool {
        match self {
            State::None => true,
            State::Scalar(Scalar::Value(Value::None)) => true,
            State::Tuple(items) => items.is_empty(),
            _ => false,
        }
    }
}

impl<Txn> Default for State<Txn> {
    fn default() -> Self {
        State::Scalar(Scalar::default())
    }
}

impl<Txn> From<Value> for State<Txn> {
    fn from(value: Value) -> Self {
        State::Scalar(Scalar::from(value))
    }
}

impl<Txn> From<Scalar> for State<Txn> {
    fn from(scalar: Scalar) -> Self {
        Self::from_scalar(scalar)
    }
}

impl<Txn> TryCastFrom<State<Txn>> for Value {
    fn can_cast_from(state: &State<Txn>) -> bool {
        matches!(state, State::Scalar(Scalar::Value(_)))
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Scalar(Scalar::Value(value)) => Some(value),
            _ => None,
        }
    }
}

impl<Txn> TryCastFrom<State<Txn>> for Vec<Value> {
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Tuple(items) => items.iter().all(Value::can_cast_from),
            State::Scalar(Scalar::Tuple(items)) => {
                items.iter().all(|item| matches!(item, Scalar::Value(_)))
            }
            state => Value::can_cast_from(state),
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Tuple(items) => items.into_iter().map(Value::opt_cast_from).collect(),
            State::Scalar(Scalar::Tuple(items)) => items
                .into_iter()
                .map(|item| Value::opt_cast_from(State::<Txn>::Scalar(item)))
                .collect(),
            state => Some(vec![Value::opt_cast_from(state)?]),
        }
    }
}

impl<Txn> TryCastFrom<State<Txn>> for Vec<State<Txn>> {
    fn can_cast_from(state: &State<Txn>) -> bool {
        matches!(state, State::Tuple(_) | State::Scalar(Scalar::Tuple(_)))
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Tuple(items) => Some(items),
            State::Scalar(Scalar::Tuple(items)) => {
                Some(items.into_iter().map(State::Scalar).collect())
            }
            _ => None,
        }
    }
}

impl<Txn> TryCastFrom<State<Txn>> for Map<State<Txn>> {
    fn can_cast_from(state: &State<Txn>) -> bool {
        matches!(state, State::Map(_) | State::Scalar(Scalar::Map(_)))
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

impl<Txn> TryCastFrom<State<Txn>> for Tensor {
    fn can_cast_from(state: &State<Txn>) -> bool {
        matches!(state, State::Collection(Collection::Tensor(_)))
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Collection(Collection::Tensor(tensor)) => Some(tensor),
            _ => None,
        }
    }
}

impl<Txn> From<Collection<Txn>> for State<Txn> {
    fn from(collection: Collection<Txn>) -> Self {
        State::Collection(collection)
    }
}

impl<Txn> From<Table<Txn>> for State<Txn> {
    fn from(table: Table<Txn>) -> Self {
        State::Collection(Collection::from(table))
    }
}

impl<Txn> TryCastFrom<State<Txn>> for Scalar {
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::None | State::Scalar(_) => true,
            State::Map(map) => map.values().all(Self::can_cast_from),
            State::Tuple(items) => items.iter().all(Self::can_cast_from),
            State::Collection(_) => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::None => Some(Scalar::Value(Value::None)),
            State::Scalar(scalar) => Some(scalar),
            State::Map(map) => map
                .into_iter()
                .map(|(key, value)| Self::opt_cast_from(value).map(|value| (key, value)))
                .collect::<Option<Map<_>>>()
                .map(Scalar::Map),
            State::Tuple(items) => items
                .into_iter()
                .map(Self::opt_cast_from)
                .collect::<Option<Vec<_>>>()
                .map(Scalar::Tuple),
            State::Collection(_) => None,
        }
    }
}

impl<Txn: tc_collection::StorageContext + 'static> tc_collection::CollectionState for State<Txn> {
    type Txn = Txn;

    fn none() -> Self {
        State::None
    }

    fn from_scalar(scalar: Scalar) -> Self {
        Self::from(scalar)
    }

    fn from_value(value: Value) -> Self {
        Self::from(value)
    }

    fn from_collection(collection: Collection<Txn>) -> Self {
        Self::from(collection)
    }

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

    fn into_tensor(self) -> tc_error::TCResult<Tensor> {
        Tensor::try_cast_from(self, |_| {
            tc_error::TCError::bad_request("expected tensor state")
        })
    }

    fn is_none(&self) -> bool {
        State::is_none(self)
    }
}

impl<Txn: tc_collection::StorageContext + 'static> tc_ir::StateInstance for State<Txn> {
    type Transaction = Txn;
}

impl<Txn> From<Number> for State<Txn> {
    fn from(number: Number) -> Self {
        State::from(Value::from(number))
    }
}

impl<Txn> From<u64> for State<Txn> {
    fn from(number: u64) -> Self {
        State::from(Number::from(number))
    }
}

#[cfg(test)]
mod tests;
