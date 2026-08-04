//! TinyChain state runtime primitives.
//!
//! BTree decode is bootstrap-wired: production code injects preloaded
//! `freqfs::DirLock` roots into `StateContext::with_state_roots(...)` and decode
//! paths consume those handles. This crate does not construct `freqfs::Cache` in
//! production runtime code.

use std::{ops::Bound, str::FromStr, sync::Arc};

use destream::{
    IntoStream,
    en::{self, EncodeMap, Error as _},
};
use number_general::Number;
use pathlink::Link;
use safecast::TryCastFrom;
use tc_collection::{
    btree::{BTree, BTreeColumnSchema, BTreeDecodeContext, PersistentFile},
    table::Table,
};
use tc_ir::{Claim, Map, NetworkTime, Scalar, Transaction, TxnId};
use tc_value::Value;

mod ops;

pub use ops::StateStream;

pub use crate::codec::{BTreeType, CollectionType, StateType, TableType, TensorType};
pub use tc_collection::tensor::{AxisRange, Range, Tensor, TensorReduceResult};
pub use tc_ir::{Class, NativeClass};

/// Collection state.
#[derive(Clone, Debug)]
pub enum Collection {
    /// In-memory BTree data with an explicit column schema.
    BTree(Box<BTreeCollection>),
    /// A relational table or a lazy table view.
    Table(Box<TableCollection>),
    /// In-memory Tensor data.
    Tensor(Tensor),
}

#[derive(Clone, Debug)]
pub struct BTreeCollection {
    pub schema: Vec<BTreeColumnSchema>,
    pub btree: BTree,
    pub bounds: (Bound<Value>, Bound<Value>),
    pub reverse: bool,
}

/// A Table together with the snapshot used to produce a routed view.
#[derive(Clone, Debug)]
pub struct TableCollection {
    pub table: Table,
    pub txn_id: Option<TxnId>,
}

impl TableCollection {
    pub fn new(table: Table) -> Self {
        Self {
            table,
            txn_id: None,
        }
    }

    pub fn with_txn(mut self, txn_id: TxnId) -> Self {
        self.txn_id = Some(txn_id);
        self
    }
}

impl BTreeCollection {
    pub fn with_schema(schema: Vec<BTreeColumnSchema>, btree: BTree) -> Self {
        Self {
            schema,
            btree,
            bounds: (Bound::Unbounded, Bound::Unbounded),
            reverse: false,
        }
    }

    pub fn slice(&self, bounds: (Bound<Value>, Bound<Value>), reverse: bool) -> Self {
        Self {
            schema: self.schema.clone(),
            btree: self.btree.clone(),
            bounds: (
                max_lower_bound(self.bounds.0.clone(), bounds.0),
                min_upper_bound(self.bounds.1.clone(), bounds.1),
            ),
            reverse,
        }
    }

    pub async fn finalized_key_stream(&self) -> std::io::Result<b_tree::Keys<Value>> {
        self.btree
            .finalized_key_stream_in(self.bounds.clone(), self.reverse)
            .await
    }
}

impl From<BTreeCollection> for Collection {
    fn from(btree: BTreeCollection) -> Self {
        Self::BTree(Box::new(btree))
    }
}

impl From<Table> for Collection {
    fn from(table: Table) -> Self {
        Self::Table(Box::new(TableCollection::new(table)))
    }
}

fn max_lower_bound(left: Bound<Value>, right: Bound<Value>) -> Bound<Value> {
    match (left, right) {
        (Bound::Unbounded, bound) | (bound, Bound::Unbounded) => bound,
        (Bound::Included(left), Bound::Included(right)) => {
            if left >= right {
                Bound::Included(left)
            } else {
                Bound::Included(right)
            }
        }
        (Bound::Included(left), Bound::Excluded(right)) => {
            if left > right {
                Bound::Included(left)
            } else {
                Bound::Excluded(right)
            }
        }
        (Bound::Excluded(left), Bound::Included(right)) => {
            if left < right {
                Bound::Included(right)
            } else {
                Bound::Excluded(left)
            }
        }
        (Bound::Excluded(left), Bound::Excluded(right)) => {
            if left >= right {
                Bound::Excluded(left)
            } else {
                Bound::Excluded(right)
            }
        }
    }
}

fn min_upper_bound(left: Bound<Value>, right: Bound<Value>) -> Bound<Value> {
    match (left, right) {
        (Bound::Unbounded, bound) | (bound, Bound::Unbounded) => bound,
        (Bound::Included(left), Bound::Included(right)) => {
            if left <= right {
                Bound::Included(left)
            } else {
                Bound::Included(right)
            }
        }
        (Bound::Included(left), Bound::Excluded(right)) => {
            if left < right {
                Bound::Included(left)
            } else {
                Bound::Excluded(right)
            }
        }
        (Bound::Excluded(left), Bound::Included(right)) => {
            if left <= right {
                Bound::Excluded(left)
            } else {
                Bound::Included(right)
            }
        }
        (Bound::Excluded(left), Bound::Excluded(right)) => {
            if left <= right {
                Bound::Excluded(left)
            } else {
                Bound::Excluded(right)
            }
        }
    }
}

impl<'en> en::IntoStream<'en> for Collection {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let mut map = encoder.encode_map(Some(1))?;
        match self {
            Collection::BTree(_) => {
                return Err(E::Error::custom(
                    "BTree literal encoding is transport-specific and must be encoded at the boundary layer",
                ));
            }
            Collection::Table(_) => {
                return Err(E::Error::custom(
                    "Table literal encoding is transport-specific and must be encoded at the boundary layer",
                ));
            }
            Collection::Tensor(tensor) => {
                let tensor_path = TensorType.path().to_string();
                map.encode_entry(tensor_path, tensor)?;
            }
        }
        map.end()
    }
}

impl<'en> en::ToStream<'en> for Collection {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        self.clone().into_stream(encoder)
    }
}

impl TryCastFrom<Collection> for Tensor {
    fn can_cast_from(collection: &Collection) -> bool {
        matches!(collection, Collection::Tensor(_))
    }

    fn opt_cast_from(collection: Collection) -> Option<Self> {
        match collection {
            Collection::Tensor(tensor) => Some(tensor),
            Collection::BTree(_) | Collection::Table(_) => None,
        }
    }
}

#[derive(Clone)]
struct NullTransaction {
    id: TxnId,
    claim: Claim,
}

impl Default for NullTransaction {
    fn default() -> Self {
        let id = TxnId::from_parts(NetworkTime::from_nanos(0), 0);
        let claim = Claim::new(
            Link::from_str("/lib/default").expect("default claim link"),
            umask::Mode::all(),
        );
        Self { id, claim }
    }
}

impl Transaction for NullTransaction {
    fn id(&self) -> TxnId {
        self.id
    }

    fn timestamp(&self) -> NetworkTime {
        self.id.timestamp()
    }

    fn claim(&self) -> &Claim {
        &self.claim
    }
}

/// Return a placeholder transaction context for decoding state without a transaction.
pub fn null_transaction() -> Arc<dyn Transaction> {
    Arc::new(NullTransaction::default())
}

#[derive(Clone)]
pub struct StateContext {
    btree_roots: Option<(
        freqfs::DirLock<PersistentFile>,
        freqfs::DirLock<PersistentFile>,
    )>,
}

impl StateContext {
    pub fn new(_transaction: Arc<dyn Transaction>) -> Self {
        Self { btree_roots: None }
    }

    pub fn with_state_roots(
        mut self,
        persistent_dir: freqfs::DirLock<PersistentFile>,
        txn_root: freqfs::DirLock<PersistentFile>,
    ) -> Self {
        self.btree_roots = Some((persistent_dir, txn_root));
        self
    }

    pub(super) fn state_decode_context(&self) -> Result<BTreeDecodeContext, String> {
        let (persistent_dir, txn_root) = self.btree_roots.clone().ok_or_else(|| {
            "BTree decode requires StateContext::with_state_roots(...) at bootstrap".to_string()
        })?;

        Ok(BTreeDecodeContext::new(persistent_dir, txn_root))
    }
}

impl From<Arc<dyn Transaction>> for StateContext {
    fn from(transaction: Arc<dyn Transaction>) -> Self {
        Self::new(transaction)
    }
}

pub fn state_context(transaction: Arc<dyn Transaction>) -> StateContext {
    StateContext::new(transaction)
}

/// TinyChain runtime state.
#[derive(Clone, Debug)]
pub enum State {
    None,
    Scalar(Scalar),
    Map(Map<State>),
    Tuple(Vec<State>),
    Collection(Collection),
}

impl State {
    pub fn is_none(&self) -> bool {
        match self {
            State::None => true,
            State::Scalar(Scalar::Value(Value::None)) => true,
            State::Tuple(items) => items.is_empty(),
            _ => false,
        }
    }
}

impl Default for State {
    fn default() -> Self {
        State::Scalar(Scalar::default())
    }
}

impl From<Value> for State {
    fn from(value: Value) -> Self {
        State::Scalar(Scalar::from(value))
    }
}

impl TryCastFrom<State> for Value {
    fn can_cast_from(state: &State) -> bool {
        matches!(state, State::Scalar(Scalar::Value(_)))
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Scalar(Scalar::Value(value)) => Some(value),
            _ => None,
        }
    }
}

impl TryCastFrom<State> for Vec<Value> {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Tuple(items) => items.iter().all(Value::can_cast_from),
            State::Scalar(Scalar::Tuple(items)) => {
                items.iter().all(|item| matches!(item, Scalar::Value(_)))
            }
            state => Value::can_cast_from(state),
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Tuple(items) => items.into_iter().map(Value::opt_cast_from).collect(),
            State::Scalar(Scalar::Tuple(items)) => items
                .into_iter()
                .map(|item| Value::opt_cast_from(State::Scalar(item)))
                .collect(),
            state => Some(vec![Value::opt_cast_from(state)?]),
        }
    }
}

impl From<Collection> for State {
    fn from(collection: Collection) -> Self {
        State::Collection(collection)
    }
}

impl From<Table> for State {
    fn from(table: Table) -> Self {
        State::Collection(Collection::from(table))
    }
}

impl TryCastFrom<State> for Scalar {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::None | State::Scalar(_) => true,
            State::Map(map) => map.values().all(Self::can_cast_from),
            State::Tuple(items) => items.iter().all(Self::can_cast_from),
            State::Collection(_) => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
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

impl<'en> en::IntoStream<'en> for State {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            State::None => encoder.encode_unit(),
            State::Scalar(scalar) => scalar.into_stream(encoder),
            State::Map(map) => map.into_stream(encoder),
            State::Tuple(items) => items.into_stream(encoder),
            State::Collection(collection) => collection.into_stream(encoder),
        }
    }
}

impl From<Number> for State {
    fn from(number: Number) -> Self {
        State::from(Value::from(number))
    }
}

impl From<u64> for State {
    fn from(number: u64) -> Self {
        State::from(Number::from(number))
    }
}

#[cfg(test)]
mod tests;
