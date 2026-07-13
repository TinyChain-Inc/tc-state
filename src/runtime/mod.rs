//! Transitional TinyChain state primitives.
//!
//! This crate exposes the placeholder collection and scalar state enums used by
//! adapters that need to exchange TinyChain values before the full transactional
//! filesystem (`freqfs`) + `Chain` + `Service` stack lands. The in-memory tensor
//! representation keeps downstream crates unblocked while we finish the shared
//! persistence layer.

use std::{str::FromStr, sync::Arc};

use destream::{
    de,
    en::{self, EncodeMap, EncodeSeq, Error as _},
    IntoStream,
};
use ha_ndarray::{ArrayBuf, Buffer, NDArray, NDArrayRead};
use number_general::{FloatType, Number, UIntType};
use pathlink::Link;
use safecast::TryCastFrom;
use tc_collection::btree::{BTree, BTreeColumnSchema, BTreeDecodeContext, PersistentFile};
use tc_ir::{Claim, Map, NetworkTime, Scalar, Transaction, TxnId};
use tc_value::{number_type_path, NumberType, Value};

mod tensor;
mod wire;

use wire::{coerce_shape, tensor_dtype_from_wire, tensor_from_parts};

pub use crate::codec::{BTreeType, CollectionType, StateType, TensorType};
pub use ha_ndarray::{AxisRange, Range};
pub use tc_ir::{Class, NativeClass};

/// Temporary tensor representation (in-memory only).
#[derive(Clone, Debug)]
pub enum Tensor {
    /// 32-bit floating point tensor.
    F32(Box<ArrayBuf<f32, Buffer<f32>>>),
    /// 64-bit floating point tensor.
    F64(Box<ArrayBuf<f64, Buffer<f64>>>),
    /// 64-bit unsigned integer tensor.
    U64(Box<ArrayBuf<u64, Buffer<u64>>>),
}

#[derive(Clone, Debug)]
pub enum TensorReduceResult {
    Scalar(Number),
    Tensor(Tensor),
}

/// Temporary collection enum.
#[derive(Clone, Debug)]
pub enum Collection {
    /// Transitional in-memory BTree data with explicit v1-style column schema.
    BTree(BTreeCollection),
    /// Tensor data stored entirely in memory. Variants cover f32 and u64 element types.
    Tensor(Tensor),
}

#[derive(Clone, Debug)]
pub struct BTreeCollection {
    pub schema: Vec<BTreeColumnSchema>,
    pub btree: BTree,
}

impl BTreeCollection {
    pub fn with_schema(schema: Vec<BTreeColumnSchema>, btree: BTree) -> Self {
        Self { schema, btree }
    }
}

impl From<ArrayBuf<f32, Buffer<f32>>> for Collection {
    fn from(tensor: ArrayBuf<f32, Buffer<f32>>) -> Self {
        Collection::Tensor(Tensor::F32(Box::new(tensor)))
    }
}

impl From<ArrayBuf<u64, Buffer<u64>>> for Collection {
    fn from(tensor: ArrayBuf<u64, Buffer<u64>>) -> Self {
        Collection::Tensor(Tensor::U64(Box::new(tensor)))
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
            Collection::BTree(_) => None,
        }
    }
}

impl<'en> en::IntoStream<'en> for Tensor {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let mut seq = encoder.encode_seq(Some(2))?;
        match self {
            Tensor::F32(array) => {
                let schema = (
                    number_type_path(&NumberType::Float(FloatType::F32)).to_string(),
                    array
                        .shape()
                        .iter()
                        .map(|dim| *dim as u64)
                        .collect::<Vec<_>>(),
                );
                seq.encode_element(schema)?;
                let values = array
                    .buffer()
                    .map_err(E::Error::custom)?
                    .to_slice()
                    .map_err(E::Error::custom)?
                    .into_vec();
                seq.encode_element(values)?;
            }
            Tensor::F64(array) => {
                let schema = (
                    number_type_path(&NumberType::Float(FloatType::F64)).to_string(),
                    array
                        .shape()
                        .iter()
                        .map(|dim| *dim as u64)
                        .collect::<Vec<_>>(),
                );
                seq.encode_element(schema)?;
                let values = array
                    .buffer()
                    .map_err(E::Error::custom)?
                    .to_slice()
                    .map_err(E::Error::custom)?
                    .into_vec();
                seq.encode_element(values)?;
            }
            Tensor::U64(array) => {
                let schema = (
                    number_type_path(&NumberType::UInt(UIntType::U64)).to_string(),
                    array
                        .shape()
                        .iter()
                        .map(|dim| *dim as u64)
                        .collect::<Vec<_>>(),
                );
                seq.encode_element(schema)?;
                let values = array
                    .buffer()
                    .map_err(E::Error::custom)?
                    .to_slice()
                    .map_err(E::Error::custom)?
                    .into_vec();
                seq.encode_element(values)?;
            }
        }
        seq.end()
    }
}

impl<'en> en::ToStream<'en> for Tensor {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        self.clone().into_stream(encoder)
    }
}

impl de::FromStream for Tensor {
    type Context = Arc<dyn Transaction>;

    async fn from_stream<D: de::Decoder>(
        _context: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct TensorVisitor;

        impl de::Visitor for TensorVisitor {
            type Value = Tensor;

            fn expecting() -> &'static str {
                "a TinyChain tensor payload"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let (dtype_path, shape): (String, Vec<u64>) = seq
                    .next_element(())
                    .await?
                    .ok_or_else(|| de::Error::custom("missing tensor schema"))?;
                let dtype = tensor_dtype_from_wire(&dtype_path).ok_or_else(|| {
                    de::Error::invalid_value(
                        dtype_path,
                        "a TinyChain numeric type path for tensor dtype",
                    )
                })?;

                let shape = coerce_shape(shape).map_err(de::Error::custom)?;

                let values = seq
                    .next_element::<Vec<Number>>(())
                    .await?
                    .ok_or_else(|| de::Error::custom("missing tensor values"))?;

                tensor_from_parts(dtype, shape, values).map_err(de::Error::custom)
            }
        }

        decoder.decode_seq(TensorVisitor).await
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
    transaction: Arc<dyn Transaction>,
    btree_roots: Option<(
        freqfs::DirLock<PersistentFile>,
        freqfs::DirLock<PersistentFile>,
    )>,
}

impl StateContext {
    pub fn new(transaction: Arc<dyn Transaction>) -> Self {
        Self {
            transaction,
            btree_roots: None,
        }
    }

    pub fn with_btree_roots(
        mut self,
        persistent_dir: freqfs::DirLock<PersistentFile>,
        txn_root: freqfs::DirLock<PersistentFile>,
    ) -> Self {
        self.btree_roots = Some((persistent_dir, txn_root));
        self
    }

    pub(crate) fn transaction(&self) -> Arc<dyn Transaction> {
        Arc::clone(&self.transaction)
    }

    pub(super) fn btree_decode_context(&self) -> Result<BTreeDecodeContext, String> {
        let (persistent_dir, txn_root) = self.btree_roots.clone().ok_or_else(|| {
            "BTree decode requires StateContext::with_btree_roots(...) at bootstrap".to_string()
        })?;

        Ok(BTreeDecodeContext::new(
            persistent_dir,
            txn_root,
            self.transaction().id(),
        ))
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

/// Transitional TinyChain state enum.
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

impl From<Collection> for State {
    fn from(collection: Collection) -> Self {
        State::Collection(collection)
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
