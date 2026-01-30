//! Transitional TinyChain state primitives.
//!
//! This crate exposes the placeholder collection and scalar state enums used by
//! adapters that need to exchange TinyChain values before the full transactional
//! filesystem (`freqfs`) + `Chain` + `Service` stack lands. The in-memory tensor
//! representation keeps downstream crates unblocked while we finish the shared
//! persistence layer.

use std::{any::Any, convert::TryFrom, sync::Arc};

use destream::{
    de,
    en::{self, EncodeMap, EncodeSeq, Error as _},
    IntoStream,
};
use ha_ndarray::{ArrayBuf, Buffer, NDArray, NDArrayRead};
use number_general::{FloatType, Number, UIntType};
use pathlink::PathBuf;
use safecast::{CastInto, TryCastFrom};
use tc_ir::Scalar;
use tc_value::{number_type_from_path, number_type_path, NumberType, Value, ValueType};

mod class;

pub use class::{CollectionType, StateType, TensorType};
pub use tc_ir::{Class, NativeClass};

/// Temporary tensor representation (in-memory only).
#[derive(Clone, Debug)]
pub enum Tensor {
    /// 32-bit floating point tensor.
    F32(Box<ArrayBuf<f32, Buffer<f32>>>),
    /// 64-bit unsigned integer tensor.
    U64(Box<ArrayBuf<u64, Buffer<u64>>>),
}

impl Tensor {
    /// Construct a dense `f32` tensor from a shape and flattened values.
    pub fn dense_f32(shape: Vec<usize>, values: Vec<f32>) -> Result<Self, String> {
        let shape = shape.into();
        let buffer = Buffer::from(values);
        ArrayBuf::new(buffer, shape)
            .map(Box::new)
            .map(Tensor::F32)
            .map_err(|err| err.to_string())
    }

    /// Construct a dense `u64` tensor from a shape and flattened values.
    pub fn dense_u64(shape: Vec<usize>, values: Vec<u64>) -> Result<Self, String> {
        let shape = shape.into();
        let buffer = Buffer::from(values);
        ArrayBuf::new(buffer, shape)
            .map(Box::new)
            .map(Tensor::U64)
            .map_err(|err| err.to_string())
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::F32(array) => array.shape(),
            Tensor::U64(array) => array.shape(),
        }
    }

    pub fn flattened_f32(&self) -> Result<Vec<f32>, String> {
        match self {
            Tensor::F32(array) => Ok(array
                .buffer()
                .map_err(|err| err.to_string())?
                .to_slice()
                .map_err(|err| err.to_string())?
                .into_vec()),
            Tensor::U64(_) => Err("tensor dtype is not f32".to_string()),
        }
    }

    pub fn flattened_u64(&self) -> Result<Vec<u64>, String> {
        match self {
            Tensor::U64(array) => Ok(array
                .buffer()
                .map_err(|err| err.to_string())?
                .to_slice()
                .map_err(|err| err.to_string())?
                .into_vec()),
            Tensor::F32(_) => Err("tensor dtype is not u64".to_string()),
        }
    }
}

/// Temporary collection enum.
#[derive(Clone, Debug)]
pub enum Collection {
    /// Tensor data stored entirely in memory. Variants cover f32 and u64 element types.
    Tensor(Tensor),
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
    type Context = StateContext;

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
                let dtype_path_buf = dtype_path
                    .parse::<PathBuf>()
                    .map_err(|err| de::Error::custom(err.to_string()))?;
                let dtype = number_type_from_path(&dtype_path_buf).ok_or_else(|| {
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

/// Opaque context handle forwarded to state deserializers.
#[derive(Clone)]
pub struct StateContext {
    inner: Arc<dyn Any + Send + Sync>,
}

impl StateContext {
    /// Returns a new context handle containing no additional metadata.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Attempts to borrow the context as the requested type.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.inner.as_ref().downcast_ref::<T>()
    }

    /// Stores typed context data alongside this handle.
    pub fn with_data<T>(data: T) -> Self
    where
        T: Any + Send + Sync + 'static,
    {
        Self {
            inner: Arc::new(data),
        }
    }
}

impl Default for StateContext {
    fn default() -> Self {
        Self {
            inner: Arc::new(()),
        }
    }
}

/// Transitional TinyChain state enum.
#[derive(Clone, Debug)]
pub enum State {
    None,
    Scalar(Scalar),
    Collection(Collection),
}

impl State {
    pub fn is_none(&self) -> bool {
        matches!(
            self,
            State::None | State::Scalar(Scalar::Value(Value::None))
        )
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

impl de::FromStream for State {
    type Context = StateContext;

    async fn from_stream<D: de::Decoder>(
        context: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct StateVisitor {
            context: StateContext,
        }

        impl de::Visitor for StateVisitor {
            type Value = State;

            fn expecting() -> &'static str {
                "a TinyChain state placeholder"
            }

            fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(State::None)
            }

            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(State::None)
            }

            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(State::from(Number::from(value)))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(State::from(Number::from(value)))
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(State::from(Number::from(value)))
            }

            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
                Ok(State::from(Number::from(value)))
            }

            fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(State::from(Value::from(value)))
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                while seq.next_element::<de::IgnoredAny>(()).await?.is_some() {}
                Err(de::Error::custom(
                    "TinyChain state cannot be decoded from a sequence",
                ))
            }

            async fn visit_map<A: de::MapAccess>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let key = map
                    .next_key::<String>(())
                    .await?
                    .ok_or_else(|| de::Error::custom("expected TinyChain state type path"))?;

                let path = key
                    .parse::<PathBuf>()
                    .map_err(|err| de::Error::custom(err.to_string()))?;

                let state_type = StateType::from_path(&path).ok_or_else(|| {
                    de::Error::invalid_value(path.to_string(), "a known TinyChain state type path")
                })?;

                match state_type {
                    StateType::Collection(CollectionType::Tensor(_)) => {
                        let tensor = map.next_value::<Tensor>(self.context.clone()).await?;
                        drain_remaining_entries(&mut map).await?;
                        Ok(State::Collection(Collection::Tensor(tensor)))
                    }
                    StateType::Scalar(value_type) => {
                        let value = decode_value_entry(value_type, &mut map).await?;
                        drain_remaining_entries(&mut map).await?;
                        Ok(State::Scalar(Scalar::from(value)))
                    }
                }
            }
        }

        decoder.decode_any(StateVisitor { context }).await
    }
}

impl<'en> en::IntoStream<'en> for State {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            State::None => encoder.encode_unit(),
            State::Scalar(Scalar::Value(value)) => value.into_stream(encoder),
            State::Scalar(Scalar::Ref(_)) => Err(E::Error::custom(
                "cannot serialize Scalar::Ref as State until TCRef encoding is implemented",
            )),
            State::Collection(collection) => collection.into_stream(encoder),
        }
    }
}

impl From<Number> for State {
    fn from(number: Number) -> Self {
        State::from(Value::from(number))
    }
}

async fn decode_value_entry<A: de::MapAccess>(
    value_type: ValueType,
    map: &mut A,
) -> Result<Value, A::Error> {
    match value_type {
        ValueType::Number => map.next_value::<Number>(()).await.map(Value::Number),
        ValueType::None => {
            let _ = map.next_value::<de::IgnoredAny>(()).await?;
            Ok(Value::None)
        }
        ValueType::String => map.next_value::<String>(()).await.map(Value::String),
    }
}

async fn drain_remaining_entries<A: de::MapAccess>(map: &mut A) -> Result<(), A::Error> {
    while map.next_key::<de::IgnoredAny>(()).await?.is_some() {
        let _ = map.next_value::<de::IgnoredAny>(()).await?;
    }
    Ok(())
}

fn tensor_from_parts(
    dtype: NumberType,
    shape: Vec<usize>,
    values: Vec<Number>,
) -> Result<Tensor, String> {
    match dtype {
        NumberType::Float(FloatType::F32) => {
            let values = numbers_to_f32(values)?;
            Tensor::dense_f32(shape, values)
        }
        NumberType::UInt(UIntType::U64) => {
            let values = numbers_to_u64(values)?;
            Tensor::dense_u64(shape, values)
        }
        other => Err(format!("unsupported tensor dtype {other}")),
    }
}

fn coerce_shape(dims: Vec<u64>) -> Result<Vec<usize>, String> {
    dims.into_iter()
        .map(|dim| usize::try_from(dim).map_err(|_| format!("invalid dimension {dim}")))
        .collect()
}

fn numbers_to_f32(values: Vec<Number>) -> Result<Vec<f32>, String> {
    values
        .into_iter()
        .map(|number| {
            if matches!(number, Number::Complex(_)) {
                Err("complex numbers are not supported in tensors".into())
            } else {
                Ok(number.cast_into())
            }
        })
        .collect()
}

fn numbers_to_u64(values: Vec<Number>) -> Result<Vec<u64>, String> {
    values
        .into_iter()
        .map(|number| {
            ensure_tensor_u64_component(&number)?;
            Ok(number.cast_into())
        })
        .collect()
}

fn ensure_tensor_u64_component(number: &Number) -> Result<(), String> {
    match number {
        Number::Bool(_) | Number::UInt(_) => Ok(()),
        Number::Int(int) => {
            let value = i64::from(*int);
            if value < 0 {
                Err("tensor values must be non-negative".into())
            } else {
                Ok(())
            }
        }
        Number::Float(float) => {
            let value = f64::from(*float);
            if !value.is_finite() {
                Err("tensor value must be finite".into())
            } else if value < 0.0 || value.fract() != 0.0 {
                Err(format!(
                    "expected a non-negative whole number but found {value}"
                ))
            } else {
                Ok(())
            }
        }
        Number::Complex(_) => Err("complex numbers are not supported in tensors".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use destream::{de, en};
    use futures::{executor::block_on, stream, TryStreamExt};
    use ha_ndarray::NDArray;

    fn encode_json<T>(value: T) -> Vec<u8>
    where
        T: for<'en> en::IntoStream<'en>,
    {
        block_on(
            destream_json::encode(value)
                .expect("encode json payload")
                .map_err(|err| err.to_string())
                .try_fold(Vec::new(), |mut acc, chunk| async move {
                    acc.extend_from_slice(&chunk);
                    Ok(acc)
                }),
        )
        .expect("collect json payload")
    }

    fn decode_json<T>(context: T::Context, bytes: Vec<u8>) -> T
    where
        T: de::FromStream,
    {
        let stream = stream::iter(vec![Ok::<Bytes, std::io::Error>(Bytes::from(bytes))]);
        block_on(destream_json::try_decode(context, stream)).expect("decode json payload")
    }

    #[test]
    fn tensor_variant_matches() {
        let shape = Vec::from([3usize]).into();
        let tensor = ArrayBuf::<f32, Buffer<f32>>::new(vec![1.0_f32, 2.0, 3.0].into(), shape)
            .expect("tensor buffer");
        let collection = Collection::from(tensor);
        assert!(Tensor::can_cast_from(&collection));
        match &collection {
            Collection::Tensor(Tensor::F32(buf)) => assert_eq!(buf.size(), 3),
            _ => panic!("expected F32 tensor"),
        }
    }

    #[test]
    fn state_context_downcasts() {
        let ctx = StateContext::with_data(String::from("hello"));
        assert!(ctx.downcast_ref::<String>().is_some());
    }

    #[test]
    fn scalar_numbers_round_trip() {
        let encoded = encode_json(true);
        let state: State = decode_json(StateContext::empty(), encoded);
        assert!(matches!(
            state,
            State::Scalar(Scalar::Value(Value::Number(_)))
        ));
    }

    #[test]
    fn tensor_round_trip() {
        let shape = Vec::from([2usize, 2usize]).into();
        let tensor = ArrayBuf::<f32, Buffer<f32>>::new(vec![1.0, 2.0, 3.0, 4.0].into(), shape)
            .expect("tensor buffer");
        let state = State::Collection(Collection::from(tensor));

        let encoded = encode_json(state.clone());
        let decoded: State = decode_json(StateContext::empty(), encoded);

        match decoded {
            State::Collection(Collection::Tensor(Tensor::F32(buf))) => {
                assert_eq!(buf.size(), 4);
            }
            other => panic!("unexpected state {other:?}"),
        }
    }

    #[test]
    fn tensor_direct_round_trip() {
        let tensor = Tensor::dense_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("dense tensor");
        let bytes = encode_json(tensor.clone());
        let decoded: Tensor = decode_json(StateContext::empty(), bytes);
        match decoded {
            Tensor::F32(buf) => assert_eq!(buf.size(), 4),
            other => panic!("unexpected tensor variant {other:?}"),
        }
    }
}
