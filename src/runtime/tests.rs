use bytes::Bytes;
use destream::{de, en};
use futures::{stream, TryStreamExt};
use number_general::{FloatType, UIntType};
use safecast::{CastInto, TryCastFrom};
use tc_collection::PersistentFile;
use tc_ir::{Claim, IntoView, Map, NetworkTime, Scalar, Transaction, TxnId};
use tc_value::{NumberType, Value};

use super::*;

#[derive(Clone, Debug)]
pub(super) struct TestTxn {
    id: TxnId,
    claim: Claim,
    root: freqfs::DirLock<PersistentFile>,
    path: Vec<String>,
    tensor_limit: usize,
}

impl TestTxn {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "tc-state-test-txn-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("transaction root");
        let cache = freqfs::Cache::<PersistentFile>::new(
            16 * 1024 * 1024,
            None,
            0,
            std::time::Duration::from_secs(3),
        );
        let root = cache.load(root).expect("load transaction root");
        Self {
            id: TxnId::from_parts(NetworkTime::from_nanos(1), 1),
            claim: Claim::new("/test".parse().expect("test claim"), umask::Mode::all()),
            root,
            path: Vec::new(),
            tensor_limit: 256 * 1024 * 1024,
        }
    }

    fn with_tensor_limit(mut self, bytes: usize) -> Self {
        self.tensor_limit = bytes;
        self
    }
}

impl Transaction for TestTxn {
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

impl tc_collection::StorageContext for TestTxn {
    fn context(
        &self,
    ) -> impl std::future::Future<Output = tc_error::TCResult<freqfs::DirLock<PersistentFile>>> + Send
    {
        let root = self.root.clone();
        let mut path = vec![self.id.to_string()];
        path.extend(self.path.clone());
        async move {
            let mut current = root;
            for name in path {
                let next = {
                    let mut dir = current.write().await;
                    dir.get_or_create_dir(name)
                        .map_err(tc_error::TCError::internal)?
                };
                current = next;
            }
            Ok(current)
        }
    }

    fn subcontext(&self, name: impl Into<String>) -> Self {
        let mut txn = self.clone();
        txn.path.push(name.into());
        txn
    }

    fn subcontext_unique(&self) -> Self {
        self.subcontext("literal")
    }

    fn materialized_tensor_bytes(&self) -> usize {
        self.tensor_limit
    }
}

type TestState = State<TestTxn>;

async fn encode_json<T>(value: T) -> Vec<u8>
where
    T: for<'en> en::IntoStream<'en>,
{
    destream_json::encode(value)
        .expect("encode json payload")
        .map_err(|err| err.to_string())
        .try_fold(Vec::new(), |mut acc, chunk| async move {
            acc.extend_from_slice(&chunk);
            Ok(acc)
        })
        .await
        .expect("collect json payload")
}

async fn decode_json<T>(context: T::Context, bytes: Vec<u8>) -> Result<T, String>
where
    T: de::FromStream,
{
    let stream = stream::iter(vec![Ok::<Bytes, std::io::Error>(Bytes::from(bytes))]);
    destream_json::try_decode(context, stream)
        .await
        .map_err(|err| err.to_string())
}

#[test]
fn state_casts_to_value_and_value_vector() {
    let value = Value::try_cast_from(TestState::from(Value::from(7_u64)), |_| "invalid value")
        .expect("cast scalar state to value");
    assert_eq!(value, Value::from(7_u64));

    let values = Vec::<Value>::try_cast_from(
        TestState::Tuple(vec![
            TestState::from(Value::from(1_u64)),
            TestState::from(Value::from(2_u64)),
        ]),
        |_| "invalid row",
    )
    .expect("cast tuple state to value vector");
    assert_eq!(values, vec![Value::from(1_u64), Value::from(2_u64)]);

    assert!(Value::opt_cast_from(TestState::Tuple(vec![])).is_none());
}

#[test]
fn production_state_context_never_invents_a_transaction() {
    let source = include_str!("mod.rs");
    for forbidden in ["null_transaction", "NullTransaction", "tc_ir::Transaction"] {
        assert!(
            !source.contains(forbidden),
            "production state context must not contain {forbidden}"
        );
    }
}

#[tokio::test]
async fn scalar_numbers_round_trip() {
    let encoded = encode_json(true).await;
    let state: TestState = decode_json(TestTxn::new(), encoded)
        .await
        .expect("decode state");

    assert!(matches!(
        state,
        State::Scalar(Scalar::Value(Value::Number(_)))
    ));
}

#[tokio::test]
async fn tensor_round_trip() {
    let tensor = Tensor::dense_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("tensor");
    let state = TestState::Collection(Collection::Tensor(tensor));

    let encoded = encode_json(state.into_view(TestTxn::new()).await.expect("state view")).await;
    let decoded: TestState = decode_json(TestTxn::new(), encoded)
        .await
        .expect("decode state");

    match decoded {
        State::Collection(Collection::Tensor(tensor)) => assert_eq!(tensor.size(), 4),
        other => panic!("unexpected state {other:?}"),
    }
}

#[tokio::test]
async fn tensor_view_rejects_materialization_above_host_limit() {
    let tensor = Tensor::dense_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("tensor");
    let state = TestState::Collection(Collection::Tensor(tensor));
    let err = match state.into_view(TestTxn::new().with_tensor_limit(15)).await {
        Ok(_) => panic!("tensor materialization must be bounded"),
        Err(err) => err,
    };
    assert_eq!(err.code(), tc_error::ErrorKind::PayloadTooLarge);
    assert_eq!(
        err.pressure().map(tc_error::Pressure::resource),
        Some("/host/resource/tensor/materialized")
    );
}

#[tokio::test]
async fn state_map_round_trip_uses_plain_json_object() {
    let mut map = Map::new();
    map.insert(
        "status".parse().expect("id"),
        TestState::from(Value::from("ok")),
    );
    map.insert(
        "count".parse().expect("id"),
        TestState::from(Value::from(7_u64)),
    );
    let state = TestState::Map(map);

    let encoded = encode_json(state.into_view(TestTxn::new()).await.expect("state view")).await;
    let text = String::from_utf8(encoded.clone()).expect("utf-8");
    assert!(text.starts_with('{'));
    assert!(!text.contains("/state/scalar/map"));
    assert!(text.contains("\"status\""));
    assert!(text.contains("\"count\""));

    let decoded: TestState = decode_json(TestTxn::new(), encoded)
        .await
        .expect("decode state");

    assert!(matches!(decoded, State::Map(_)));
}

#[tokio::test]
async fn state_scalar_ref_serializes() {
    let state = TestState::Scalar(Scalar::from(tc_ir::TCRef::Id(
        "$foo".parse().expect("IdRef"),
    )));

    let encoded = encode_json(state.into_view(TestTxn::new()).await.expect("state view")).await;
    let text = String::from_utf8(encoded).expect("utf-8");
    assert_eq!(text, r#"{"$foo":[]}"#);
}

#[test]
fn tensor_facade_read_write_value_roundtrip() {
    let mut tensor = Tensor::dense_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("tensor");

    let value = tensor.read_value(&[1, 0]).expect("read value");
    let value: f64 = value.cast_into();
    assert_eq!(value, 3.0);

    tensor
        .write_value(&[0, 1], number_general::Number::from(9.5_f64))
        .expect("write value");

    assert_eq!(
        tensor.flattened_f32().expect("values"),
        vec![1.0, 9.5, 3.0, 4.0]
    );
}

#[tokio::test]
async fn tensor_f64_direct_round_trip() {
    let tensor = Tensor::dense_f64(vec![2, 2], vec![1.5, 2.5, 3.5, 4.5]).expect("tensor");

    let encoded = encode_json(tensor).await;
    let decoded: Tensor = decode_json((), encoded).await.expect("decode tensor");

    assert_eq!(decoded.shape(), &[2, 2]);
    assert_eq!(
        decoded.flattened_f64().expect("f64 values"),
        vec![1.5, 2.5, 3.5, 4.5]
    );
}

#[tokio::test]
async fn tensor_u64_direct_round_trip() {
    let tensor = Tensor::dense_u64(vec![2, 2], vec![1, 2, 3, 4]).expect("tensor");

    let encoded = encode_json(tensor).await;
    let decoded: Tensor = decode_json((), encoded).await.expect("decode tensor");

    assert_eq!(decoded.shape(), &[2, 2]);
    assert_eq!(
        decoded.flattened_u64().expect("u64 values"),
        vec![1, 2, 3, 4]
    );
}

#[test]
fn tensor_facade_cast_roundtrip() {
    let tensor = Tensor::dense_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("tensor");

    let as_f64 = tensor
        .clone()
        .cast(NumberType::Float(FloatType::F64))
        .expect("cast f64");
    assert_eq!(
        as_f64.flattened_f64().expect("f64 values"),
        vec![1.0, 2.0, 3.0, 4.0]
    );

    let as_u64 = tensor
        .cast(NumberType::UInt(UIntType::U64))
        .expect("cast u64");
    assert_eq!(
        as_u64.flattened_u64().expect("u64 values"),
        vec![1, 2, 3, 4]
    );
}

#[test]
fn tensor_facade_reduce_and_reduce_axes() {
    let tensor = Tensor::dense_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("tensor");

    let sum: f64 = tensor.reduce("sum").expect("reduce sum").cast_into();
    assert_eq!(sum, 10.0);

    let reduced = tensor
        .reduce_axes("sum", Some(vec![1]), false)
        .expect("reduce axis 1");
    let TensorReduceResult::Tensor(reduced) = reduced else {
        panic!("expected tensor reduction output");
    };

    assert_eq!(reduced.shape(), &[2]);
    assert_eq!(
        reduced.flattened_f32().expect("reduced values"),
        vec![3.0, 7.0]
    );
}

#[test]
fn tensor_facade_broadcast_roundtrip() {
    let tensor = Tensor::dense_f32(vec![2, 1], vec![1.0, 2.0]).expect("tensor");
    let broadcast = tensor.broadcast(vec![2, 3]).expect("broadcast");

    assert_eq!(broadcast.shape(), &[2, 3]);
    assert_eq!(
        broadcast.flattened_f32().expect("broadcast values"),
        vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    );
}

#[test]
fn tensor_facade_matmul_roundtrip() {
    let left = Tensor::dense_f32(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("left");
    let right =
        Tensor::dense_f32(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).expect("right");

    let product = left.matmul(&right).expect("matmul");
    assert_eq!(product.shape(), &[2, 2]);
    assert_eq!(
        product.flattened_f32().expect("matmul values"),
        vec![58.0, 64.0, 139.0, 154.0]
    );
}

#[test]
fn tensor_facade_slice_roundtrip() {
    let tensor = Tensor::dense_u64(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).expect("tensor");
    let range: Range = vec![AxisRange::from(0..2), AxisRange::from(1..3)]
        .into_iter()
        .collect();

    let sliced = tensor.slice(range).expect("slice");
    assert_eq!(sliced.shape(), &[2, 2]);
    assert_eq!(
        sliced.flattened_u64().expect("slice values"),
        vec![2, 3, 5, 6]
    );
}

#[test]
fn production_sources_do_not_construct_freqfs_cache() {
    const SOURCES: [(&str, &str); 9] = [
        ("src/lib.rs", include_str!("../lib.rs")),
        ("src/codec/class.rs", include_str!("../codec/class.rs")),
        ("src/codec/decode.rs", include_str!("../codec/decode.rs")),
        ("src/codec/helpers.rs", include_str!("../codec/helpers.rs")),
        ("src/codec/mod.rs", include_str!("../codec/mod.rs")),
        ("src/codec/parse.rs", include_str!("../codec/parse.rs")),
        ("src/runtime/class.rs", include_str!("class.rs")),
        ("src/runtime/route.rs", include_str!("route.rs")),
        ("src/runtime/mod.rs", include_str!("mod.rs")),
    ];

    for (path, source) in SOURCES {
        assert!(
            !source.contains("Cache::new("),
            "production source {path} must not call Cache::new"
        );
    }
}
