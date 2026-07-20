use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use destream::{de, en};
use futures::{stream, TryStreamExt};
use number_general::{FloatType, UIntType};
use safecast::CastInto;
use tc_collection::btree::PersistentFile;
use tc_ir::{Map, Scalar};
use tc_value::{NumberType, Value, ValueType};

use super::*;

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

fn unique_test_root(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();

    std::env::temp_dir().join(format!("tc-state-{label}-{nanos}"))
}

fn load_btree_roots(
    root: &Path,
) -> (
    freqfs::DirLock<PersistentFile>,
    freqfs::DirLock<PersistentFile>,
) {
    std::fs::create_dir_all(root.join("persistent")).expect("create persistent root dir");
    std::fs::create_dir_all(root.join("txn")).expect("create txn root dir");

    let cache = freqfs::Cache::<PersistentFile>::new(16 * 1024 * 1024, None);
    let persistent = Arc::clone(&cache)
        .load(root.join("persistent"))
        .expect("load persistent root");
    let txn = Arc::clone(&cache)
        .load(root.join("txn"))
        .expect("load txn root");

    (persistent, txn)
}

fn run_async_with_large_stack(
    name: &str,
    test_fn: impl FnOnce() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
    + Send
    + 'static,
) {
    std::thread::Builder::new()
        .name(name.to_string())
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("create test runtime");

            runtime.block_on(test_fn());
        })
        .expect("spawn test thread")
        .join()
        .expect("join test thread");
}

#[tokio::test]
async fn scalar_numbers_round_trip() {
    let encoded = encode_json(true).await;
    let state: State = decode_json(state_context(null_transaction()), encoded)
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
    let state = State::Collection(Collection::Tensor(tensor));

    let encoded = encode_json(state).await;
    let decoded: State = decode_json(state_context(null_transaction()), encoded)
        .await
        .expect("decode state");

    match decoded {
        State::Collection(Collection::Tensor(Tensor::F32(buf))) => assert_eq!(buf.size(), 4),
        other => panic!("unexpected state {other:?}"),
    }
}

#[tokio::test]
async fn state_map_round_trip_uses_plain_json_object() {
    let mut map = Map::new();
    map.insert(
        "status".parse().expect("id"),
        State::from(Value::from("ok")),
    );
    map.insert(
        "count".parse().expect("id"),
        State::from(Value::from(7_u64)),
    );
    let state = State::Map(map);

    let encoded = encode_json(state).await;
    let text = String::from_utf8(encoded.clone()).expect("utf-8");
    assert!(text.starts_with('{'));
    assert!(!text.contains("/state/scalar/map"));
    assert!(text.contains("\"status\""));
    assert!(text.contains("\"count\""));

    let decoded: State = decode_json(state_context(null_transaction()), encoded)
        .await
        .expect("decode state");

    assert!(matches!(decoded, State::Map(_)));
}

#[tokio::test]
async fn state_scalar_ref_serializes() {
    let state = State::Scalar(Scalar::from(tc_ir::TCRef::Id(
        "$foo".parse().expect("IdRef"),
    )));

    let encoded = encode_json(state).await;
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
    let decoded: Tensor = decode_json(null_transaction(), encoded)
        .await
        .expect("decode tensor");

    assert_eq!(decoded.shape(), &[2, 2]);
    assert_eq!(decoded.flattened_f64().expect("f64 values"), vec![1.5, 2.5, 3.5, 4.5]);
}

#[tokio::test]
async fn tensor_u64_direct_round_trip() {
    let tensor = Tensor::dense_u64(vec![2, 2], vec![1, 2, 3, 4]).expect("tensor");

    let encoded = encode_json(tensor).await;
    let decoded: Tensor = decode_json(null_transaction(), encoded)
        .await
        .expect("decode tensor");

    assert_eq!(decoded.shape(), &[2, 2]);
    assert_eq!(decoded.flattened_u64().expect("u64 values"), vec![1, 2, 3, 4]);
}

#[test]
fn tensor_facade_cast_roundtrip() {
    let tensor = Tensor::dense_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("tensor");

    let as_f64 = tensor
        .clone()
        .cast(NumberType::Float(FloatType::F64))
        .expect("cast f64");
    assert_eq!(as_f64.flattened_f64().expect("f64 values"), vec![1.0, 2.0, 3.0, 4.0]);

    let as_u64 = tensor
        .cast(NumberType::UInt(UIntType::U64))
        .expect("cast u64");
    assert_eq!(as_u64.flattened_u64().expect("u64 values"), vec![1, 2, 3, 4]);
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
    assert_eq!(reduced.flattened_f32().expect("reduced values"), vec![3.0, 7.0]);
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
    assert_eq!(sliced.flattened_u64().expect("slice values"), vec![2, 3, 5, 6]);
}

#[tokio::test]
async fn btree_decode_fails_clearly_without_roots() {
    let payload = br#"{"/state/collection/btree":null}"#.to_vec();
    let err = decode_json::<State>(state_context(null_transaction()), payload)
        .await
        .expect_err("decode should fail without btree roots");

    assert!(
        err.contains("StateContext::with_btree_roots"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn btree_decode_path_uses_provided_roots() {
    let root = unique_test_root("btree-decode");
    std::fs::create_dir_all(&root).expect("create temp root");
    let (persistent, txn_root) = load_btree_roots(&root);

    let context = state_context(null_transaction()).with_btree_roots(persistent, txn_root);
    let payload = br#"{"/state/collection/btree":null}"#.to_vec();
    let result = decode_json::<State>(context, payload).await;

    if let Err(err) = result {
        assert!(
            !err.contains("StateContext::with_btree_roots"),
            "decode should move past missing-roots failure when roots are provided: {err}"
        );
    }
}

#[test]
fn btree_decode_valid_payload_succeeds_with_roots() {
    run_async_with_large_stack("tc-state-btree-valid-payload", || {
        Box::pin(async move {
            let root = unique_test_root("btree-valid-payload");
            std::fs::create_dir_all(&root).expect("create temp root");
            let (persistent, txn_root) = load_btree_roots(&root);

            let decode_txn = null_transaction();
            let context =
                state_context(Arc::clone(&decode_txn)).with_btree_roots(persistent, txn_root);
            let payload =
                br#"{"/state/collection/btree":[[["id","/state/scalar/value/number"]],[1,2,3]]}"#
                    .to_vec();

            let decoded: State = decode_json(context, payload).await.expect("decode btree state");

            let State::Collection(Collection::BTree(btree)) = decoded else {
                panic!("expected decoded BTree collection state");
            };

            assert_eq!(btree.schema.len(), 1);
            assert_eq!(btree.schema[0].name, "id");
            assert_eq!(btree.schema[0].dtype, ValueType::Number);

            // Successful decode with schema materialization proves the bootstrap-root decode path
            // accepted a valid [schema, rows] BTree payload.
            assert_eq!(btree.schema[0].max_size, None);
        })
    });
}

#[test]
fn production_sources_do_not_construct_freqfs_cache() {
    const SOURCES: [(&str, &str); 8] = [
        ("src/lib.rs", include_str!("../lib.rs")),
        ("src/codec/class.rs", include_str!("../codec/class.rs")),
        ("src/codec/decode.rs", include_str!("../codec/decode.rs")),
        ("src/codec/helpers.rs", include_str!("../codec/helpers.rs")),
        ("src/codec/mod.rs", include_str!("../codec/mod.rs")),
        ("src/codec/parse.rs", include_str!("../codec/parse.rs")),
        ("src/runtime/mod.rs", include_str!("mod.rs")),
        ("src/runtime/tensor.rs", include_str!("tensor.rs")),
    ];

    for (path, source) in SOURCES {
        assert!(
            !source.contains("Cache::new("),
            "production source {path} must not call Cache::new"
        );
    }
}
