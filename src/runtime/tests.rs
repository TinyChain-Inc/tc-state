use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use destream::{de, en};
use futures::{stream, TryStreamExt};
use safecast::CastInto;
use tc_collection::btree::PersistentFile;
use tc_ir::{Map, Scalar};
use tc_value::Value;

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
