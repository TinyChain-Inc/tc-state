use std::sync::Arc;

use bytes::Bytes;
use destream::{de, en};
use futures::{stream, StreamExt, TryStreamExt};
use number_general::{FloatType, Number};
use safecast::{CastFrom, TryCastFrom};
use tc_collection::PersistentFile;
use tc_ir::{IntoView, Map, NetworkTime, Scalar, Transaction, TxnId};
use tc_value::class::NativeClass;
use tc_value::{NumberType, Value};

use crate::*;
pub(crate) use temp_root::TempRoot;

#[path = "../../common/temp_root.rs"]
mod temp_root;

#[derive(Clone, Debug)]
pub(crate) struct TestTxn {
    id: TxnId,
    root: freqfs::DirLock<PersistentFile>,
    path: Vec<String>,
    next: std::sync::Arc<std::sync::atomic::AtomicU64>,
    tensor_limit: usize,
    // Last: release this transaction's directory handle before its cleanup owner.
    _temp: Arc<TempRoot>,
}

impl TestTxn {
    pub(crate) fn new() -> (Arc<TempRoot>, Self) {
        let temp = TempRoot::new();
        let cache = freqfs::Cache::<PersistentFile>::new(
            16 * 1024 * 1024,
            None,
            0,
            std::time::Duration::from_secs(3),
        );

        let root = cache
            .load(temp.path().to_path_buf())
            .expect("load transaction root");

        let txn = Self {
            id: TxnId::from_parts(NetworkTime::from_nanos(1), 1),
            root,
            path: Vec::new(),
            next: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            tensor_limit: 256 * 1024 * 1024,
            _temp: Arc::clone(&temp),
        };

        (temp, txn)
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
}

impl tc_collection::StorageContext for TestTxn {
    type File = PersistentFile;

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
        self.subcontext(format!("literal-{}", {
            self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        }))
    }

    fn materialized_tensor_bytes(&self) -> usize {
        self.tensor_limit
    }
}

type TestState = State<TestTxn>;

async fn tensor_values<T: CastFrom<number_general::Number>>(
    tensor: &Tensor<TestTxn>,
    txn: &TestTxn,
) -> Vec<T> {
    tensor
        .elements(txn.id())
        .await
        .unwrap()
        .map_ok(T::cast_from)
        .try_collect()
        .await
        .unwrap()
}

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

#[tokio::test]
async fn state_decoder_rejects_the_shared_shadowing_fixture() {
    let fixture = include_bytes!("../../../../tc-ir/fixtures/lexical/invalid-shadow.json");
    let (_root, txn) = TestTxn::new();
    let error = decode_json::<TestState>(txn, fixture.to_vec())
        .await
        .expect_err("reject nested lexical shadowing");

    assert!(error.contains("shadowed OpDef binding"), "{error}");
}

#[test]
fn state_casts_to_value_and_value_vector() {
    for value in [Value::from(7_u64), Value::Number(1.25_f64.into())] {
        let Value::Number(expected) = value else {
            unreachable!()
        };

        let state = TestState::from(Value::Number(expected));
        assert!(Number::can_cast_from(&&state));
        assert_eq!(Number::opt_cast_from(&state), Some(expected));
        assert_eq!(
            tc_collection::CollectionState::as_number(&state),
            Some(expected)
        );
    }

    for state in [
        TestState::from(Value::None),
        TestState::from(Value::from("7")),
        TestState::Tuple(vec![TestState::from(Value::from(7_u64))]),
        TestState::Map(Default::default()),
    ] {
        assert!(!Number::can_cast_from(&&state));
        assert!(Number::opt_cast_from(&state).is_none());
        assert!(tc_collection::CollectionState::as_number(&state).is_none());
    }

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

    assert_eq!(
        Value::opt_cast_from(TestState::Tuple(vec![])),
        Some(Value::Tuple(vec![]))
    );

    for depth in [tc_value::MAX_VALUE_DEPTH, tc_value::MAX_VALUE_DEPTH + 1] {
        let mut value = Value::from(7_u64);
        for _ in 0..depth {
            value = Value::Tuple(vec![value]);
        }

        let state = TestState::from(value);
        let accepted = depth == tc_value::MAX_VALUE_DEPTH;
        assert_eq!(Value::can_cast_from(&state), accepted);
        assert_eq!(Value::opt_cast_from(state.clone()).is_some(), accepted);
        assert_eq!(Vec::<Value>::can_cast_from(&state), accepted);
        assert_eq!(Vec::<Value>::opt_cast_from(state).is_some(), accepted);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recursive_value_conversions_share_one_depth_allowance() {
    tokio::spawn(async {
        let limit = tc_value::MAX_VALUE_DEPTH;
        for (state_depth, scalar_depth, value_depth) in [
            (0, 0, limit),
            (0, limit, 0),
            (limit, 0, 0),
            (21, 21, 22),
            (21, 21, 23),
            (limit + 1, 0, 0),
            (0, limit + 1, 0),
            (0, 0, limit + 1),
        ] {
            let mut value = Value::from(7_u64);
            for _ in 0..value_depth {
                value = Value::Tuple(vec![value]);
            }

            let mut scalar = Scalar::Value(value);
            for _ in 0..scalar_depth {
                scalar = Scalar::Tuple(vec![scalar]);
            }

            let mut state = TestState::Scalar(scalar);
            for _ in 0..state_depth {
                state = State::Tuple(vec![state]);
            }

            let accepted = state_depth + scalar_depth + value_depth <= limit;
            assert_eq!(Value::can_cast_from(&state), accepted);
            let converted = Value::opt_cast_from(state);
            assert_eq!(converted.is_some(), accepted);
            if let Some(value) = converted {
                assert!(value.within_depth(limit));
                let mut leaf = &value;
                for _ in 0..state_depth + scalar_depth + value_depth {
                    let Value::Tuple(items) = leaf else {
                        panic!("expected tuple")
                    };

                    assert_eq!(items.len(), 1);
                    leaf = &items[0];
                }

                assert_eq!(leaf, &Value::from(7_u64));
            }
        }

        let embedded = Value::Tuple(vec![7_u64.into()]);
        let singleton = TestState::from(embedded.clone());
        assert_eq!(Vec::<Value>::opt_cast_from(singleton), Some(vec![embedded]));
        let structural = TestState::Scalar(Scalar::Tuple(vec![
            Scalar::from(7_u64),
            Scalar::Tuple(vec![Scalar::from(8_u64)]),
        ]));

        assert!(Vec::<Value>::can_cast_from(&structural));
        assert_eq!(
            Vec::<Value>::opt_cast_from(structural),
            Some(vec![7_u64.into(), Value::Tuple(vec![8_u64.into()])])
        );

        for empty in [
            TestState::Tuple(vec![]),
            TestState::Scalar(Scalar::Tuple(vec![])),
            TestState::from(Value::Tuple(vec![])),
        ] {
            assert!(Value::can_cast_from(&empty));
            assert_eq!(Value::opt_cast_from(empty), Some(Value::Tuple(vec![])));
        }

        for rejected in [
            TestState::None,
            TestState::Map(Map::new()),
            TestState::Scalar(Scalar::Map(Map::new())),
            TestState::Scalar(Scalar::from(tc_ir::TCRef::Id("$foo".parse().unwrap()))),
        ] {
            let state = TestState::Tuple(vec![TestState::from(7_u64), rejected]);
            assert!(!Value::can_cast_from(&state));
            assert!(Value::opt_cast_from(state.clone()).is_none());
            assert!(!Vec::<Value>::can_cast_from(&state));
            assert!(Vec::<Value>::opt_cast_from(state).is_none());
        }
    })
    .await
    .unwrap();
}

#[test]
fn scalar_value_tuple_uses_canonical_state_tuple_conversion() {
    let tuple = TestState::from(Value::Tuple(vec![1u64.into(), 2u64.into()]));
    assert!(Vec::<TestState>::can_cast_from(&tuple));
    let values = Vec::<TestState>::opt_cast_from(tuple).unwrap();
    assert_eq!(values.len(), 2);
    assert_eq!(Value::opt_cast_from(values[1].clone()), Some(2u64.into()));
}

#[tokio::test]
async fn scalar_numbers_round_trip() {
    let encoded = encode_json(true).await;
    let (_decode_root, decode_txn) = TestTxn::new();
    let state: TestState = decode_json(decode_txn, encoded)
        .await
        .expect("decode state");

    assert!(Value::opt_cast_from(state).is_some_and(|value| matches!(&value, Value::Number(_))));
    for (dtype, number, accepted) in [
        (
            NumberType::Int(number_general::IntType::I8),
            Number::from(127u64),
            true,
        ),
        (
            NumberType::Int(number_general::IntType::I8),
            Number::from(128u64),
            false,
        ),
        (
            NumberType::Int(number_general::IntType::I8),
            Number::from(1.5f64),
            false,
        ),
        (
            NumberType::Float(FloatType::F32),
            Number::from(f64::MIN_POSITIVE),
            false,
        ),
    ] {
        let class = tc_value::ValueType::Number(dtype);
        let encoded = encode_json(std::collections::BTreeMap::from([(
            class.path().to_string(),
            number,
        )]))
        .await;

        let (_root, txn) = TestTxn::new();
        let decoded: Result<TestState, _> = decode_json(txn, encoded).await;
        if accepted {
            let value = Value::opt_cast_from(decoded.unwrap()).unwrap();
            assert_eq!(value, Value::Number(number));
            assert!(class.accepts(&value.class()));
        } else {
            assert!(decoded.is_err(), "{dtype:?}");
        }
    }
}

#[tokio::test]
async fn class_instance_round_trip_is_self_contained() {
    let identity: pathlink::Link = "/class/example-devco/vector/1.0.0"
        .parse()
        .expect("Class identity");

    let parent = ClassParent::Native(StateType::Tuple);
    let prototype: Map<_> = [(
        "dimensions".parse().expect("member ID"),
        Scalar::from(Value::from(3_u64)),
    )]
    .into_iter()
    .collect();

    let body = ClassBody::new(parent, prototype);
    let borrowed = destream_json::encode(&body)
        .unwrap()
        .try_fold(Vec::new(), |mut bytes, chunk| async move {
            bytes.extend_from_slice(&chunk);
            Ok(bytes)
        })
        .await
        .unwrap();

    assert_eq!(borrowed, encode_json(body.clone()).await);
    let class = ClassDef::from_body(identity, body);
    let borrowed = tbon::en::encode(&class)
        .unwrap()
        .try_fold(Vec::new(), |mut bytes, chunk| async move {
            bytes.extend_from_slice(&chunk);
            Ok(bytes)
        })
        .await
        .unwrap();

    let owned = tbon::en::encode(class.clone())
        .unwrap()
        .try_fold(Vec::new(), |mut bytes, chunk| async move {
            bytes.extend_from_slice(&chunk);
            Ok(bytes)
        })
        .await
        .unwrap();

    assert_eq!(borrowed, owned);
    let members: Map<_> = [(
        "name".parse().expect("member ID"),
        TestState::from(Value::from("v")),
    )]
    .into_iter()
    .collect();

    let instance = ClassInstance::new(
        TestState::Tuple(vec![TestState::from(1_u64)]),
        class.clone(),
        members,
    );

    let state = TestState::from(Object::Instance(instance));

    let (_view_root, view_txn) = TestTxn::new();
    let encoded = encode_json(state.into_view(view_txn).await.expect("instance view")).await;
    let (_decode_root, decode_txn) = TestTxn::new();
    let decoded: TestState = decode_json(decode_txn, encoded)
        .await
        .expect("decode instance");

    let State::Object(object) = decoded else {
        panic!("expected decoded object")
    };

    let Object::Instance(instance) = *object else {
        panic!("expected decoded instance")
    };

    assert_eq!(instance.class(), &class);
    assert!(matches!(&instance.parent(), State::Tuple(items) if items.len() == 1));
    assert_eq!(instance.members().len(), 1);

    let class_state = TestState::from(Object::Class(class.clone()));
    let (_class_view_root, class_view_txn) = TestTxn::new();
    let encoded = encode_json(
        class_state
            .into_view(class_view_txn)
            .await
            .expect("Class view"),
    )
    .await;

    let (_class_decode_root, class_decode_txn) = TestTxn::new();
    let decoded: TestState = decode_json(class_decode_txn, encoded)
        .await
        .expect("decode Class");

    assert!(matches!(
        &decoded,
        State::Object(object) if matches!(object.as_ref(), Object::Class(decoded) if decoded == &class)
    ));

    let body = String::from_utf8(encode_json(class).await).unwrap();
    for (name, fields) in [
        ("missing parent", "[]".to_owned()),
        ("missing class", "[null]".to_owned()),
        ("missing members", format!("[null,{body}]")),
        ("non-map members", format!("[null,{body},[]]")),
        (
            "duplicate member",
            format!("[null,{body},{{\"x\":null,\"x\":null}}]"),
        ),
        ("extra field", format!("[null,{body},{{}},null]")),
    ] {
        let path = crate::StateType::Object(crate::ObjectType::Instance).path();
        let bytes = format!("{{\"{path}\":{fields}}}").into_bytes();
        let (_root, txn) = TestTxn::new();
        assert!(
            decode_json::<TestState>(txn, bytes).await.is_err(),
            "{name}: {fields}"
        );
    }
}

#[tokio::test]
async fn tensor_round_trip() {
    let (_root, txn) = TestTxn::new();
    let tensor = Tensor::dense_f32(&txn, vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
        .await
        .expect("tensor");

    let state = TestState::Collection(Collection::Tensor(tensor));

    let (_view_root, view_txn) = TestTxn::new();
    let encoded = encode_json(state.into_view(view_txn).await.expect("state view")).await;
    let (_decode_root, decode_txn) = TestTxn::new();
    let decoded: TestState = decode_json(decode_txn, encoded.clone())
        .await
        .expect("decode state");

    match decoded {
        State::Collection(Collection::Tensor(tensor)) => assert_eq!(tensor.size().unwrap(), 4),
        other => panic!("unexpected state {:?}", other),
    }

    // Tuple indices and map keys delegate independent storage contexts.
    let nested = format!(
        "[{{\"left\":{0}}},{{\"right\":{0}}}]",
        std::str::from_utf8(&encoded).unwrap()
    )
    .into_bytes();

    let (_nested_root, nested_txn) = TestTxn::new();
    let decoded: TestState = decode_json(nested_txn.clone(), nested).await.unwrap();
    let State::Tuple(items) = &decoded else {
        panic!("nested tuple")
    };

    let State::Map(left) = &items[0] else {
        panic!("left map")
    };

    let State::Map(right) = &items[1] else {
        panic!("right map")
    };

    let State::Collection(Collection::Tensor(left)) = &left[&"left".parse::<tc_ir::Id>().unwrap()]
    else {
        panic!("left tensor")
    };

    let State::Collection(Collection::Tensor(right)) =
        &right[&"right".parse::<tc_ir::Id>().unwrap()]
    else {
        panic!("right tensor")
    };

    left.write_value(&nested_txn, &[0, 0], number_general::Number::from(9_f32))
        .await
        .unwrap();

    assert_eq!(
        tensor_values::<f32>(left, &nested_txn).await,
        vec![9., 2., 3., 4.]
    );

    assert_eq!(
        tensor_values::<f32>(right, &nested_txn).await,
        vec![1., 2., 3., 4.]
    );

    // Cancel after the first collection is decoded and the next tuple stalls.
    let (_pending_root, pending_txn) = TestTxn::new();
    let before = Arc::strong_count(&pending_txn.next);
    let owner = Arc::new(());
    let observer = Arc::downgrade(&owner);
    let (ready, waiting) = tokio::sync::oneshot::channel();
    let mut ready = Some(ready);
    let pending = stream::poll_fn(move |_| {
        let _keep_alive = &owner;
        if let Some(ready) = ready.take() {
            ready.send(()).unwrap();
        }

        std::task::Poll::Pending::<Option<Bytes>>
    });

    let prefix = format!("[{},[", std::str::from_utf8(&encoded).unwrap());
    let source = stream::iter([Bytes::from(prefix)]).chain(pending);
    let mut decoding = Box::pin(destream_json::decode::<_, TestState>(
        pending_txn.clone(),
        source,
    ));

    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        tokio::select! {
            result = &mut decoding => panic!("unexpected completion: {result:?}"),
            ready = waiting => ready.unwrap(),
        }
    })
    .await
    .unwrap();

    assert!(observer.upgrade().is_some());
    drop(decoding);
    assert!(observer.upgrade().is_none());
    assert_eq!(Arc::strong_count(&pending_txn.next), before);
}

#[tokio::test]
async fn tensor_view_streams_logical_size_above_host_materialization_limit() {
    let (_root, txn) = TestTxn::new();
    let txn = txn.with_tensor_limit(16 * 1024);
    use tc_collection::StorageContext;

    let schema = tc_collection::tensor::TensorSchema::new(
        NumberType::Float(FloatType::F32),
        vec![8192].into(),
        None,
    )
    .unwrap();

    let tensor = Tensor::<TestTxn>::create(
        txn.subcontext_unique().context().await.unwrap(),
        schema,
        4096,
    )
    .await
    .unwrap();

    let state = TestState::Collection(Collection::Tensor(tensor));
    let encoded =
        destream_json::encode(state.into_view(txn).await.expect("streamed state view")).unwrap();

    assert!(
        encoded
            .try_fold(0usize, |n, chunk| async move { Ok(n + chunk.len()) })
            .await
            .unwrap()
            > 8192
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

    let (_view_root, view_txn) = TestTxn::new();
    let encoded = encode_json(state.into_view(view_txn).await.expect("state view")).await;
    let text = String::from_utf8(encoded.clone()).expect("utf-8");
    assert!(text.starts_with('{'));
    assert!(!text.contains("/state/scalar/map"));
    assert!(text.contains("\"status\""));
    assert!(text.contains("\"count\""));

    let (_decode_root, decode_txn) = TestTxn::new();
    let decoded: TestState = decode_json(decode_txn, encoded)
        .await
        .expect("decode state");

    assert!(matches!(&decoded, State::Map(_)));
}

#[tokio::test]
async fn state_scalar_ref_serializes() {
    let state = TestState::Scalar(Scalar::from(tc_ir::TCRef::Id(
        "$foo".parse().expect("IdRef"),
    )));

    let (_view_root, view_txn) = TestTxn::new();
    let encoded = encode_json(state.into_view(view_txn).await.expect("state view")).await;
    let text = String::from_utf8(encoded).expect("utf-8");
    assert_eq!(text, r#"{"$foo":[]}"#);
}

#[path = "stack_tests.rs"]
mod stack_tests;
