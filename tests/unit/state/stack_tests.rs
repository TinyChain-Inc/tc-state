use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

use futures::{FutureExt, Stream, StreamExt};
use tc_ir::Scalar;

use super::*;

fn wire(depth: usize, map: bool, binary: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    for _ in 0..depth {
        bytes.extend_from_slice(if map {
            if binary {
                b"{\"k\""
            } else {
                b"{\"k\":"
            }
        } else {
            b"["
        });
    }

    bytes.extend_from_slice(if binary { b"\x01" } else { b"null" });
    bytes.extend(std::iter::repeat_n(if map { b'}' } else { b']' }, depth));
    bytes
}

fn assert_depth(state: &TestState, depth: usize, map: bool) {
    let mut current = state;
    for _ in 0..depth {
        current = match &current {
            State::Map(values) if map => {
                assert_eq!(values.len(), 1);
                values.values().next().unwrap()
            }
            State::Tuple(values) if !map => {
                assert_eq!(values.len(), 1);
                &values[0]
            }
            _ => panic!("unexpected container"),
        };
    }

    assert!(current.is_none());
}

#[tokio::test]
async fn typed_states_take_precedence_and_invocations_delegate_to_ir() {
    use tc_value::class::NativeClass;

    let (_root, txn) = TestTxn::new();
    let string_path = tc_value::ValueType::String.path();
    let get = tc_ir::OpRef::Get((
        tc_ir::Subject::Link(string_path.clone().into()),
        Scalar::default(),
    ));

    let wire = destream_json::encode(&get).unwrap();
    let state: TestState = destream_json::try_decode(txn.clone(), wire).await.unwrap();
    assert!(matches!(state, State::Scalar(Scalar::Ref(value)) if *value == tc_ir::TCRef::Op(get)));

    for (input, accepted) in [
        (
            format!(r#"{{"{string_path}":"https://example.com/"}}"#),
            true,
        ),
        (format!(r#"{{"{string_path}":[null]}}"#), false),
        (r#"{"https://example.com/run":[null]}"#.to_string(), true),
        (r#"{"$input":[]}"#.to_string(), true),
        (
            format!(r#"{{"{}":null}}"#, tc_value::ValueType::Tuple.path()),
            false,
        ),
    ] {
        let input = stream::iter([Bytes::copy_from_slice(input.as_bytes())]);
        let result = destream_json::decode::<_, TestState>(txn.clone(), input).await;
        assert_eq!(result.is_ok(), accepted, "{result:?}");
    }
}

#[tokio::test]
async fn state_maps_and_tagged_records_reject_ambiguous_entries() {
    use tc_value::class::NativeClass;

    let (_root, txn) = TestTxn::new();
    let tuple = crate::StateType::Tuple.path();
    let string = tc_value::ValueType::String.path();
    for binary in [false, true] {
        for json in [
            r#"{"x":null,"x":[]}"#.to_owned(),
            format!(r#"{{"{tuple}":[],"ignored":[]}}"#),
            format!(r#"{{"{string}":"value","ignored":[]}}"#),
        ] {
            let bytes = if binary {
                json.replace("\":", "\"")
                    .replace(',', "")
                    .replace("null", "\u{1}")
                    .into_bytes()
            } else {
                json.into_bytes()
            };

            let source = futures::stream::iter([Bytes::from(bytes)]);
            let result = if binary {
                tbon::de::decode::<_, TestState>(txn.clone(), source)
                    .await
                    .map_err(|e| e.to_string())
            } else {
                destream_json::decode::<_, TestState>(txn.clone(), source)
                    .await
                    .map_err(|e| e.to_string())
            };

            assert!(result.is_err());
        }

        for json in [
            r#"{"x":null,"x":["#.to_owned(),
            format!(r#"{{"{tuple}":[],"ignored":["#),
        ] {
            let prefix = if binary {
                json.replace("\":", "\"")
                    .replace(',', "")
                    .replace("null", "\u{1}")
                    .into_bytes()
            } else {
                json.into_bytes()
            };

            let owner = Arc::new(());
            let released = Arc::downgrade(&owner);
            let source = futures::stream::iter([Bytes::from(prefix)])
                .chain(futures::stream::pending())
                .inspect(move |_| {
                    let _ = &owner;
                });

            let decode = async {
                if binary {
                    tbon::de::decode::<_, TestState>(txn.clone(), source)
                        .await
                        .map_err(|e| e.to_string())
                } else {
                    destream_json::decode::<_, TestState>(txn.clone(), source)
                        .await
                        .map_err(|e| e.to_string())
                }
            };

            assert!(decode
                .now_or_never()
                .expect("reject without reading unwanted payload")
                .is_err());

            assert!(released.upgrade().is_none());
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn canonical_state_depth_and_round_trip_on_normal_workers() {
    tokio::spawn(async {
        let (_root, txn) = TestTxn::new();
        for binary in [false, true] {
            for map in [false, true] {
                for chunk in [1, 4096] {
                    for depth in [
                        2,
                        tc_ir::MAX_SCALAR_DEPTH,
                        tc_ir::MAX_SCALAR_DEPTH + 1,
                        64,
                        16_384,
                    ] {
                        let bytes = wire(depth, map, binary);
                        let source = stream::iter(bytes.chunks(chunk).map(Bytes::copy_from_slice));
                        let result = if binary {
                            tbon::de::decode::<_, TestState>(txn.clone(), source)
                                .await
                                .map_err(|error| error.to_string())
                        } else {
                            destream_json::decode::<_, TestState>(txn.clone(), source)
                                .await
                                .map_err(|error| error.to_string())
                        };

                        if depth > tc_ir::MAX_SCALAR_DEPTH {
                            assert!(result.is_err());
                            continue;
                        }

                        let state = result.unwrap();
                        assert_depth(&state, depth, map);
                        assert!(state.within_depth(depth));
                        assert!(!state.within_depth(depth - 1));
                        let view = state.into_view(txn.clone()).await.unwrap();
                        let encoded = if binary {
                            tbon::en::encode(view)
                                .unwrap()
                                .try_fold(Vec::new(), |mut out, bytes| async move {
                                    out.extend_from_slice(&bytes);
                                    Ok(out)
                                })
                                .await
                                .unwrap()
                        } else {
                            encode_json(view).await
                        };

                        assert_eq!(encoded, bytes);
                    }
                }
            }
        }
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn canonical_state_structural_lifecycle_on_normal_workers() {
    tokio::spawn(async {
        let (_root, txn) = TestTxn::new();
        let depth = 64;
        for map in [false, true] {
            let mut state = TestState::default();
            for _ in 0..depth {
                state = if map {
                    State::from(Map::from_iter([("k".parse().unwrap(), state)]))
                } else {
                    State::from(vec![state])
                };
            }

            assert_depth(&state, depth, map);
            let cloned = state.clone();
            assert_depth(&cloned, depth, map);
            let scalar = Scalar::opt_cast_from(cloned).unwrap();
            let rebuilt = State::from(scalar.clone());
            assert_depth(&rebuilt, depth, map);
            let resolved = crate::resolve::resolve_scalar(
                scalar,
                &Arc::new(std::collections::HashMap::new()),
                &txn,
                None,
            )
            .await
            .unwrap();

            assert_depth(&resolved, depth, map);
        }
    })
    .await
    .unwrap();
}

struct Stalled {
    first: Option<Bytes>,
    released: Arc<AtomicBool>,
}

impl Stream for Stalled {
    type Item = Bytes;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Bytes>> {
        match self.first.take() {
            Some(bytes) => Poll::Ready(Some(bytes)),
            None => Poll::Pending,
        }
    }
}

impl Drop for Stalled {
    fn drop(&mut self) {
        self.released.store(true, Ordering::SeqCst);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deep_state_failure_cancellation_and_stream_drop_release_ownership() {
    tokio::spawn(async {
        for binary in [false, true] {
            let mut complete = wire(tc_ir::MAX_SCALAR_DEPTH - 1, false, binary);
            complete.insert(0, b'[');
            if !binary {
                complete.push(b',');
            }

            for prefix in [vec![b'['; tc_ir::MAX_SCALAR_DEPTH], complete] {
                for poll in [false, true] {
                    let released = Arc::new(AtomicBool::new(false));
                    let source = Stalled {
                        first: Some(prefix.clone().into()),
                        released: released.clone(),
                    };

                    let (_root, txn) = TestTxn::new();
                    let before = Arc::strong_count(&txn.next);
                    let held = txn.clone();
                    let mut decoding = Box::pin(async move {
                        if binary {
                            tbon::de::decode::<_, TestState>(held, source)
                                .await
                                .map_err(|e| e.to_string())
                        } else {
                            destream_json::decode::<_, TestState>(held, source)
                                .await
                                .map_err(|e| e.to_string())
                        }
                    });

                    if poll {
                        assert!(futures::poll!(&mut decoding).is_pending());
                    }

                    drop(decoding);
                    assert!(released.load(Ordering::SeqCst));
                    assert_eq!(Arc::strong_count(&txn.next), before);
                }

                let (_root, txn) = TestTxn::new();
                let before = Arc::strong_count(&txn.next);
                let source = stream::iter([
                    Ok(Bytes::from(prefix.clone())),
                    Err(std::io::Error::other("injected State source failure")),
                ]);

                let error = if binary {
                    tbon::de::try_decode::<_, _, TestState>(txn.clone(), source)
                        .await
                        .err()
                        .unwrap()
                        .to_string()
                } else {
                    destream_json::try_decode::<_, _, TestState>(txn.clone(), source)
                        .await
                        .err()
                        .unwrap()
                        .to_string()
                };

                assert!(error.contains("injected State source failure"));
                assert_eq!(Arc::strong_count(&txn.next), before);
                let mut malformed = prefix;
                malformed.push(b'!');
                let source = stream::iter([Bytes::from(malformed)]);
                let result = if binary {
                    tbon::de::decode::<_, TestState>(txn.clone(), source)
                        .await
                        .map_err(|e| e.to_string())
                } else {
                    destream_json::decode::<_, TestState>(txn.clone(), source)
                        .await
                        .map_err(|e| e.to_string())
                };

                assert!(result.is_err());
                assert_eq!(Arc::strong_count(&txn.next), before);
            }
        }

        let (_root, txn) = TestTxn::new();
        let mut state = State::default();
        for _ in 0..64 {
            state = State::Tuple(vec![state]);
        }

        let copy = state.clone();
        assert_depth(&copy, 64, false);
        assert!(format!("{copy:?}").contains("Tuple"));
        drop(copy);
        let mut encoded = destream_json::encode(state.into_view(txn).await.unwrap()).unwrap();
        assert!(encoded.next().await.unwrap().is_ok());
        drop(encoded);
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mixed_object_ownership_and_control_flow_at_supported_depth() {
    tokio::spawn(async {
        let (_root, txn) = TestTxn::new();
        let class = ClassDef::from_body(
            "/class/example-devco/stack/1.0.0".parse().unwrap(),
            ClassBody::new(ClassParent::Native(StateType::Tuple), Map::new()),
        );

        for binary in [false, true] {
            for depth in [
                tc_ir::MAX_SCALAR_DEPTH - 2,
                tc_ir::MAX_SCALAR_DEPTH - 1,
                tc_ir::MAX_SCALAR_DEPTH,
            ] {
                let mut state = State::default();
                for _ in 0..depth {
                    let members = [("value".parse().unwrap(), State::from(7_u64))]
                        .into_iter()
                        .collect();

                    state = State::from(Object::Instance(ClassInstance::new(
                        state,
                        class.clone(),
                        members,
                    )));
                }

                let copy = state.clone();
                assert_eq!(
                    state.within_depth(tc_ir::MAX_SCALAR_DEPTH),
                    depth < tc_ir::MAX_SCALAR_DEPTH
                );

                drop(state);
                let view = copy.into_view(txn.clone()).await.unwrap();
                let result = if binary {
                    tbon::de::try_decode::<_, _, TestState>(
                        txn.clone(),
                        tbon::en::encode(view).unwrap(),
                    )
                    .await
                    .map_err(|error| error.to_string())
                } else {
                    decode_json::<TestState>(txn.clone(), encode_json(view).await).await
                };

                // Each instance adds one level; its prototype/members map adds one.
                if depth == tc_ir::MAX_SCALAR_DEPTH {
                    assert!(result.unwrap_err().contains("nesting"));
                    continue;
                }

                let mut state = result.unwrap();
                for _ in 0..depth {
                    let State::Object(object) = state else {
                        panic!("instance")
                    };

                    let Object::Instance(instance) = *object else {
                        panic!("instance")
                    };

                    assert_eq!(instance.class(), &class);
                    assert_eq!(instance.members().len(), 1);
                    let (parent, _, _) = instance.into_parts();
                    state = parent;
                }

                assert!(state.is_none());
            }
        }

        let mut reference = tc_ir::TCRef::Id("$truth".parse().unwrap());
        for _ in 0..64 {
            reference = tc_ir::TCRef::Cond(Box::new(tc_ir::Cond::new(
                reference,
                Scalar::from(Value::from(true)),
                Scalar::from(Value::from(false)),
            )));
        }

        let values = Arc::new(
            [("truth".parse().unwrap(), State::from(Value::from(true)))]
                .into_iter()
                .collect(),
        );

        let resolved = crate::resolve::resolve_scalar(Scalar::from(reference), &values, &txn, None)
            .await
            .unwrap();

        assert_eq!(Value::opt_cast_from(resolved), Some(Value::from(true)));
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn configured_limits_and_wide_state_consumption_are_independent_of_stack_depth() {
    async fn check_limit<T: de::FromStream>(
        context: T::Context,
        depth: usize,
        limit: usize,
        accepted: bool,
    ) {
        let source = stream::iter([Bytes::from(wire(depth, false, true))]);
        let result = tbon::de::decode_with_max_depth::<_, T>(context, source, limit).await;
        assert_eq!(result.is_ok(), accepted);
    }

    tokio::spawn(async {
        for limit in [1, 64, 4096] {
            for depth in [limit, limit + 1] {
                // Raising the codec limit cannot raise Value's native tuple limit.
                check_limit::<Value>(
                    (),
                    depth,
                    limit,
                    depth <= limit.min(tc_value::MAX_VALUE_DEPTH),
                )
                .await;

                check_limit::<Scalar>(
                    (),
                    depth,
                    limit,
                    depth <= limit.min(tc_ir::MAX_SCALAR_DEPTH),
                )
                .await;

                let (_root, txn) = TestTxn::new();
                check_limit::<TestState>(
                    txn,
                    depth,
                    limit,
                    depth <= limit.min(tc_ir::MAX_SCALAR_DEPTH),
                )
                .await;
            }
        }

        for map in [false, true] {
            let scalar = if map {
                Scalar::Map(
                    (0..8192)
                        .map(|i| (format!("k{i}").parse().unwrap(), Scalar::default()))
                        .collect(),
                )
            } else {
                Scalar::Tuple((0..65_536).map(|_| Scalar::default()).collect())
            };

            for binary in [false, true] {
                let (_root, txn) = TestTxn::new();
                let view = TestState::from(scalar.clone())
                    .into_view(txn.clone())
                    .await
                    .unwrap();

                let decoded = if binary {
                    let encoded = tbon::en::encode(view).unwrap();
                    tbon::de::try_decode::<_, _, TestState>(txn, encoded)
                        .await
                        .unwrap()
                } else {
                    let encoded = destream_json::encode(view).unwrap();
                    destream_json::try_decode::<_, _, TestState>(txn, encoded)
                        .await
                        .unwrap()
                };

                assert_eq!(Scalar::opt_cast_from(decoded), Some(scalar.clone()));
            }
        }
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deep_object_routes_preserve_all_verbs_and_terminal_observations() {
    use tc_ir::{OpDef, Public, Route};

    tokio::spawn(async {
        let (_root, txn) = TestTxn::new();
        let form = vec![("result".parse().unwrap(), Scalar::from(42_u64))];
        let operations = [
            ("read", OpDef::Get(("key".parse().unwrap(), form.clone()))),
            (
                "write",
                OpDef::Put((
                    "key".parse().unwrap(),
                    "value".parse().unwrap(),
                    form.clone(),
                )),
            ),
            ("call", OpDef::Post(form.clone())),
            ("remove", OpDef::Delete(("key".parse().unwrap(), form))),
        ];

        let leaf = ClassDef::from_body(
            "/class/example-devco/routing-leaf/1.0.0".parse().unwrap(),
            ClassBody::new(
                ClassParent::Native(StateType::Tuple),
                operations
                    .iter()
                    .map(|(name, op)| (name.parse().unwrap(), Scalar::Op(op.clone())))
                    .collect(),
            ),
        );

        let wrapper = ClassDef::from_body(
            "/class/example-devco/routing-wrapper/1.0.0"
                .parse()
                .unwrap(),
            ClassBody::new(ClassParent::Native(StateType::Tuple), Map::new()),
        );

        for mixed in [false, true] {
            let marker: Arc<[u8]> = vec![1_u8; 16].into();
            let mut state = State::from(Object::Instance(ClassInstance::new(
                State::from(Object::Class(leaf.clone())),
                leaf.clone(),
                Map::from_iter([
                    ("literal".parse().unwrap(), State::from(7_u64)),
                    (
                        "held".parse().unwrap(),
                        State::from(Value::Bytes(marker.clone())),
                    ),
                ]),
            )));

            let mut path = Vec::new();
            for level in 0..62 {
                state = match (mixed, level % 3) {
                    (true, 1) => {
                        path.push("hop".parse().unwrap());
                        State::Map(Map::from_iter([("hop".parse().unwrap(), state)]))
                    }
                    (true, 2) => {
                        path.push("hop".parse().unwrap());
                        State::from(Object::Instance(ClassInstance::new(
                            State::default(),
                            wrapper.clone(),
                            Map::from_iter([("hop".parse().unwrap(), state)]),
                        )))
                    }
                    _ => State::from(Object::Instance(ClassInstance::new(
                        state,
                        wrapper.clone(),
                        Map::new(),
                    ))),
                };
            }

            // Empty-path parent routing selects the Class constructor without
            // recursively calling Object::route through every stored parent.
            if !mixed {
                assert!(state.route(&[]).unwrap().get().is_some());
                assert!(state.route(&[]).unwrap().post().is_some());
            }

            path.push("read".parse().unwrap());
            let held = Arc::strong_count(&marker);
            let unpolled = state.get(&txn, &path, Scalar::default());
            assert!(Arc::strong_count(&marker) > held);
            drop(unpolled);
            assert_eq!(Arc::strong_count(&marker), held);
            let read = state.get(&txn, &path, Scalar::default()).await.unwrap();
            assert_eq!(
                Scalar::opt_cast_from(read),
                Some(Scalar::Op(operations[0].1.clone()))
            );

            *path.last_mut().unwrap() = "write".parse().unwrap();
            state
                .put(&txn, &path, Scalar::default(), State::from(3_u64))
                .await
                .unwrap();

            *path.last_mut().unwrap() = "call".parse().unwrap();
            let called = state.post(&txn, &path, Map::new()).await.unwrap();
            assert_eq!(
                Scalar::opt_cast_from(called),
                Some(Scalar::Op(operations[2].1.clone()))
            );

            *path.last_mut().unwrap() = "remove".parse().unwrap();
            state.delete(&txn, &path, Scalar::default()).await.unwrap();
            *path.last_mut().unwrap() = "literal".parse().unwrap();
            let value = state.get(&txn, &path, Scalar::from(99_u64)).await.unwrap();
            assert_eq!(Value::opt_cast_from(value), Some(Value::from(7_u64)));
            *path.last_mut().unwrap() = "read".parse().unwrap();
            let error = state.post(&txn, &path, Map::new()).await.unwrap_err();
            assert_eq!(error.code(), tc_error::ErrorKind::MethodNotAllowed);
            assert!(error.to_string().contains("Class method"));
            path.push("tail".parse().unwrap());
            let error = state.get(&txn, &path, Scalar::default()).await.unwrap_err();
            assert_eq!(error.code(), tc_error::ErrorKind::NotFound);
            assert!(error.to_string().contains("/read/tail"));
            assert_eq!(Arc::strong_count(&marker), held);
            // A map's terminal member is an observation, including Objects.
            let object = State::from(Object::Instance(ClassInstance::new(
                state,
                wrapper.clone(),
                Map::new(),
            )));

            let map = State::Map(Map::from_iter([("object".parse().unwrap(), object)]));
            let path = ["object".parse().unwrap()];
            let observed = map.get(&txn, &path, Scalar::default()).await.unwrap();
            assert!(matches!(&observed, State::Object(_)));
            assert_eq!(
                map.put(&txn, &path, Scalar::default(), State::default())
                    .await
                    .unwrap_err()
                    .code(),
                tc_error::ErrorKind::MethodNotAllowed
            );

            drop(observed);
            drop(map);
            assert_eq!(Arc::strong_count(&marker), 1);
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn iterative_object_routes_preserve_delegated_diagnostic_paths() {
    use tc_ir::{OpDef, Public};

    let (_root, txn) = TestTxn::new();
    let method = State::from(Scalar::Op(OpDef::Post(vec![(
        "result".parse().unwrap(),
        Scalar::default(),
    )])));

    let nested = State::Map(Map::from_iter([
        ("b".parse().unwrap(), State::from(1_u64)),
        ("method".parse().unwrap(), method),
    ]));

    let map = State::Map(Map::from_iter([("a".parse().unwrap(), nested)]));
    let class = ClassDef::from_body(
        "/class/example-devco/routing-diagnostic/1.0.0"
            .parse()
            .unwrap(),
        ClassBody::new(ClassParent::Native(StateType::Tuple), Map::new()),
    );

    for member in [false, true] {
        let instance = if member {
            ClassInstance::new(
                State::default(),
                class.clone(),
                Map::from_iter([("inside".parse().unwrap(), map.clone())]),
            )
        } else {
            ClassInstance::new(map.clone(), class.clone(), Map::new())
        };

        let state = State::from(Object::Instance(instance));
        for suffix in ["/a/b/missing", "/a/method/tail"] {
            let routed = format!("{}{suffix}", if member { "/inside" } else { "" });
            let path: pathlink::PathBuf = routed.parse().unwrap();
            let error = state.get(&txn, &path, Scalar::default()).await.unwrap_err();
            assert_eq!(error.code(), tc_error::ErrorKind::NotFound);
            assert!(error.to_string().contains(suffix), "{error}");
        }

        let routed = if member {
            "/inside/a/b/add"
        } else {
            "/a/b/add"
        };

        let path: pathlink::PathBuf = routed.parse().unwrap();
        let error = state
            .delete(&txn, &path, Scalar::default())
            .await
            .unwrap_err();

        assert_eq!(error.code(), tc_error::ErrorKind::MethodNotAllowed);
        assert!(error.to_string().contains("/a/b/add"), "{error}");
    }

    // An instance member does not bypass the pre-existing ancestry check.
    let inherited = ClassDef::from_body(
        "/class/example-devco/child/1.0.0".parse().unwrap(),
        ClassBody::new(
            ClassParent::Class("/class/example-devco/unresolved/1.0.0".parse().unwrap()),
            Map::new(),
        ),
    );

    let marker: Arc<[u8]> = vec![2_u8; 16].into();
    let state = State::from(Object::Instance(ClassInstance::new(
        State::default(),
        inherited,
        Map::from_iter([(
            "held".parse().unwrap(),
            State::from(Value::Bytes(marker.clone())),
        )]),
    )));

    let held = Arc::strong_count(&marker);
    let error = state
        .get(&txn, &["held".parse().unwrap()], Scalar::default())
        .await
        .unwrap_err();

    assert!(error.to_string().contains("test Class"));
    assert_eq!(Arc::strong_count(&marker), held);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nested_state_codec_delegation_preserves_remaining_depth() {
    tokio::spawn(async {
        let (_root, txn) = TestTxn::new();
        for binary in [false, true] {
            for (outer, payload, accepted) in [
                (
                    tc_ir::MAX_SCALAR_DEPTH / 2,
                    format!(
                        "{{\"/state/scalar/value/tuple\":{}}}",
                        String::from_utf8(wire(tc_ir::MAX_SCALAR_DEPTH / 2, false, false)).unwrap()
                    ),
                    true,
                ),
                (
                    tc_ir::MAX_SCALAR_DEPTH / 2 + 1,
                    format!(
                        "{{\"/state/scalar/value/tuple\":{}}}",
                        String::from_utf8(wire(tc_ir::MAX_SCALAR_DEPTH / 2, false, false)).unwrap()
                    ),
                    false,
                ),
                (
                    tc_ir::MAX_SCALAR_DEPTH - 2,
                    "{\"/state/scalar/op/post\":[[\"result\",null]]}".to_owned(),
                    true,
                ),
                (
                    tc_ir::MAX_SCALAR_DEPTH - 1,
                    "{\"/state/scalar/op/post\":[[\"result\",null]]}".to_owned(),
                    false,
                ),
            ] {
                let json = format!("{}{}{}", "[".repeat(outer), payload, "]".repeat(outer));
                let bytes = if binary {
                    json.replace("\":", "\"")
                        .replace(',', "")
                        .replace("null", "\u{1}")
                        .into_bytes()
                } else {
                    json.into_bytes()
                };

                let source = stream::iter([Bytes::from(bytes)]);
                let result = if binary {
                    tbon::de::decode::<_, TestState>(txn.clone(), source)
                        .await
                        .map_err(|error| error.to_string())
                } else {
                    destream_json::decode::<_, TestState>(txn.clone(), source)
                        .await
                        .map_err(|error| error.to_string())
                };

                assert_eq!(
                    result.is_ok(),
                    accepted,
                    "outer depth {outer}, payload {payload}: {result:?}"
                );
            }

            let mut literal = Scalar::default();
            for _ in 0..tc_ir::MAX_SCALAR_DEPTH / 2 {
                literal = Scalar::Tuple(vec![literal]);
            }

            let class = ClassDef::from_body(
                "/class/example-devco/depth/1.0.0".parse().unwrap(),
                ClassBody::new(
                    ClassParent::Native(StateType::Tuple),
                    Map::from_iter([("payload".parse().unwrap(), literal)]),
                ),
            );

            for depth in [
                tc_ir::MAX_SCALAR_DEPTH / 2 - 2,
                tc_ir::MAX_SCALAR_DEPTH / 2 - 1,
            ] {
                let mut state = State::from(Object::Class(class.clone()));
                for _ in 0..depth {
                    state = State::Tuple(vec![state]);
                }

                let view = state.into_view(txn.clone()).await.unwrap();
                let bytes = if binary {
                    tbon::en::encode(view)
                        .unwrap()
                        .map_ok(|chunk| chunk.to_vec())
                        .try_concat()
                        .await
                        .unwrap()
                } else {
                    encode_json(view).await
                };

                let source = stream::iter([Bytes::from(bytes)]);
                let result = if binary {
                    tbon::de::decode::<_, TestState>(txn.clone(), source)
                        .await
                        .map_err(|error| error.to_string())
                } else {
                    destream_json::decode::<_, TestState>(txn.clone(), source)
                        .await
                        .map_err(|error| error.to_string())
                };

                assert_eq!(
                    result.is_ok(),
                    depth == tc_ir::MAX_SCALAR_DEPTH / 2 - 2,
                    "Class at outer depth {depth}: {result:?}"
                );

                if let Err(error) = result {
                    assert!(error.contains("nesting"), "{error}");
                }
            }
        }
    })
    .await
    .unwrap();
}
