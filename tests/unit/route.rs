use safecast::TryCastFrom;
use tc_ir::{Public, Route};
use tc_value::Value;

use crate::Gateway;
use crate::state::tests::TestTxn;
use super::*;

impl Gateway for TestTxn {
    fn check_write(&self) -> tc_error::TCResult<()> {
        Ok(())
    }

    async fn resolve_class(&self, _identity: &Link) -> tc_error::TCResult<ClassDef> {
        Err(tc_error::TCError::not_found("test Class"))
    }

    async fn get(&self, _target: Link, _key: Scalar) -> tc_error::TCResult<State<Self>> {
        Err(tc_error::TCError::not_found("test target"))
    }

    async fn put(
        &self,
        _target: Link,
        _key: Scalar,
        _value: State<Self>,
    ) -> tc_error::TCResult<()> {
        Err(tc_error::TCError::not_found("test target"))
    }

    async fn post(
        &self,
        _target: Link,
        _params: Map<State<Self>>,
    ) -> tc_error::TCResult<State<Self>> {
        Err(tc_error::TCError::not_found("test target"))
    }

    async fn delete(&self, _target: Link, _key: Scalar) -> tc_error::TCResult<()> {
        Err(tc_error::TCError::not_found("test target"))
    }
}

impl OpExecutor for TestTxn {
    async fn execute_op(
        &self,
        definition: OpDef,
        _args: State<Self>,
        _subject: Option<State<Self>>,
        _declared_by: Option<Link>,
    ) -> tc_error::TCResult<State<Self>> {
        Ok(State::Scalar(Scalar::Op(definition)))
    }
}

#[tokio::test]
async fn scalar_attributes_construct_subjects_only_for_methods() {
    use std::cell::Cell;

    use safecast::TryCastFrom;

    use tc_ir::{IdRef, OpRef, Subject, TCRef};

    let (_root, txn) = TestTxn::new();
    let literal = Scalar::from(TCRef::Id(IdRef::new("literal".parse().unwrap())));
    for scalar in [Scalar::default(), Scalar::from(42_u64), literal] {
        let result = route_scalar(&scalar, || panic!("unused subject"))
            .get()
            .unwrap()(&txn, Scalar::default())
        .await
        .unwrap();

        assert_eq!(Scalar::opt_cast_from(result), Some(scalar));
    }

    let reference = Scalar::from(TCRef::Op(OpRef::Get((
        Subject::Link("/lib/test/missing/1.0.0".parse().unwrap()),
        Scalar::default(),
    ))));

    let error = route_scalar(&reference, || panic!("unused subject"))
        .get()
        .unwrap()(&txn, Scalar::default())
    .await
    .unwrap_err();

    assert!(error.to_string().contains("test target"));

    let nested = State::Map(
        [("value".parse().unwrap(), State::from(reference.clone()))]
            .into_iter()
            .collect(),
    );

    let map = State::Map([("nested".parse().unwrap(), nested)].into_iter().collect());
    let result = map
        .get(
            &txn,
            &["nested".parse().unwrap(), "value".parse().unwrap()],
            Scalar::default(),
        )
        .await
        .unwrap();

    assert_eq!(Scalar::opt_cast_from(result), Some(reference));

    let key: tc_ir::Id = "key".parse().unwrap();
    for op in [
        OpDef::Get((key.clone(), vec![])),
        OpDef::Put((key.clone(), "value".parse().unwrap(), vec![])),
        OpDef::Post(vec![]),
        OpDef::Delete((key, vec![])),
    ] {
        let calls = Cell::new(0);
        let handler = route_scalar(&Scalar::Op(op.clone()), || {
            calls.set(calls.get() + 1);
            State::from(42_u64)
        });

        assert_eq!(calls.get(), 1);
        match op {
            OpDef::Get(_) => {
                handler.get().unwrap()(&txn, Scalar::default())
                    .await
                    .unwrap();
            }
            OpDef::Put(_) => {
                handler.put().unwrap()(&txn, Scalar::default(), State::None)
                    .await
                    .unwrap();
            }
            OpDef::Post(_) => {
                handler.post().unwrap()(&txn, Map::new()).await.unwrap();
            }
            OpDef::Delete(_) => {
                handler.delete().unwrap()(&txn, Scalar::default())
                    .await
                    .unwrap();
            }
        }

        assert_eq!(calls.get(), 1);
    }
}

#[tokio::test]
async fn static_routes_native_number_operations() {
    let routes = Static::<TestTxn>::default();
    let path: pathlink::PathBuf = "/scalar/value/number/add".parse().unwrap();
    let handler = routes.route(&path).expect("number route");
    let get = handler.get().expect("GET handler");
    let (_root, txn) = TestTxn::new();
    let result = get(
        &txn,
        Scalar::Tuple(vec![
            Scalar::from(Value::Number(2.into())),
            Scalar::from(Value::Number(3.into())),
        ]),
    )
    .await
    .expect("add");

    assert_eq!(Value::opt_cast_from(result), Some(Value::from(5_u64)));

    let get = routes.route(&path).unwrap().get().unwrap();
    let overflow = get(
        &txn,
        Scalar::Tuple(vec![
            Scalar::from(Value::from(u64::MAX)),
            Scalar::from(Value::from(1u64)),
        ]),
    )
    .await;

    assert_eq!(
        overflow.unwrap_err().code(),
        tc_error::ErrorKind::BadRequest
    );

    let get = routes.route(&path).unwrap().get().unwrap();
    assert!(get(
        &txn,
        Scalar::Tuple(vec![
            Scalar::from(Value::from(1u64)),
            Scalar::from(Value::from(1u64))
        ])
    )
    .await
    .is_ok());

    for (left, right) in [
        (number_general::Number::from(u64::MAX), 1u64.into()),
        (number_general::Number::from(i64::MIN), (-1i64).into()),
    ] {
        let params = [
            ("l".parse().unwrap(), State::from(Value::Number(left))),
            ("r".parse().unwrap(), State::from(Value::Number(right))),
        ]
        .into_iter()
        .collect();

        let error = routes.route(&path).unwrap().post().unwrap()(&txn, params)
            .await
            .unwrap_err();

        assert_eq!(error.code(), tc_error::ErrorKind::BadRequest);
    }
}

#[tokio::test]
async fn native_values_own_their_operations() {
    let (_root, txn) = TestTxn::new();
    let number = State::from(Value::Number(2.into()));
    let mut params = Map::new();
    params.insert("r".parse().unwrap(), State::from(Value::Number(3.into())));
    let add_path = ["add".parse().unwrap()];
    let handler = number.route(&add_path).expect("number handler");
    let sum = handler.post().expect("POST handler")(&txn, params)
        .await
        .expect("add");

    assert_eq!(Value::opt_cast_from(sum), Some(Value::from(5_u64)));

    let maximum = State::from(Value::from(u64::MAX));
    let params = [("r".parse().unwrap(), State::from(1u64))]
        .into_iter()
        .collect();

    let error = maximum.route(&add_path).unwrap().post().unwrap()(&txn, params)
        .await
        .unwrap_err();

    assert_eq!(error.code(), tc_error::ErrorKind::BadRequest);

    let path = pathlink::PathBuf::from(pathlink::path_label(&["example"]));
    let link = State::from(Value::Link(path.to_string().parse().unwrap()));
    let eq_path = ["eq".parse().unwrap()];
    let params = [(
        "r".parse().unwrap(),
        State::from(Value::String(path.to_string())),
    )]
    .into_iter()
    .collect();

    let equal = link.route(&eq_path).unwrap().post().unwrap()(&txn, params)
        .await
        .unwrap();

    assert_eq!(Value::opt_cast_from(equal), Some(Value::from(false)));

    for (left, right, expected) in [
        (
            Value::Link(path.clone().into()),
            Value::Link(path.clone().into()),
            true,
        ),
        (
            Value::from(path.to_string()),
            Value::Link(path.clone().into()),
            false,
        ),
        (
            Value::Tuple(vec![Value::Link(path.clone().into())]),
            Value::Tuple(vec![Value::from(path.to_string())]),
            false,
        ),
        (Value::from(7_u64), Value::from(7_f64), true),
        (Value::from(7_u64), Value::from("7"), false),
    ] {
        let subject = State::from(left);
        let params = [("r".parse().unwrap(), State::from(right))]
            .into_iter()
            .collect();

        let equal = subject.route(&eq_path).unwrap().post().unwrap()(&txn, params)
            .await
            .unwrap();

        assert_eq!(Value::opt_cast_from(equal), Some(Value::from(expected)));
    }

    let template = State::from(Value::from("{{link}} {{number}} {{string}}"));
    let render_path = ["render".parse().unwrap()];
    let params = [
        ("link".parse().unwrap(), link),
        ("number".parse().unwrap(), State::from(7u64)),
        ("string".parse().unwrap(), State::from(Value::from("text"))),
    ]
    .into_iter()
    .collect();

    let rendered = template.route(&render_path).unwrap().post().unwrap()(&txn, params)
        .await
        .unwrap();

    assert_eq!(
        Value::opt_cast_from(rendered),
        Some(Value::String(format!("{path} 7 text")))
    );

    let params = [("link".parse().unwrap(), State::from(Value::None))]
        .into_iter()
        .collect();

    let error = template.route(&render_path).unwrap().post().unwrap()(&txn, params)
        .await
        .unwrap_err();

    assert_eq!(error.code(), tc_error::ErrorKind::BadRequest);

    let tuple = State::Tuple(vec![
        State::from(1_u64),
        State::from(2_u64),
        State::from(3_u64),
    ]);

    let mut params = Map::new();
    params.insert("start".parse().unwrap(), State::from(1_u64));
    params.insert("stop".parse().unwrap(), State::from(3_u64));
    let slice_path = ["slice".parse().unwrap()];
    let handler = tuple.route(&slice_path).expect("tuple handler");
    let slice = handler.post().expect("POST handler")(&txn, params)
        .await
        .expect("slice");

    assert!(matches!(&slice, State::Tuple(items) if items.len() == 2));

    // The native and Scalar tuple representations enforce the same checked index contract.
    for subject in [
        &tuple,
        &State::Scalar(Scalar::Tuple(vec![
            Scalar::from(Value::from(1u64)),
            Scalar::from(Value::from(2u64)),
            Scalar::from(Value::from(3u64)),
        ])),
    ] {
        let lookup_path = ["get".parse().unwrap()];
        for invalid in [
            number_general::Number::from(-1i64),
            0.5.into(),
            f64::NAN.into(),
            f64::INFINITY.into(),
            [1., 1.].into(),
        ] {
            let params = [("i".parse().unwrap(), State::from(Value::Number(invalid)))]
                .into_iter()
                .collect();

            let error = subject.route(&lookup_path).unwrap().post().unwrap()(&txn, params)
                .await
                .unwrap_err();

            assert_eq!(error.code(), tc_error::ErrorKind::BadRequest);
        }

        for valid in [
            number_general::Number::from(1u64),
            1f64.into(),
            [1., 0.].into(),
        ] {
            let params = [("i".parse().unwrap(), State::from(Value::Number(valid)))]
                .into_iter()
                .collect();

            let result = subject.route(&lookup_path).unwrap().post().unwrap()(&txn, params)
                .await
                .unwrap();

            assert_eq!(Value::opt_cast_from(result), Some(Value::from(2u64)));
        }
    }

    for parameter in ["start", "stop"] {
        for invalid in [
            number_general::Number::from(u64::MAX),
            0.5.into(),
            f64::NAN.into(),
            f64::INFINITY.into(),
            [1., 1.].into(),
        ] {
            let mut params: Map<_> = [
                ("start".parse().unwrap(), State::from(0u64)),
                ("stop".parse().unwrap(), State::from(3u64)),
            ]
            .into_iter()
            .collect();

            params.insert(
                parameter.parse().unwrap(),
                State::from(Value::Number(invalid)),
            );

            let error = tuple.route(&slice_path).unwrap().post().unwrap()(&txn, params)
                .await
                .unwrap_err();

            assert_eq!(error.code(), tc_error::ErrorKind::BadRequest);
        }
    }

    let params = [
        (
            "start".parse().unwrap(),
            State::from(Value::Number((-2i64).into())),
        ),
        (
            "stop".parse().unwrap(),
            State::from(Value::Number(3f64.into())),
        ),
    ]
    .into_iter()
    .collect();

    let result = tuple.route(&slice_path).unwrap().post().unwrap()(&txn, params)
        .await
        .unwrap();

    assert!(matches!(result, State::Tuple(items) if items.len() == 2));

    let map = State::Map(
        [("answer".parse().unwrap(), State::from(42_u64))]
            .into_iter()
            .collect(),
    );

    let mut params = Map::new();
    params.insert(
        "i".parse().unwrap(),
        State::from(Value::String("answer".into())),
    );

    let get_path = ["get".parse().unwrap()];
    let handler = map.route(&get_path).expect("map handler");
    let answer = handler.post().expect("POST handler")(&txn, params)
        .await
        .expect("get");

    assert_eq!(Value::opt_cast_from(answer), Some(Value::from(42_u64)));
}

#[tokio::test]
async fn iterative_selection_binds_the_selected_instance_and_declaring_class() {
    let (_root, txn) = TestTxn::new();
    let definition = OpDef::Post(vec![("result".parse().unwrap(), Scalar::default())]);
    let declared: Link = "/class/example-devco/declared/1.0.0".parse().unwrap();
    let class = ClassDef::from_body(
        declared.clone(),
        super::super::ClassBody::new(
            super::super::ClassParent::Native(super::super::StateType::Tuple),
            Map::from_iter([("call".parse().unwrap(), Scalar::Op(definition.clone()))]),
        ),
    );

    let selected = ClassInstance::new(
        State::default(),
        class,
        Map::from_iter([("marker".parse().unwrap(), State::from(42_u64))]),
    );

    let wrapper = ClassDef::from_body(
        "/class/example-devco/wrapper/1.0.0".parse().unwrap(),
        super::super::ClassBody::new(
            super::super::ClassParent::Native(super::super::StateType::Tuple),
            Map::new(),
        ),
    );

    let outer = ClassInstance::new(
        State::from(Object::Instance(selected)),
        wrapper,
        Map::from_iter([("marker".parse().unwrap(), State::from(99_u64))]),
    );

    let Selected::Method(method, _) = (InstanceHandler {
        instance: outer,
        path: vec!["call".parse().unwrap()],
    })
    .select(&txn, false)
    .await
    .unwrap() else {
        panic!("bound method")
    };

    assert_eq!(method.declared_by, Some(declared));
    assert_eq!(method.definition, definition);
    let State::Object(object) = &method.subject else {
        panic!("instance")
    };

    let Object::Instance(instance) = object.as_ref() else {
        panic!("instance")
    };

    assert_eq!(
        Value::opt_cast_from(instance.members()["marker"].clone()),
        Some(Value::from(42_u64))
    );
}

#[test]
fn class_and_instance_are_native_state_routes() {
    let identity: Link = "/class/example-devco/example/1.0.0"
        .parse()
        .expect("class identity");

    let mut prototype = Map::new();
    prototype.insert(
        "call".parse().expect("method name"),
        Scalar::Op(OpDef::Post(vec![(
            "result".parse().unwrap(),
            Scalar::default(),
        )])),
    );

    let class = ClassDef::new(
        identity,
        [1; 32],
        super::super::ClassParent::Native(super::super::StateType::Tuple),
        prototype,
    );

    let class_state = State::<TestTxn>::Object(Box::new(Object::Class(class.clone())));
    assert!(class_state.route(&[]).is_some());

    let instance = ClassInstance::new(State::None, class, Map::new());
    let instance_state = State::<TestTxn>::Object(Box::new(Object::Instance(instance)));
    assert!(instance_state.route(&[]).is_none());
    assert!(instance_state
        .route(&["call".parse().expect("method path")])
        .is_some());
}
