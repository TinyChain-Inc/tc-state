use std::sync::{Arc, Mutex};

use tc_ir::{Transaction, TxnId};

use crate::Gateway;
use crate::state::tests::{TempRoot, TestTxn};
use super::*;

#[derive(Clone)]
struct MockTxn {
    storage: TestTxn,
    read_only: bool,
    outbound: Arc<Mutex<Vec<Link>>>,
    subjects: Arc<Mutex<Vec<Option<State<Self>>>>>,
}

impl MockTxn {
    fn new() -> (Arc<TempRoot>, Self) {
        let (root, storage) = TestTxn::new();
        let txn = Self {
            storage,
            read_only: false,
            outbound: Arc::default(),
            subjects: Arc::default(),
        };

        (root, txn)
    }
}

impl Transaction for MockTxn {
    fn id(&self) -> TxnId {
        self.storage.id()
    }
}

impl tc_collection::StorageContext for MockTxn {
    type File = tc_collection::PersistentFile;

    fn context(
        &self,
    ) -> impl std::future::Future<Output = TCResult<freqfs::DirLock<tc_collection::PersistentFile>>> + Send
    {
        self.storage.context()
    }

    fn subcontext(&self, name: impl Into<String>) -> Self {
        let mut txn = self.clone();
        txn.storage = txn.storage.subcontext(name);
        txn
    }

    fn subcontext_unique(&self) -> Self {
        let mut txn = self.clone();
        txn.storage = txn.storage.subcontext_unique();
        txn
    }

    fn materialized_tensor_bytes(&self) -> usize {
        self.storage.materialized_tensor_bytes()
    }
}

impl Gateway for MockTxn {
    fn check_write(&self) -> TCResult<()> {
        if self.read_only {
            Err(TCError::unauthorized("read-only execution"))
        } else {
            Ok(())
        }
    }

    async fn resolve_class(&self, _identity: &Link) -> TCResult<crate::ClassDef> {
        Err(TCError::not_found("test Class"))
    }

    async fn get(&self, target: Link, _key: Scalar) -> TCResult<State<Self>> {
        self.outbound.lock().expect("outbound calls").push(target);
        Ok(State::from(Value::from(true)))
    }

    async fn put(&self, target: Link, _key: Scalar, _value: State<Self>) -> TCResult<()> {
        self.outbound.lock().expect("outbound calls").push(target);
        Ok(())
    }

    async fn post(&self, target: Link, _params: Map<State<Self>>) -> TCResult<State<Self>> {
        self.outbound.lock().expect("outbound calls").push(target);
        Ok(State::from(Value::from(true)))
    }

    async fn delete(&self, target: Link, _key: Scalar) -> TCResult<()> {
        self.outbound.lock().expect("outbound calls").push(target);
        Ok(())
    }
}

impl OpExecutor for MockTxn {
    async fn execute_op(
        &self,
        _definition: OpDef,
        _args: State<Self>,
        subject: Option<State<Self>>,
        _declared_by: Option<Link>,
    ) -> TCResult<State<Self>> {
        self.subjects.lock().expect("subjects").push(subject);
        Ok(State::None)
    }
}

#[tokio::test]
async fn preserves_outbound_transaction_and_concrete_subject() {
    let (_root, txn) = MockTxn::new();
    let target: Link = "/lib/example-devco/math/1.0.0".parse().expect("target");
    let get = TCRef::Op(OpRef::Get((
        Subject::Link(target.clone()),
        Scalar::default(),
    )));

    fn assert_send_static(_: &(impl Send + 'static)) {}

    let held = Arc::strong_count(&txn.outbound);
    let unpolled = {
        let caller = txn.clone();
        resolve_ref(get.clone(), &caller, None)
    };

    assert_send_static(&unpolled);
    assert_eq!(Arc::strong_count(&txn.outbound), held + 1);
    assert!(txn.outbound.lock().unwrap().is_empty());
    drop(unpolled);
    assert_eq!(Arc::strong_count(&txn.outbound), held);

    resolve_ref(get, &txn, None).await.expect("outbound GET");
    assert_eq!(&*txn.outbound.lock().expect("outbound calls"), &[target]);

    let subject = State::from(Value::from("instance"));
    let cond = TCRef::Cond(Box::new(Cond::new(
        TCRef::Op(OpRef::Get((
            Subject::Link("/lib/example-devco/flag/1.0.0".parse().expect("flag")),
            Scalar::default(),
        ))),
        Scalar::Op(OpDef::Post(vec![(
            "result".parse().unwrap(),
            Scalar::default(),
        )])),
        Scalar::default(),
    )));

    resolve_ref(cond, &txn, Some(subject.clone()))
        .await
        .expect("nested OpDef");

    let subjects = txn.subjects.lock().expect("subjects");
    assert_eq!(subjects.len(), 1);
    assert!(matches!(subjects[0].as_ref(), Some(State::Scalar(_))));
}

#[tokio::test]
async fn read_only_resolution_rejects_effects_but_allows_calculation_and_literals() {
    use tc_collection::StorageContext;
    use tc_value::class::NativeClass;

    let (_root, mut txn) = MockTxn::new();
    let dtype = tc_value::ValueType::Number(tc_value::NumberType::Number);
    let schema = tc_collection::btree::BTreeSchema::try_from(vec![
        tc_collection::btree::BTreeColumnSchema {
            name: "item".into(),
            dtype,
            max_size: None,
        },
    ])
    .unwrap();

    let tree = tc_collection::btree::BTree::<MockTxn>::create(
        txn.subcontext_unique().context().await.unwrap(),
        schema,
    )
    .await
    .unwrap();

    tree.insert_row(&txn, vec![1u64.into()]).await.unwrap();
    let subject = State::from(crate::Collection::BTree(Box::new(tree.clone())));

    let values = HashMap::from([
        ("tree".parse().unwrap(), subject.clone()),
        ("number".parse().unwrap(), State::from(Value::from(3u64))),
    ]);

    let target = Target::new(
        Subject::Ref(
            tc_ir::IdRef::new("number".parse().unwrap()),
            pathlink::PathBuf::new(),
        ),
        &values,
        Some(&subject),
    )
    .unwrap();

    assert!(matches!(target, Target::Borrowed(state, _) if std::ptr::eq(state, &values["number"])));
    txn.read_only = true;

    for receiver in [pathlink::label("self"), pathlink::label("tree")] {
        let put = OpRef::Put((
            Subject::Ref(
                tc_ir::IdRef::new(receiver.into()),
                pathlink::PathBuf::new().append(pathlink::label("insert")),
            ),
            Scalar::default(),
            Scalar::Tuple(vec![Scalar::from(2u64)]),
        ));

        let delete = OpRef::Delete((
            Subject::Ref(tc_ir::IdRef::new(receiver.into()), pathlink::PathBuf::new()),
            Scalar::Tuple(vec![Scalar::from(1u64)]),
        ));

        for operation in [put, delete] {
            let error = resolve_scalar(TCRef::Op(operation).into(), &values, &txn, Some(&subject))
                .await
                .err()
                .unwrap();

            assert_eq!(error.code(), tc_error::ErrorKind::Unauthorized);
            assert!(tree.contains_row(txn.id(), &[1u64.into()]).await.unwrap());
            assert!(!tree.contains_row(txn.id(), &[2u64.into()]).await.unwrap());
        }
    }

    let target = Link::from(pathlink::path_label(&[
        "service", "example", "readonly", "1.0.0",
    ]));

    for operation in [
        OpRef::Put((
            Subject::Link(target.clone()),
            Scalar::default(),
            Scalar::default(),
        )),
        OpRef::Delete((Subject::Link(target), Scalar::default())),
    ] {
        let error = resolve_ref(TCRef::Op(operation), &txn, None)
            .await
            .err()
            .unwrap();

        assert_eq!(error.code(), tc_error::ErrorKind::Unauthorized);
    }

    assert!(txn.outbound.lock().unwrap().is_empty());

    let add = OpRef::Post((
        Subject::Ref(
            tc_ir::IdRef::new(pathlink::label("number").into()),
            pathlink::PathBuf::new().append(pathlink::label("add")),
        ),
        [("r".parse().unwrap(), Scalar::from(2u64))]
            .into_iter()
            .collect(),
    ));

    let result = resolve_scalar(TCRef::Op(add).into(), &values, &txn, None)
        .await
        .unwrap();

    assert_eq!(Value::opt_cast_from(result), Some(5u64.into()));

    let schema = tc_collection::tensor::TensorSchema::new(
        tc_value::NumberType::UInt(number_general::UIntType::U64),
        vec![2].into(),
        None,
    )
    .unwrap();

    let literal = OpRef::Put((
        Subject::Link(
            tc_collection::TensorType
                .path()
                .to_string()
                .parse()
                .unwrap(),
        ),
        Scalar::from(Value::from(schema)),
        Scalar::Tuple(vec![Scalar::from(3u64), Scalar::from(4u64)]),
    ));

    let result = resolve_ref(TCRef::Op(literal), &txn, None).await.unwrap();
    let State::Collection(crate::Collection::Tensor(tensor)) = result else {
        panic!("literal must construct a fresh Tensor");
    };

    assert_eq!(
        tensor.read_value(txn.id(), &[1]).await.unwrap(),
        4u64.into()
    );
}
