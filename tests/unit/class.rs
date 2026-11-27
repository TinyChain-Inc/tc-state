use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use bytes::Bytes;
use futures::{stream, TryStreamExt};
use tc_value::Value;

use super::*;

type TestState = State<crate::state::tests::TestTxn>;

fn id(value: &str) -> Id {
    value.parse().expect("Id")
}

fn scalar(value: u64) -> Scalar {
    Scalar::from(Value::from(value))
}

fn class_link(name: &str) -> Link {
    format!("/class/example-devco/{name}/1.0.0")
        .parse()
        .expect("Class identity")
}

fn class(name: &str, parent: ClassParent, members: &[(&str, Scalar)]) -> ClassDef {
    let prototype = members
        .iter()
        .cloned()
        .map(|(name, value)| (id(name), value))
        .collect();

    ClassDef::from_body(class_link(name), ClassBody::new(parent, prototype))
}

fn instance(
    class: ClassDef,
    members: &[(&str, TestState)],
) -> ClassInstance<crate::state::tests::TestTxn> {
    let members = members
        .iter()
        .cloned()
        .map(|(name, value)| (id(name), value))
        .collect();

    ClassInstance::new(TestState::from(crate::State::None), class, members)
}

#[test]
fn instance_then_prototype_then_parent_then_native() {
    let base = class(
        "base",
        ClassParent::Native(StateType::Tuple),
        &[("base", scalar(2)), ("overridden", scalar(3))],
    );

    let derived = class(
        "derived",
        ClassParent::Class(class_link("base")),
        &[("prototype", scalar(4)), ("overridden", scalar(5))],
    );

    let instance = instance(
        derived,
        &[
            ("instance", TestState::from(6_u64)),
            ("overridden", TestState::from(7_u64)),
        ],
    );

    let classes = BTreeMap::from([(class_link("base"), base)]);

    assert_eq!(
        instance
            .resolve_member(&id("instance"), &classes, |_, _, _| None)
            .expect("instance member")
            .source(),
        MemberSource::Instance
    );

    assert_eq!(
        instance
            .resolve_member(&id("prototype"), &classes, |_, _, _| None)
            .expect("prototype member")
            .source(),
        MemberSource::Prototype
    );

    assert_eq!(
        instance
            .resolve_member(&id("base"), &classes, |_, _, _| None)
            .expect("inherited member")
            .source(),
        MemberSource::Prototype
    );

    assert_eq!(
        instance
            .resolve_member(&id("overridden"), &classes, |_, _, _| None)
            .expect("override")
            .source(),
        MemberSource::Instance
    );

    let native_value = TestState::from(8_u64);
    let resolved = instance
        .resolve_member(&id("native"), &classes, |parent, _, _| {
            assert_eq!(parent, Some(&StateType::Tuple));
            Some(&native_value)
        })
        .expect("native member");

    assert_eq!(resolved.source(), MemberSource::Native);
}

#[test]
fn prototype_method_is_bound_to_exact_instance() {
    let method = tc_ir::OpDef::Post(vec![(id("result"), Scalar::default())]);
    let class = class(
        "class",
        ClassParent::Native(StateType::Tuple),
        &[("call", Scalar::Op(method))],
    );

    let instance = instance(class, &[]);
    let classes = BTreeMap::new();
    let resolved = instance
        .resolve_member(&id("call"), &classes, |_, _, _| None)
        .expect("bound method");

    let ResolvedMember::BoundMethod {
        declared_by,
        instance: bound,
        ..
    } = resolved
    else {
        panic!("expected bound method");
    };

    assert_eq!(declared_by, instance.class().identity());
    assert!(std::ptr::eq(bound, &instance));
}

#[test]
fn inherited_method_reports_its_declaring_class() {
    let parent = class(
        "parent",
        ClassParent::Native(StateType::Tuple),
        &[(
            "call",
            Scalar::Op(tc_ir::OpDef::Post(vec![(id("result"), Scalar::default())])),
        )],
    );

    let child = class("child", ClassParent::Class(parent.identity().clone()), &[]);
    let instance = instance(child, &[]);
    let classes = BTreeMap::from([(parent.identity().clone(), parent.clone())]);
    let ResolvedMember::BoundMethod { declared_by, .. } = instance
        .resolve_member(&id("call"), &classes, |_, _, _| None)
        .expect("inherited method")
    else {
        panic!("expected bound method");
    };

    assert_eq!(declared_by, parent.identity());
}

#[test]
fn invalid_parent_cycle_depth_and_missing_member_are_typed() {
    let derived = class("derived", ClassParent::Class(class_link("missing")), &[]);
    let invalid_instance = instance(derived, &[]);
    assert!(matches!(
        invalid_instance.resolve_member(&id("x"), &BTreeMap::new(), |_, _, _| None),
        Err(ClassError::InvalidParent(_))
    ));

    let a = class("a", ClassParent::Class(class_link("b")), &[]);
    let b = class("b", ClassParent::Class(class_link("a")), &[]);
    let cycle_instance = instance(a.clone(), &[]);
    let classes = BTreeMap::from([(class_link("a"), a), (class_link("b"), b)]);
    assert!(matches!(
        cycle_instance.resolve_member(&id("x"), &classes, |_, _, _| None),
        Err(ClassError::InheritanceCycle)
    ));

    let base = class("base", ClassParent::Native(StateType::Tuple), &[]);
    let middle = class("middle", ClassParent::Class(class_link("base")), &[]);
    let top = class("top", ClassParent::Class(class_link("middle")), &[]);
    let depth_instance = instance(top, &[]);
    let classes = BTreeMap::from([(class_link("base"), base), (class_link("middle"), middle)]);
    assert!(matches!(
        depth_instance.resolve_member_with_limit(&id("x"), &classes, |_, _, _| None, 2),
        Err(ClassError::InheritanceDepthExceeded { limit: 2 })
    ));

    let base = class("base", ClassParent::Native(StateType::Tuple), &[]);
    let instance = instance(base, &[]);
    assert!(matches!(
        instance.resolve_member(&id("x"), &BTreeMap::new(), |_, _, _| None),
        Err(ClassError::MissingMember { member }) if member == id("x")
    ));
}

#[test]
fn lookup_does_not_project_parent() {
    let first = class(
        "stable",
        ClassParent::Native(StateType::Tuple),
        &[("x", scalar(1))],
    );

    let instance = instance(first, &[]);
    let native_calls = AtomicUsize::new(0);
    instance
        .resolve_member(&id("x"), &BTreeMap::new(), |_, _, _| {
            native_calls.fetch_add(1, Ordering::SeqCst);
            None
        })
        .expect("prototype member");

    assert_eq!(native_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn validation_accepts_same_kind_override_and_rejects_kind_change() {
    let base = class(
        "base",
        ClassParent::Native(StateType::Tuple),
        &[("x", scalar(1))],
    );

    let classes = BTreeMap::from([(class_link("base"), base)]);
    class(
        "valid",
        ClassParent::Class(class_link("base")),
        &[("x", scalar(2))],
    )
    .validate(&classes)
    .expect("same-kind override");

    let invalid = class(
        "invalid",
        ClassParent::Class(class_link("base")),
        &[(
            "x",
            Scalar::Op(tc_ir::OpDef::Post(vec![(id("result"), Scalar::default())])),
        )],
    );

    assert!(matches!(
        invalid.validate(&classes),
        Err(ClassError::UnsupportedOverride { member }) if member == id("x")
    ));
}

#[test]
fn batch_validation_accepts_inherited_prototypes() {
    let base_id: Link = "/class/example-devco/base/1.0.0"
        .parse()
        .expect("base identity");

    let derived_id: Link = "/class/example-devco/derived/1.0.0"
        .parse()
        .expect("derived identity");

    let base = ClassDef::from_body(
        base_id.clone(),
        ClassBody::new(
            ClassParent::Native(StateType::Tuple),
            [(id("value"), scalar(1))].into_iter().collect(),
        ),
    );

    let derived = ClassDef::from_body(
        derived_id.clone(),
        ClassBody::new(ClassParent::Class(base_id.clone()), Map::new()),
    );

    let classes = [(base_id, base), (derived_id.clone(), derived)]
        .into_iter()
        .collect();

    validate_classes(&classes, [derived_id]).expect("valid inherited prototype");
}

#[tokio::test]
async fn validated_class_codec_is_symmetric_and_digest_checked() {
    let identity: Link = "/class/example-devco/vector/1.0.0"
        .parse()
        .expect("identity");

    let parent = ClassParent::Native(StateType::Tuple);
    let prototype: Map<Scalar> = [(id("dimensions"), scalar(3))].into_iter().collect();
    let body = ClassBody::new(parent, prototype);
    let definition = ClassDef::from_body(identity, body);
    let encoded = destream_json::encode(definition.clone()).expect("encode");
    let decoded: ClassDef = destream_json::try_decode((), encoded)
        .await
        .expect("decode");

    assert_eq!(decoded, definition);
    assert!(decoded.validate_digest().is_ok());

    let invalid = stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(
        br#"{"manifest_version":2,"identity":"/class/example-devco/vector/1.0.0","digest":["sha256","4c24b66b4b5a80be4e96597c5b70d653dc223fbe2d505b9c865f10c8f14c6e91"],"parent":"/state/tuple","prototype":{}}"#,
    ))]);

    assert!(destream_json::try_decode::<_, _, ClassDef>((), invalid)
        .await
        .is_err());

    let mismatched = stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(
        br#"{"identity":"/class/example-devco/vector/1.0.0","digest":"0000000000000000000000000000000000000000000000000000000000000000","parent":"/state/scalar/tuple","prototype":{"dimensions":3}}"#,
    ))]);

    assert!(destream_json::try_decode::<_, _, ClassDef>((), mismatched)
        .await
        .is_err());
}

#[test]
fn class_hash_composes_its_link_and_bare_body() {
    let body = ClassBody::new(
        ClassParent::Native(StateType::Tuple),
        [(id("dimensions"), scalar(3))].into_iter().collect(),
    );

    let first = ClassDef::from_body(
        "/class/example-devco/vector/1.0.0".parse().unwrap(),
        body.clone(),
    );

    let same = ClassDef::from_body(first.identity().clone(), body.clone());
    let other = ClassDef::from_body("/class/example-devco/vector/2.0.0".parse().unwrap(), body);
    assert_eq!(
        AsyncHash::<async_hash::Sha256>::hash(&first),
        AsyncHash::<async_hash::Sha256>::hash(&same)
    );

    assert_ne!(
        AsyncHash::<async_hash::Sha256>::hash(&first),
        AsyncHash::<async_hash::Sha256>::hash(&other)
    );
}

#[test]
fn class_scalar_uses_the_v1_parent_or_post_form() {
    let parent: Link = "/class/example-devco/base/1.0.0".parse().unwrap();
    let empty: Scalar = ClassBody::new(ClassParent::Class(parent.clone()), Map::new()).into();
    assert_eq!(empty, Scalar::from(tc_value::Value::Link(parent.clone())));

    let prototype: Map<Scalar> = [(id("dimensions"), scalar(3))].into_iter().collect();
    let populated: Scalar =
        ClassBody::new(ClassParent::Class(parent.clone()), prototype.clone()).into();

    assert_eq!(
        populated,
        Scalar::from(tc_ir::TCRef::Op(tc_ir::OpRef::Post((
            tc_ir::Subject::Link(parent),
            prototype,
        ))))
    );
}

#[tokio::test]
async fn bare_class_body_fixture_is_symmetric_and_rejects_metadata() {
    let valid = stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(include_bytes!(
        "../../fixtures/class_definition.json"
    )))]);

    let definition: ClassBody = destream_json::try_decode((), valid)
        .await
        .expect("Class fixture");

    let encoded = destream_json::encode(definition).expect("encode");
    let bytes = encoded
        .try_collect::<Vec<_>>()
        .await
        .expect("collect")
        .concat();

    assert_eq!(
        String::from_utf8(bytes).expect("utf8"),
        include_str!("../../fixtures/class_definition.json").trim_end()
    );

    let invalid = stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(include_bytes!(
        "../../fixtures/class_definition_invalid.json"
    )))]);

    assert!(destream_json::try_decode::<_, _, ClassBody>((), invalid)
        .await
        .is_err());
}
