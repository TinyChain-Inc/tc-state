use super::*;

#[test]
fn tuple_path_is_canonical_scalar_tuple() {
    let path = StateType::Tuple.path().to_string();
    assert_eq!(path, "/state/scalar/tuple");
}

#[test]
fn accepts_canonical_tuple_path() {
    let canonical = "/state/scalar/tuple"
        .parse::<PathBuf>()
        .expect("canonical tuple path");

    assert_eq!(
        StateType::from_path(canonical.as_ref()),
        Some(StateType::Tuple)
    );
}

#[test]
fn accepts_canonical_btree_collection_path() {
    let canonical = "/state/collection/btree"
        .parse::<PathBuf>()
        .expect("canonical btree path");

    assert_eq!(
        StateType::from_path(canonical.as_ref()),
        Some(StateType::Collection(CollectionType::BTree(
            tc_collection::BTreeType,
        )))
    );
}
