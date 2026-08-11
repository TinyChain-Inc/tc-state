use pathlink::{label, path_label, Label, PathBuf, PathLabel, PathSegment};
pub use tc_collection::CollectionType;
use tc_ir::{Class, NativeClass};
use tc_value::ValueType;

const STATE_SCALAR_TUPLE_PATH: PathLabel = path_label(&["state", "scalar", "tuple"]);

const LABEL_STATE: Label = label("state");
const LABEL_SCALAR: Label = label("scalar");
const LABEL_TUPLE: Label = label("tuple");

/// TinyChain state classes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StateType {
    Scalar(ValueType),
    Tuple,
    Collection(CollectionType),
}

impl Class for StateType {}

impl NativeClass for StateType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path_matches(path, &STATE_SCALAR_TUPLE_PATH) {
            return Some(Self::Tuple);
        }

        if let Some(collection) = CollectionType::from_path(path) {
            return Some(Self::Collection(collection));
        }

        ValueType::from_path(path).map(Self::Scalar)
    }

    fn path(&self) -> PathBuf {
        match self {
            Self::Scalar(value_type) => value_type.path(),
            Self::Tuple => PathBuf::new()
                .append(LABEL_STATE)
                .append(LABEL_SCALAR)
                .append(LABEL_TUPLE),
            Self::Collection(collection_type) => collection_type.path(),
        }
    }
}

impl From<ValueType> for StateType {
    fn from(value_type: ValueType) -> Self {
        StateType::Scalar(value_type)
    }
}

impl From<CollectionType> for StateType {
    fn from(collection_type: CollectionType) -> Self {
        StateType::Collection(collection_type)
    }
}

fn path_matches(path: &[PathSegment], expected: &PathLabel) -> bool {
    path.len() == expected[..].len()
        && path
            .iter()
            .enumerate()
            .all(|(i, segment)| segment.as_str() == expected[i])
}

#[cfg(test)]
mod tests {
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
}
