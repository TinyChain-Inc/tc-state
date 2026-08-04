use pathlink::{Label, PathBuf, PathLabel, PathSegment, label, path_label};
use tc_ir::{Class, NativeClass};
use tc_value::ValueType;

const STATE_COLLECTION_BTREE_PATH: PathLabel = path_label(&["state", "collection", "btree"]);
const STATE_COLLECTION_TABLE_PATH: PathLabel = path_label(&["state", "collection", "table"]);
const STATE_COLLECTION_TENSOR_PATH: PathLabel = path_label(&["state", "collection", "tensor"]);
const STATE_SCALAR_TUPLE_PATH: PathLabel = path_label(&["state", "scalar", "tuple"]);

const LABEL_STATE: Label = label("state");
const LABEL_COLLECTION: Label = label("collection");
const LABEL_BTREE: Label = label("btree");
const LABEL_TABLE: Label = label("table");
const LABEL_SCALAR: Label = label("scalar");
const LABEL_TENSOR: Label = label("tensor");
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

/// TinyChain collection classes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollectionType {
    BTree(BTreeType),
    Table(TableType),
    Tensor(TensorType),
}

impl Class for CollectionType {}

impl NativeClass for CollectionType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if let Some(btree) = BTreeType::from_path(path) {
            return Some(Self::BTree(btree));
        }

        if let Some(table) = TableType::from_path(path) {
            return Some(Self::Table(table));
        }

        TensorType::from_path(path).map(Self::Tensor)
    }

    fn path(&self) -> PathBuf {
        match self {
            Self::BTree(btree) => btree.path(),
            Self::Table(table) => table.path(),
            Self::Tensor(tensor) => tensor.path(),
        }
    }
}

impl From<BTreeType> for CollectionType {
    fn from(btree_type: BTreeType) -> Self {
        CollectionType::BTree(btree_type)
    }
}

impl From<TableType> for CollectionType {
    fn from(table_type: TableType) -> Self {
        CollectionType::Table(table_type)
    }
}

impl From<TensorType> for CollectionType {
    fn from(tensor_type: TensorType) -> Self {
        CollectionType::Tensor(tensor_type)
    }
}

/// Tensor collection class.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TensorType;

impl Class for TensorType {}

impl NativeClass for TensorType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path_matches(path, &STATE_COLLECTION_TENSOR_PATH) {
            Some(Self)
        } else {
            None
        }
    }

    fn path(&self) -> PathBuf {
        PathBuf::new()
            .append(LABEL_STATE)
            .append(LABEL_COLLECTION)
            .append(LABEL_TENSOR)
    }
}

/// The canonical Table collection class.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TableType;

impl Class for TableType {}

impl NativeClass for TableType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path_matches(path, &STATE_COLLECTION_TABLE_PATH) {
            Some(Self)
        } else {
            None
        }
    }

    fn path(&self) -> PathBuf {
        PathBuf::new()
            .append(LABEL_STATE)
            .append(LABEL_COLLECTION)
            .append(LABEL_TABLE)
    }
}

/// BTree collection class.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BTreeType;

impl Class for BTreeType {}

impl NativeClass for BTreeType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path_matches(path, &STATE_COLLECTION_BTREE_PATH) {
            Some(Self)
        } else {
            None
        }
    }

    fn path(&self) -> PathBuf {
        PathBuf::new()
            .append(LABEL_STATE)
            .append(LABEL_COLLECTION)
            .append(LABEL_BTREE)
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
            Some(StateType::Collection(CollectionType::BTree(BTreeType)))
        );
    }
}
