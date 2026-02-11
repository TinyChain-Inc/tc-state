use pathlink::{label, path_label, Label, PathBuf, PathLabel, PathSegment};
use tc_ir::{Class, NativeClass};
use tc_value::ValueType;

const STATE_COLLECTION_TENSOR_PATH: PathLabel = path_label(&["state", "collection", "tensor"]);
const STATE_SCALAR_MAP_PATH: PathLabel = path_label(&["state", "scalar", "map"]);
const STATE_TUPLE_PATH: PathLabel = path_label(&["state", "tuple"]);

const LABEL_STATE: Label = label("state");
const LABEL_COLLECTION: Label = label("collection");
const LABEL_MAP: Label = label("map");
const LABEL_SCALAR: Label = label("scalar");
const LABEL_TENSOR: Label = label("tensor");
const LABEL_TUPLE: Label = label("tuple");

/// Transitional TinyChain state classes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StateType {
    Scalar(ValueType),
    Map,
    Tuple,
    Collection(CollectionType),
}

impl Class for StateType {}

impl NativeClass for StateType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path_matches(path, &STATE_SCALAR_MAP_PATH) {
            return Some(Self::Map);
        }
        if path_matches(path, &STATE_TUPLE_PATH) {
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
            Self::Map => PathBuf::new()
                .append(LABEL_STATE)
                .append(LABEL_SCALAR)
                .append(LABEL_MAP),
            Self::Tuple => PathBuf::new().append(LABEL_STATE).append(LABEL_TUPLE),
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

/// Transitional collection classes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollectionType {
    Tensor(TensorType),
}

impl Class for CollectionType {}

impl NativeClass for CollectionType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        TensorType::from_path(path).map(Self::Tensor)
    }

    fn path(&self) -> PathBuf {
        match self {
            Self::Tensor(tensor) => tensor.path(),
        }
    }
}

impl From<TensorType> for CollectionType {
    fn from(tensor_type: TensorType) -> Self {
        CollectionType::Tensor(tensor_type)
    }
}

/// Transitional tensor class (single variant for now).
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

fn path_matches(path: &[PathSegment], expected: &PathLabel) -> bool {
    path.len() == expected[..].len()
        && path
            .iter()
            .enumerate()
            .all(|(i, segment)| segment.as_str() == expected[i])
}
