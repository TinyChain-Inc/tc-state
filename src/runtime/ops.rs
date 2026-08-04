use std::ops::Bound;

use futures::{StreamExt, stream::BoxStream};
use number_general::{FloatType, Number, UIntType};
use pathlink::{Link, PathBuf, PathSegment};
use safecast::{CastFrom, CastInto, TryCastFrom};
use tc_error::{TCError, TCResult};
use tc_ir::{Id, Map, NativeClass, Scalar, Transaction, TxnId};
use tc_value::{NumberType, Value, number_type_from_path, number_type_path};

use super::{
    AxisRange, BTreeCollection, BTreeType, Collection, Range, State, TableCollection, TableType,
    Tensor, TensorReduceResult, TensorType,
};
use tc_collection::tensor::{
    batched_matmul, broadcast_add, broadcast_reduce_sum, tensor_op_result, tensor_transpose,
};

type BTreeBounds = (Bound<Value>, Bound<Value>);

/// A format-neutral State result.
///
/// Large collection results are represented as State items so adapters can encode
/// them incrementally without knowing the concrete collection type.
pub enum StateStream {
    State(State),
    Map(Vec<(String, StateStream)>),
    Tuple(Vec<StateStream>),
    Sequence(BoxStream<'static, TCResult<State>>),
}

impl State {
    /// Apply a local GET after the executor has resolved its reference and key.
    pub async fn get(
        &self,
        path: &[PathSegment],
        key: State,
        txn: &dyn Transaction,
    ) -> TCResult<Option<State>> {
        match self {
            State::Collection(Collection::BTree(btree)) => btree.get(path, key, txn.id()).await,
            State::Collection(Collection::Table(table)) => table.get(path, key, txn).await,
            State::Collection(Collection::Tensor(tensor)) => tensor.get(path, key),
            _ => Ok(None),
        }
    }

    /// Apply a local PUT after the executor has resolved its reference and values.
    pub async fn put(
        &self,
        path: &[PathSegment],
        key: State,
        value: State,
        txn: &dyn Transaction,
    ) -> TCResult<Option<State>> {
        match self {
            State::Collection(Collection::Table(table)) => table.put(path, key, value, txn).await,
            _ => Ok(None),
        }
    }

    /// Apply a local POST after the executor has resolved its reference and parameters.
    pub async fn post(
        &self,
        path: &[PathSegment],
        params: Map<State>,
        txn: &dyn Transaction,
    ) -> TCResult<Option<State>> {
        match self {
            State::Collection(Collection::BTree(btree)) => btree.post(path, params, txn.id()).await,
            State::Collection(Collection::Table(table)) => table.post(path, params, txn).await,
            State::Collection(Collection::Tensor(tensor)) => tensor.post(path, params),
            _ => Ok(None),
        }
    }

    /// Apply a local DELETE after the executor has resolved its reference and key.
    pub async fn delete(
        &self,
        path: &[PathSegment],
        key: State,
        txn: &dyn Transaction,
    ) -> TCResult<Option<State>> {
        match self {
            State::Collection(Collection::BTree(btree)) => btree.delete(path, key, txn.id()).await,
            State::Collection(Collection::Table(table)) => table.delete(path, key, txn).await,
            _ => Ok(None),
        }
    }

    /// Convert this State into a format-neutral result stream.
    pub async fn into_result_stream(self) -> TCResult<StateStream> {
        match self {
            State::Collection(Collection::BTree(btree)) => btree.into_result_stream().await,
            State::Collection(Collection::Table(table)) => table.into_result_stream().await,
            state => Ok(StateStream::State(state)),
        }
    }

    /// Construct a native State literal for a built-in class URI.
    pub fn from_put(link: &Link, key: State, value: State) -> TCResult<Option<State>> {
        if link.path() != &TensorType.path() {
            return Ok(None);
        }

        <Tensor as TensorStateOps>::from_literal(key, value)
            .map(|tensor| Some(State::Collection(Collection::Tensor(tensor))))
    }
}

impl TableCollection {
    async fn get(
        &self,
        path: &[PathSegment],
        key: State,
        txn: &dyn Transaction,
    ) -> TCResult<Option<State>> {
        let tc_collection::table::Table::File(table) = &self.table else {
            return Ok(None);
        };

        let Some(handler) = tc_collection::table::public::route::<State>(table, path) else {
            return Ok(None);
        };

        let result = handler
            .get(txn, Scalar::try_cast_from(key, invalid_state)?)?
            .await?
            .with_table_txn(txn.id());
        Ok(Some(result))
    }

    async fn put(
        &self,
        path: &[PathSegment],
        key: State,
        value: State,
        txn: &dyn Transaction,
    ) -> TCResult<Option<State>> {
        let tc_collection::table::Table::File(table) = &self.table else {
            return Ok(None);
        };

        let Some(handler) = tc_collection::table::public::route::<State>(table, path) else {
            return Ok(None);
        };

        let mut request = Map::new();
        request.insert(
            "key".parse::<Id>().expect("fixed Table key parameter name"),
            Scalar::try_cast_from(key, invalid_state)?,
        );
        request.insert(
            "value"
                .parse::<Id>()
                .expect("fixed Table value parameter name"),
            Scalar::try_cast_from(value, invalid_state)?,
        );
        handler.put(txn, request)?.await?;
        Ok(Some(State::None))
    }

    async fn post(
        &self,
        path: &[PathSegment],
        params: Map<State>,
        txn: &dyn Transaction,
    ) -> TCResult<Option<State>> {
        let tc_collection::table::Table::File(table) = &self.table else {
            return Ok(None);
        };

        let Some(handler) = tc_collection::table::public::route::<State>(table, path) else {
            return Ok(None);
        };

        let request = Scalar::try_cast_from(State::Map(params), invalid_state)?;
        let Scalar::Map(request) = request else {
            unreachable!("State::Map always converts into Scalar::Map");
        };
        let result = handler.post(txn, request)?.await?.with_table_txn(txn.id());
        Ok(Some(result))
    }

    async fn delete(
        &self,
        path: &[PathSegment],
        key: State,
        txn: &dyn Transaction,
    ) -> TCResult<Option<State>> {
        let tc_collection::table::Table::File(table) = &self.table else {
            return Ok(None);
        };

        let Some(handler) = tc_collection::table::public::route::<State>(table, path) else {
            return Ok(None);
        };

        handler
            .delete(txn, Scalar::try_cast_from(key, invalid_state)?)?
            .await?;
        Ok(Some(State::None))
    }

    async fn into_result_stream(self: Box<Self>) -> TCResult<StateStream> {
        let txn_id = self.txn_id.ok_or_else(|| {
            TCError::internal("Table response is missing its transaction snapshot")
        })?;
        let schema = State::from(Value::cast_from(self.table.schema().clone()));

        let rows = self
            .table
            .row_stream(txn_id)
            .await?
            .map(|row| {
                row.map(|row| State::from(Value::Tuple(row.into_vec())))
                    .map_err(TCError::from)
            })
            .boxed();

        Ok(StateStream::Map(vec![(
            TableType.path().to_string(),
            StateStream::Tuple(vec![
                StateStream::State(schema),
                StateStream::Sequence(rows),
            ]),
        )]))
    }
}

impl State {
    fn with_table_txn(mut self, txn_id: TxnId) -> Self {
        match &mut self {
            State::Map(entries) => {
                for value in entries.values_mut() {
                    value.bind_table_txn(txn_id);
                }
            }
            State::Tuple(items) => {
                for value in items {
                    value.bind_table_txn(txn_id);
                }
            }
            State::Collection(Collection::Table(table)) => table.txn_id = Some(txn_id),
            State::None | State::Scalar(_) | State::Collection(_) => {}
        }

        self
    }

    fn bind_table_txn(&mut self, txn_id: TxnId) {
        let state = std::mem::replace(self, State::None);
        *self = state.with_table_txn(txn_id);
    }
}

fn invalid_state(state: &State) -> TCError {
    TCError::bad_request(format!("expected a scalar State, not {state:?}"))
}

trait TensorStateOps {
    fn get(&self, path: &[PathSegment], key: State) -> TCResult<Option<State>>;
    fn post(&self, path: &[PathSegment], params: Map<State>) -> TCResult<Option<State>>;
    fn from_literal(key: State, value: State) -> TCResult<Self>
    where
        Self: Sized;
}

impl TensorStateOps for Tensor {
    fn get(&self, path: &[PathSegment], key: State) -> TCResult<Option<State>> {
        if path.is_empty() {
            let range = tensor_range_from_state(key, self.shape())?;
            let tensor = self.clone().slice(range).map_err(TCError::bad_request)?;
            return Ok(Some(State::Collection(Collection::Tensor(tensor))));
        }

        if path.len() != 1 {
            return Ok(None);
        }

        let tensor = match path[0].as_str() {
            "broadcast" => self.clone().broadcast(shape_from_state(key)?),
            "cast" => self.clone().cast(tensor_dtype_from_state(key)?),
            "expand_dims" => self.clone().expand_dims(optional_shape_from_state(key)?),
            "reshape" => self.clone().reshape(shape_from_state(key)?),
            "transpose" => {
                tensor_transpose(self, &shape_from_state(key)?).map_err(|err| err.to_string())
            }
            _ => return Ok(None),
        }
        .map_err(TCError::bad_request)?;

        Ok(Some(State::Collection(Collection::Tensor(tensor))))
    }

    fn post(&self, path: &[PathSegment], params: Map<State>) -> TCResult<Option<State>> {
        if path.len() != 1 {
            return Ok(None);
        }

        let state = match path[0].as_str() {
            "dtype" => State::from(Value::String(
                number_type_path(&self.number_type()).to_string(),
            )),
            "ndim" => State::from(Value::Number(Number::from(self.shape().len() as u64))),
            "shape" => State::Scalar(Scalar::Tuple(
                self.shape()
                    .iter()
                    .map(|dim| Scalar::Value(Value::Number(Number::from(*dim as u64))))
                    .collect(),
            )),
            "size" => State::from(Value::Number(Number::from(self.size() as u64))),
            "all" => tensor_truthy_state(self, true)?,
            "any" => tensor_truthy_state(self, false)?,
            "cond" => State::Collection(Collection::Tensor(
                Tensor::cond(
                    self,
                    &tensor_param(&params, "then")?,
                    &tensor_param(&params, "or_else")?,
                )
                .map_err(TCError::bad_request)?,
            )),
            "max" | "min" | "mean" | "norm" | "product" | "std" | "sum" => match self
                .reduce_axes(
                    path[0].as_str(),
                    optional_axes_param(&params)?,
                    bool_param(&params, "keepdims")?,
                )
                .map_err(TCError::bad_request)?
            {
                TensorReduceResult::Scalar(number) => State::from(Value::Number(number)),
                TensorReduceResult::Tensor(tensor) => State::Collection(Collection::Tensor(tensor)),
            },
            "broadcast_reduce" => State::Collection(Collection::Tensor(tensor_op_result(
                broadcast_reduce_sum(self, &shape_param(&params, "target_shape")?),
            )?)),
            "matmul" => State::Collection(Collection::Tensor(tensor_op_result(batched_matmul(
                self,
                &tensor_param(&params, "r")?,
            ))?)),
            "transpose" => State::Collection(Collection::Tensor(tensor_op_result(
                tensor_transpose(self, &shape_param(&params, "perm")?),
            )?)),
            "add" => State::Collection(Collection::Tensor(tensor_op_result(broadcast_add(
                self,
                &tensor_param(&params, "r")?,
            ))?)),
            "sub" | "mul" | "div" | "and" | "or" | "xor" => State::Collection(Collection::Tensor(
                self.binary_op(&tensor_param(&params, "r")?, path[0].as_str())
                    .map_err(TCError::bad_request)?,
            )),
            "not" => State::Collection(Collection::Tensor(
                self.unary_not().map_err(TCError::bad_request)?,
            )),
            _ => return Ok(None),
        };

        Ok(Some(state))
    }

    fn from_literal(key: State, value: State) -> TCResult<Self> {
        let key = scalar_tuple(key, "tensor literal key")?;
        if key.len() != 2 {
            return Err(TCError::bad_request(
                "tensor literal key must be [dtype, shape]",
            ));
        }

        let dtype = tensor_dtype_from_state(key[0].clone())?;
        let shape = shape_from_state(key[1].clone())?;
        let values = numbers_from_state(value, "tensor literal values")?;
        match dtype {
            NumberType::Float(FloatType::F32) => {
                Self::dense_f32(shape, values.into_iter().map(CastInto::cast_into).collect())
                    .map_err(TCError::bad_request)
            }
            NumberType::Float(FloatType::F64) => {
                Self::dense_f64(shape, values.into_iter().map(CastInto::cast_into).collect())
                    .map_err(TCError::bad_request)
            }
            NumberType::UInt(UIntType::U64) => {
                Self::dense_u64(shape, values.into_iter().map(CastInto::cast_into).collect())
                    .map_err(TCError::bad_request)
            }
            dtype => Err(TCError::bad_request(format!(
                "unsupported tensor literal dtype {dtype}"
            ))),
        }
    }
}

impl BTreeCollection {
    async fn get(
        &self,
        path: &[PathSegment],
        key: State,
        txn_id: TxnId,
    ) -> TCResult<Option<State>> {
        if path.len() != 1 {
            return Ok(None);
        }

        let state = match path[0].as_str() {
            "contains" => {
                let row = row_from_state(key, "BTree row")?;
                State::from(Value::from(self.btree.contains_row(txn_id, &row).await))
            }
            "count" => {
                let collection = self.slice_from_key(key)?;
                let count = collection
                    .btree
                    .slice(collection.bounds.clone(), collection.reverse)
                    .count(txn_id)
                    .await;
                State::from(Value::from(count))
            }
            "is_empty" => {
                let collection = self.slice_from_key(key)?;
                let is_empty = collection
                    .btree
                    .slice(collection.bounds.clone(), collection.reverse)
                    .is_empty(txn_id)
                    .await;
                State::from(Value::from(is_empty))
            }
            "slice" => {
                let (bounds, reverse) = slice_bounds_from_state(key)?;
                State::Collection(Collection::from(self.slice(bounds, reverse)))
            }
            _ => return Ok(None),
        };

        Ok(Some(state))
    }

    async fn post(
        &self,
        path: &[PathSegment],
        params: Map<State>,
        txn_id: TxnId,
    ) -> TCResult<Option<State>> {
        if path.len() != 1 {
            return Ok(None);
        }

        let row_id: Id = "row".parse().expect("valid BTree row parameter id");
        let Some(row) = params.get(&row_id).cloned() else {
            return Err(TCError::bad_request("missing BTree row parameter"));
        };
        let row = row_from_state(row, "BTree row")?;

        match path[0].as_str() {
            "insert" => self
                .btree
                .insert_row(txn_id, row)
                .await
                .map_err(|err| TCError::bad_request(err.to_string()))?,
            "delete" => self
                .btree
                .delete_row(txn_id, row)
                .await
                .map_err(|err| TCError::bad_request(err.to_string()))?,
            _ => return Ok(None),
        }

        Ok(Some(State::None))
    }

    async fn delete(
        &self,
        path: &[PathSegment],
        key: State,
        txn_id: TxnId,
    ) -> TCResult<Option<State>> {
        if !path.is_empty() {
            return Ok(None);
        }

        self.btree
            .delete_row(txn_id, row_from_state(key, "BTree row")?)
            .await
            .map_err(|err| TCError::bad_request(err.to_string()))?;

        Ok(Some(State::None))
    }

    async fn into_result_stream(self: Box<Self>) -> TCResult<StateStream> {
        let arity = self.schema.len();
        let rows = self
            .finalized_key_stream()
            .await
            .map_err(|err| TCError::internal(err.to_string()))?
            .map(move |row| {
                let row = row.map_err(|err| TCError::internal(err.to_string()))?;
                if row.len() != arity {
                    return Err(TCError::internal(format!(
                        "BTree row arity {} does not match schema arity {arity}",
                        row.len()
                    )));
                }

                let value = if arity == 1 {
                    row.into_iter().next().expect("unary BTree row")
                } else {
                    Value::Tuple(row.to_vec())
                };

                Ok(State::from(value))
            });

        let schema = self
            .schema
            .iter()
            .cloned()
            .map(schema_stream)
            .collect::<Vec<_>>();

        Ok(StateStream::Map(vec![(
            BTreeType.path().to_string(),
            StateStream::Tuple(vec![
                StateStream::Tuple(schema),
                StateStream::Sequence(Box::pin(rows)),
            ]),
        )]))
    }

    fn slice_from_key(&self, key: State) -> TCResult<Self> {
        if key.is_none() {
            return Ok(self.clone());
        }

        let (bounds, reverse) = slice_bounds_from_state(key)?;
        Ok(self.slice(bounds, reverse))
    }
}

fn schema_stream(column: tc_collection::btree::BTreeColumnSchema) -> StateStream {
    let mut values = vec![
        State::from(Value::String(column.name)),
        State::from(Value::String(column.dtype.path().to_string())),
    ];
    if let Some(max_size) = column.max_size {
        values.push(State::from(Value::Number(max_size)));
    }

    StateStream::State(State::Tuple(values))
}

fn row_from_state(state: State, context: &str) -> TCResult<Vec<Value>> {
    Vec::<Value>::try_cast_from(state, |state| {
        TCError::bad_request(format!("expected {context} values but found {state:?}"))
    })
}

fn slice_bounds_from_state(key: State) -> TCResult<(BTreeBounds, bool)> {
    let map = match key {
        State::Map(map) => map,
        State::Scalar(Scalar::Map(map)) => map
            .into_iter()
            .map(|(id, scalar)| (id, State::Scalar(scalar)))
            .collect(),
        state => {
            return Err(TCError::bad_request(format!(
                "expected BTree slice key map but found {state:?}"
            )));
        }
    };

    let start = bound_from_map(&map, "start", Bound::Included)?;
    let end = bound_from_map(&map, "end", Bound::Excluded)?;
    let reverse_id: Id = "reverse".parse().expect("valid BTree reverse parameter id");
    let reverse = map
        .get(&reverse_id)
        .cloned()
        .map(|state| {
            Value::try_cast_from(state, |state| {
                TCError::bad_request(format!("expected BTree slice reverse but found {state:?}"))
            })
        })
        .transpose()?
        .map(|value| match value {
            Value::Number(number) => Ok(number.cast_into()),
            value => Err(TCError::bad_request(format!(
                "expected BTree slice reverse boolean but found {value:?}"
            ))),
        })
        .transpose()?
        .unwrap_or(false);

    Ok(((start, end), reverse))
}

fn bound_from_map(
    map: &Map<State>,
    name: &str,
    bound: fn(Value) -> Bound<Value>,
) -> TCResult<Bound<Value>> {
    let id: Id = name.parse().expect("valid BTree bound parameter id");
    match map.get(&id).cloned() {
        Some(state) if !state.is_none() => Value::try_cast_from(state, |state| {
            TCError::bad_request(format!("expected BTree {name} bound but found {state:?}"))
        })
        .map(bound),
        _ => Ok(Bound::Unbounded),
    }
}

fn tensor_param(params: &Map<State>, name: &str) -> TCResult<Tensor> {
    match required_param(params, name)? {
        State::Collection(Collection::Tensor(tensor)) => Ok(tensor),
        state => Err(TCError::bad_request(format!(
            "expected tensor parameter {name} but found {state:?}"
        ))),
    }
}

fn shape_param(params: &Map<State>, name: &str) -> TCResult<Vec<usize>> {
    shape_from_state(required_param(params, name)?)
}

fn optional_axes_param(params: &Map<State>) -> TCResult<Option<Vec<usize>>> {
    if let Some(state) = optional_param(params, "axes")? {
        return optional_axes_from_state(state);
    }
    if let Some(state) = optional_param(params, "axis")? {
        return optional_axes_from_state(state);
    }
    Ok(None)
}

fn bool_param(params: &Map<State>, name: &str) -> TCResult<bool> {
    match optional_param(params, name)? {
        Some(State::Scalar(Scalar::Value(Value::Number(number)))) => Ok(number.cast_into()),
        Some(state) => Err(TCError::bad_request(format!(
            "expected tensor {name} to be a boolean but found {state:?}"
        ))),
        None => Ok(false),
    }
}

fn required_param(params: &Map<State>, name: &str) -> TCResult<State> {
    optional_param(params, name)?
        .ok_or_else(|| TCError::bad_request(format!("missing tensor parameter {name}")))
}

fn optional_param(params: &Map<State>, name: &str) -> TCResult<Option<State>> {
    let id: Id = name
        .parse()
        .map_err(|err| TCError::internal(format!("invalid tensor parameter {name}: {err}")))?;
    Ok(params.get(&id).cloned())
}

fn scalar_tuple(state: State, context: &str) -> TCResult<Vec<State>> {
    match state {
        State::Tuple(items) => Ok(items),
        State::Scalar(Scalar::Tuple(items)) => Ok(items.into_iter().map(State::Scalar).collect()),
        state => Err(TCError::bad_request(format!(
            "expected {context} to be a tuple but found {state:?}"
        ))),
    }
}

fn numbers_from_state(state: State, context: &str) -> TCResult<Vec<Number>> {
    scalar_tuple(state, context)?
        .into_iter()
        .map(|state| match state {
            State::Scalar(Scalar::Value(Value::Number(number))) => Ok(number),
            state => Err(TCError::bad_request(format!(
                "expected {context} elements to be numbers but found {state:?}"
            ))),
        })
        .collect()
}

fn shape_from_state(state: State) -> TCResult<Vec<usize>> {
    scalar_tuple(state, "tensor shape")?
        .into_iter()
        .map(|state| match state {
            State::Scalar(Scalar::Value(Value::Number(number))) => {
                number_to_usize(number, "tensor shape dimension")
            }
            state => Err(TCError::bad_request(format!(
                "expected tensor shape dimension to be a number but found {state:?}"
            ))),
        })
        .collect()
}

fn optional_shape_from_state(state: State) -> TCResult<Option<Vec<usize>>> {
    if state.is_none() {
        Ok(None)
    } else {
        shape_from_state(state).map(Some)
    }
}

fn optional_axes_from_state(state: State) -> TCResult<Option<Vec<usize>>> {
    if state.is_none() {
        return Ok(None);
    }
    match state {
        State::Scalar(Scalar::Value(Value::Number(number))) => Ok(Some(vec![number_to_usize(
            number,
            "tensor reduction axis",
        )?])),
        state => shape_from_state(state).map(Some),
    }
}

fn tensor_dtype_from_state(state: State) -> TCResult<NumberType> {
    let raw = match state {
        State::Scalar(Scalar::Value(Value::String(dtype))) => dtype,
        State::Scalar(Scalar::Value(Value::Link(link))) => link.to_string(),
        state => {
            return Err(TCError::bad_request(format!(
                "expected tensor dtype to be a string or link but found {state:?}"
            )));
        }
    };

    parse_tensor_number_type(&raw)
        .ok_or_else(|| TCError::bad_request(format!("unsupported tensor dtype {raw}")))
}

fn parse_tensor_number_type(raw: &str) -> Option<NumberType> {
    match raw {
        "f32" => Some(NumberType::Float(FloatType::F32)),
        "f64" => Some(NumberType::Float(FloatType::F64)),
        "u64" => Some(NumberType::UInt(UIntType::U64)),
        _ => raw
            .parse::<PathBuf>()
            .ok()
            .and_then(|path| number_type_from_path(path.as_ref())),
    }
}

fn tensor_range_from_state(bounds: State, shape: &[usize]) -> TCResult<Range> {
    let bounds = scalar_tuple(bounds, "tensor slice")?;
    if bounds.len() != shape.len() {
        return Err(TCError::bad_request(format!(
            "tensor slice bounds rank {} does not match tensor rank {}",
            bounds.len(),
            shape.len()
        )));
    }

    bounds
        .into_iter()
        .zip(shape.iter().copied())
        .enumerate()
        .map(|(axis, (bound, dim))| tensor_axis_range_from_state(bound, axis, dim))
        .collect()
}

fn tensor_axis_range_from_state(state: State, axis: usize, dim: usize) -> TCResult<AxisRange> {
    match state {
        State::Scalar(Scalar::Value(Value::Number(number))) => {
            let index = number_to_usize(number, "tensor slice index")?;
            if index >= dim {
                return Err(TCError::bad_request(format!(
                    "tensor slice index {index} is out of bounds for axis {axis} with dim {dim}"
                )));
            }
            Ok(AxisRange::At(index))
        }
        state => {
            let parts = scalar_tuple(state, "tensor slice range")?;
            if parts.is_empty() || parts.len() > 3 {
                return Err(TCError::bad_request(
                    "tensor slice range must have 1 to 3 components",
                ));
            }
            let start = state_usize(parts[0].clone(), "tensor slice start")?;
            let stop = if let Some(stop) = parts.get(1) {
                state_usize(stop.clone(), "tensor slice stop")?
            } else {
                dim
            };
            let step = if let Some(step) = parts.get(2) {
                state_usize(step.clone(), "tensor slice step")?
            } else {
                1
            };
            if step == 0 || start > stop || stop > dim {
                return Err(TCError::bad_request(format!(
                    "invalid tensor slice range for axis {axis}"
                )));
            }
            Ok(AxisRange::In(start, stop, step))
        }
    }
}

fn state_usize(state: State, context: &str) -> TCResult<usize> {
    match state {
        State::Scalar(Scalar::Value(Value::Number(number))) => number_to_usize(number, context),
        state => Err(TCError::bad_request(format!(
            "expected {context} to be a number but found {state:?}"
        ))),
    }
}

fn number_to_usize(number: Number, context: &str) -> TCResult<usize> {
    let value: i64 = number.cast_into();
    if value < 0 {
        return Err(TCError::bad_request(format!(
            "expected {context} to be non-negative"
        )));
    }
    Ok(value as usize)
}

fn tensor_truthy_state(tensor: &Tensor, all: bool) -> TCResult<State> {
    let values = tensor.values_f64().map_err(TCError::bad_request)?;
    let value = if all {
        values.iter().all(|value| *value != 0.0)
    } else {
        values.iter().any(|value| *value != 0.0)
    };
    Ok(State::from(Value::Number(Number::Bool(value.into()))))
}
