//! Transitional TinyChain state primitives.
//!
//! This crate exposes the placeholder collection and scalar state enums used by
//! adapters that need to exchange TinyChain values before the full transactional
//! filesystem (`freqfs`) + `Chain` + `Service` stack lands. The in-memory tensor
//! representation keeps downstream crates unblocked while we finish the shared
//! persistence layer.

use std::{convert::TryFrom, str::FromStr, sync::Arc};

use destream::{
    de,
    en::{self, EncodeMap, EncodeSeq, Error as _},
    IntoStream,
};
use ha_ndarray::{ArrayBuf, Buffer, NDArray, NDArrayRead, NDArrayTransform};
use number_general::{FloatType, Number, UIntType};
use pathlink::Link;
use pathlink::PathBuf;
use safecast::{CastInto, TryCastFrom};
use tc_ir::{Claim, Id, Map, NetworkTime, Scalar, Transaction, TxnId};
use tc_value::{number_type_from_path, number_type_path, NumberType, Value, ValueType};

mod class;

pub use class::{CollectionType, StateType, TensorType};
pub use ha_ndarray::{AxisRange, Range};
pub use tc_ir::{Class, NativeClass};

/// Temporary tensor representation (in-memory only).
#[derive(Clone, Debug)]
pub enum Tensor {
    /// 32-bit floating point tensor.
    F32(Box<ArrayBuf<f32, Buffer<f32>>>),
    /// 64-bit floating point tensor.
    F64(Box<ArrayBuf<f64, Buffer<f64>>>),
    /// 64-bit unsigned integer tensor.
    U64(Box<ArrayBuf<u64, Buffer<u64>>>),
}

#[derive(Clone, Debug)]
pub enum TensorReduceResult {
    Scalar(Number),
    Tensor(Tensor),
}

impl Tensor {
    /// Construct a dense `f32` tensor from a shape and flattened values.
    pub fn dense_f32(shape: Vec<usize>, values: Vec<f32>) -> Result<Self, String> {
        let shape = shape.into();
        let buffer = Buffer::from(values);
        ArrayBuf::new(buffer, shape)
            .map(Box::new)
            .map(Tensor::F32)
            .map_err(|err| err.to_string())
    }

    /// Construct a dense `f64` tensor from a shape and flattened values.
    pub fn dense_f64(shape: Vec<usize>, values: Vec<f64>) -> Result<Self, String> {
        let shape = shape.into();
        let buffer = Buffer::from(values);
        ArrayBuf::new(buffer, shape)
            .map(Box::new)
            .map(Tensor::F64)
            .map_err(|err| err.to_string())
    }

    /// Construct a dense `u64` tensor from a shape and flattened values.
    pub fn dense_u64(shape: Vec<usize>, values: Vec<u64>) -> Result<Self, String> {
        let shape = shape.into();
        let buffer = Buffer::from(values);
        ArrayBuf::new(buffer, shape)
            .map(Box::new)
            .map(Tensor::U64)
            .map_err(|err| err.to_string())
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::F32(array) => array.shape(),
            Tensor::F64(array) => array.shape(),
            Tensor::U64(array) => array.shape(),
        }
    }

    pub fn number_type(&self) -> NumberType {
        match self {
            Tensor::F32(_) => NumberType::Float(FloatType::F32),
            Tensor::F64(_) => NumberType::Float(FloatType::F64),
            Tensor::U64(_) => NumberType::UInt(UIntType::U64),
        }
    }

    pub fn dtype_tag(&self) -> &'static str {
        match self {
            Tensor::F32(_) => "f32",
            Tensor::F64(_) => "f64",
            Tensor::U64(_) => "u64",
        }
    }

    pub fn size(&self) -> usize {
        self.shape().iter().product()
    }

    pub fn flattened_f32(&self) -> Result<Vec<f32>, String> {
        match self {
            Tensor::F32(array) => Ok(array
                .buffer()
                .map_err(|err| err.to_string())?
                .to_slice()
                .map_err(|err| err.to_string())?
                .into_vec()),
            Tensor::F64(_) => Err("tensor dtype is not f32".to_string()),
            Tensor::U64(_) => Err("tensor dtype is not f32".to_string()),
        }
    }

    pub fn flattened_f64(&self) -> Result<Vec<f64>, String> {
        match self {
            Tensor::F64(array) => Ok(array
                .buffer()
                .map_err(|err| err.to_string())?
                .to_slice()
                .map_err(|err| err.to_string())?
                .into_vec()),
            Tensor::F32(_) => Err("tensor dtype is not f64".to_string()),
            Tensor::U64(_) => Err("tensor dtype is not f64".to_string()),
        }
    }

    pub fn flattened_u64(&self) -> Result<Vec<u64>, String> {
        match self {
            Tensor::U64(array) => Ok(array
                .buffer()
                .map_err(|err| err.to_string())?
                .to_slice()
                .map_err(|err| err.to_string())?
                .into_vec()),
            Tensor::F32(_) => Err("tensor dtype is not u64".to_string()),
            Tensor::F64(_) => Err("tensor dtype is not u64".to_string()),
        }
    }

    pub fn values_f64(&self) -> Result<Vec<f64>, String> {
        match self {
            Tensor::F32(_) => Ok(self.flattened_f32()?.into_iter().map(f64::from).collect()),
            Tensor::F64(_) => self.flattened_f64(),
            Tensor::U64(_) => Ok(self
                .flattened_u64()?
                .into_iter()
                .map(|value| value as f64)
                .collect()),
        }
    }

    pub fn from_f64_like(&self, shape: Vec<usize>, values: Vec<f64>) -> Result<Self, String> {
        match self {
            Tensor::F64(_) => Tensor::dense_f64(shape, values),
            Tensor::U64(_)
                if values
                    .iter()
                    .all(|value| *value >= 0.0 && value.fract() == 0.0) =>
            {
                Tensor::dense_u64(
                    shape,
                    values.into_iter().map(|value| value as u64).collect(),
                )
            }
            _ => Tensor::dense_f32(
                shape,
                values.into_iter().map(|value| value as f32).collect(),
            ),
        }
    }

    pub fn cast(self, dtype: &str) -> Result<Self, String> {
        let dtype = normalize_dtype_tag(dtype)
            .ok_or_else(|| format!("unsupported tensor dtype {dtype}"))?;

        if self.dtype_tag() == dtype {
            return Ok(self);
        }

        let shape = self.shape().to_vec();
        let values = self.values_f64()?;

        match dtype {
            "f32" => Tensor::dense_f32(shape, values.into_iter().map(|v| v as f32).collect()),
            "f64" => Tensor::dense_f64(shape, values),
            "u64" => {
                let mut out = Vec::with_capacity(values.len());
                for value in values {
                    if !value.is_finite() {
                        return Err("tensor cast to u64 requires finite values".to_string());
                    }
                    if value < 0.0 || value.fract() != 0.0 {
                        return Err(format!(
                            "tensor cast to u64 requires non-negative whole numbers, found {value}"
                        ));
                    }
                    out.push(value as u64);
                }

                Tensor::dense_u64(shape, out)
            }
            other => Err(format!("unsupported tensor dtype {other}")),
        }
    }

    pub fn read_value(&self, coord: &[u64]) -> Result<Number, String> {
        let offset = coord_offset(self.shape(), coord)?;

        match self {
            Tensor::F32(_) => Ok(Number::from(self.flattened_f32()?[offset])),
            Tensor::F64(_) => Ok(Number::from(self.flattened_f64()?[offset])),
            Tensor::U64(_) => Ok(Number::from(self.flattened_u64()?[offset])),
        }
    }

    pub fn write_value(&mut self, coord: &[u64], value: Number) -> Result<(), String> {
        let shape = self.shape().to_vec();
        let offset = coord_offset(&shape, coord)?;

        let next = match self {
            Tensor::F32(_) => {
                ensure_non_complex(&value)?;
                let mut values = self.flattened_f32()?;
                values[offset] = value.cast_into();
                Tensor::dense_f32(shape, values)?
            }
            Tensor::F64(_) => {
                ensure_non_complex(&value)?;
                let mut values = self.flattened_f64()?;
                values[offset] = value.cast_into();
                Tensor::dense_f64(shape, values)?
            }
            Tensor::U64(_) => {
                ensure_tensor_u64_component(&value)?;
                let mut values = self.flattened_u64()?;
                values[offset] = value.cast_into();
                Tensor::dense_u64(shape, values)?
            }
        };

        *self = next;
        Ok(())
    }

    pub fn reshape(self, shape: Vec<usize>) -> Result<Self, String> {
        if shape.iter().product::<usize>() != self.size() {
            return Err("tensor reshape changes size".to_string());
        }

        self.from_f64_like(shape, self.values_f64()?)
    }

    pub fn expand_dims(self, axes: Option<Vec<usize>>) -> Result<Self, String> {
        let mut shape = self.shape().to_vec();

        if let Some(axes) = axes {
            for axis in axes {
                if axis > shape.len() {
                    return Err("expand_dims axis out of bounds".to_string());
                }

                shape.insert(axis, 1);
            }
        } else {
            shape.push(1);
        }

        self.from_f64_like(shape, self.values_f64()?)
    }

    pub fn broadcast(self, shape: Vec<usize>) -> Result<Self, String> {
        let source_shape = self.shape().to_vec();
        if !can_broadcast_to(&source_shape, &shape) {
            return Err(format!(
                "cannot broadcast tensor shape {:?} into {:?}",
                source_shape, shape
            ));
        }

        let source_values = self.values_f64()?;
        let out_len = shape.iter().product::<usize>();
        let mut out = Vec::with_capacity(out_len);

        for linear_idx in 0..out_len {
            let out_coord = unravel_index(linear_idx, &shape);
            let source_coord = project_broadcast_index(&out_coord, &source_shape)?;
            let source_offset = coord_offset_usize(&source_shape, &source_coord)?;
            out.push(source_values[source_offset]);
        }

        self.from_f64_like(shape, out)
    }

    pub fn transpose(self, permutation: Option<Vec<usize>>) -> Result<Self, String> {
        let shape = self.shape().to_vec();
        let permutation = if let Some(permutation) = permutation {
            if permutation.len() != shape.len() {
                return Err("transpose permutation rank must match tensor rank".to_string());
            }

            let mut seen = vec![false; shape.len()];
            for axis in &permutation {
                if *axis >= shape.len() {
                    return Err("transpose axis out of bounds".to_string());
                }
                if seen[*axis] {
                    return Err("transpose permutation contains duplicate axis".to_string());
                }
                seen[*axis] = true;
            }

            permutation
        } else {
            (0..shape.len()).rev().collect()
        };

        let out_shape: Vec<usize> = permutation.iter().map(|axis| shape[*axis]).collect();
        let out_len = out_shape.iter().product::<usize>();
        let values = self.values_f64()?;
        let mut out = vec![0.0; out_len];

        for (linear_idx, out_value) in out.iter_mut().enumerate() {
            let out_coord = unravel_index(linear_idx, &out_shape);
            let mut in_coord = vec![0usize; shape.len()];
            for (out_axis, in_axis) in permutation.iter().copied().enumerate() {
                in_coord[in_axis] = out_coord[out_axis];
            }

            let in_offset = coord_offset_usize(&shape, &in_coord)?;
            *out_value = values[in_offset];
        }

        self.from_f64_like(out_shape, out)
    }

    pub fn slice(self, range: Range) -> Result<Self, String> {
        match self {
            Tensor::F32(array) => {
                let sliced = array.slice(range.clone()).map_err(|err| err.to_string())?;
                let shape = sliced.shape().to_vec();
                let values = sliced
                    .buffer()
                    .map_err(|err| err.to_string())?
                    .to_slice()
                    .map_err(|err| err.to_string())?
                    .into_vec();

                Tensor::dense_f32(shape, values)
            }
            Tensor::F64(array) => {
                let sliced = array.slice(range.clone()).map_err(|err| err.to_string())?;
                let shape = sliced.shape().to_vec();
                let values = sliced
                    .buffer()
                    .map_err(|err| err.to_string())?
                    .to_slice()
                    .map_err(|err| err.to_string())?
                    .into_vec();

                Tensor::dense_f64(shape, values)
            }
            Tensor::U64(array) => {
                let sliced = array.slice(range).map_err(|err| err.to_string())?;
                let shape = sliced.shape().to_vec();
                let values = sliced
                    .buffer()
                    .map_err(|err| err.to_string())?
                    .to_slice()
                    .map_err(|err| err.to_string())?
                    .into_vec();

                Tensor::dense_u64(shape, values)
            }
        }
    }

    pub fn reduce(&self, op: &str) -> Result<Number, String> {
        let values = self.values_f64()?;
        if values.is_empty() {
            return Err("cannot reduce an empty tensor".to_string());
        }

        let value = match op {
            "max" => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            "min" => values.iter().copied().fold(f64::INFINITY, f64::min),
            "mean" => values.iter().sum::<f64>() / values.len() as f64,
            "norm" => values.iter().map(|v| v * v).sum::<f64>().sqrt(),
            "product" => values.iter().product::<f64>(),
            "std" => {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / values.len() as f64)
                    .sqrt()
            }
            "sum" => values.iter().sum::<f64>(),
            other => return Err(format!("unsupported tensor reduction {other}")),
        };

        Ok(Number::from(value))
    }

    pub fn reduce_axes(
        &self,
        op: &str,
        axes: Option<Vec<usize>>,
        keepdims: bool,
    ) -> Result<TensorReduceResult, String> {
        let shape = self.shape().to_vec();
        let rank = shape.len();
        let values = self.values_f64()?;

        if values.is_empty() {
            return Err("cannot reduce an empty tensor".to_string());
        }

        let axes = axes.unwrap_or_else(|| (0..rank).collect());
        let mut reduce_mask = vec![false; rank];
        for axis in axes {
            if axis >= rank {
                return Err(format!("reduction axis {axis} is out of bounds"));
            }
            reduce_mask[axis] = true;
        }

        let out_shape = if keepdims {
            shape
                .iter()
                .enumerate()
                .map(|(axis, dim)| if reduce_mask[axis] { 1 } else { *dim })
                .collect::<Vec<_>>()
        } else {
            shape
                .iter()
                .enumerate()
                .filter_map(|(axis, dim)| (!reduce_mask[axis]).then_some(*dim))
                .collect::<Vec<_>>()
        };

        let in_strides = strides(&shape);
        let out_strides = strides(&out_shape);
        let out_size = if out_shape.is_empty() {
            1
        } else {
            out_shape.iter().product()
        };

        let mut acc = vec![ReduceAccum::default(); out_size];
        for (flat_idx, value) in values.into_iter().enumerate() {
            let coord = coord_from_offset(flat_idx, &shape, &in_strides);

            let out_coord = if keepdims {
                coord
                    .iter()
                    .enumerate()
                    .map(|(axis, c)| if reduce_mask[axis] { 0 } else { *c })
                    .collect::<Vec<_>>()
            } else {
                coord
                    .iter()
                    .enumerate()
                    .filter_map(|(axis, c)| (!reduce_mask[axis]).then_some(*c))
                    .collect::<Vec<_>>()
            };

            let out_offset = offset_from_coord(&out_coord, &out_strides);
            acc[out_offset].update(value);
        }

        let out_values = acc
            .into_iter()
            .map(|acc| acc.finalize(op))
            .collect::<Result<Vec<_>, _>>()?;

        if out_shape.is_empty() {
            Ok(TensorReduceResult::Scalar(Number::from(out_values[0])))
        } else {
            let tensor = self.from_f64_like(out_shape, out_values)?;
            Ok(TensorReduceResult::Tensor(tensor))
        }
    }

    pub fn binary_op(&self, right: &Self, op: &str) -> Result<Self, String> {
        let left_values = self.values_f64()?;
        let right_values = right.values_f64()?;
        let shape = broadcast_shape(self.shape(), right.shape())?;

        let len = shape.iter().product::<usize>();
        let out = (0..len)
            .map(|idx| {
                let out_coord = unravel_index(idx, &shape);
                let left_coord = project_broadcast_index(&out_coord, self.shape())?;
                let right_coord = project_broadcast_index(&out_coord, right.shape())?;

                let l = left_values[coord_offset_usize(self.shape(), &left_coord)?];
                let r = right_values[coord_offset_usize(right.shape(), &right_coord)?];
                match op {
                    "add" => Ok(l + r),
                    "sub" => Ok(l - r),
                    "mul" => Ok(l * r),
                    "div" => Ok(l / r),
                    "and" => Ok(f64::from(l != 0.0 && r != 0.0)),
                    "or" => Ok(f64::from(l != 0.0 || r != 0.0)),
                    "xor" => Ok(f64::from((l != 0.0) ^ (r != 0.0))),
                    other => Err(format!("unsupported tensor binary op {other}")),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.from_f64_like(shape, out)
    }

    pub fn unary_not(&self) -> Result<Self, String> {
        let values = self
            .values_f64()?
            .into_iter()
            .map(|value| f64::from(value == 0.0))
            .collect();

        self.from_f64_like(self.shape().to_vec(), values)
    }

    pub fn cond(cond: &Self, then_tensor: &Self, else_tensor: &Self) -> Result<Self, String> {
        let cond_values = cond.values_f64()?;
        let then_values = then_tensor.values_f64()?;
        let else_values = else_tensor.values_f64()?;

        if then_values.len() != else_values.len() {
            return Err("tensor cond branches must have equal size".to_string());
        }

        if cond_values.len() != 1 && cond_values.len() != then_values.len() {
            return Err("tensor cond must be scalar or equal size".to_string());
        }

        let out = (0..then_values.len())
            .map(|idx| {
                if cond_values[if cond_values.len() == 1 { 0 } else { idx }] != 0.0 {
                    then_values[idx]
                } else {
                    else_values[idx]
                }
            })
            .collect();

        then_tensor.from_f64_like(then_tensor.shape().to_vec(), out)
    }

    pub fn matmul(&self, right: &Self) -> Result<Self, String> {
        let left_shape = self.shape();
        let right_shape = right.shape();
        if left_shape.len() != 2 || right_shape.len() != 2 {
            return Err("tensor matmul currently supports 2D tensors".to_string());
        }

        let rows = left_shape[0];
        let inner = left_shape[1];
        if right_shape[0] != inner {
            return Err("tensor matmul inner dimensions do not match".to_string());
        }

        let cols = right_shape[1];
        let left_values = self.values_f64()?;
        let right_values = right.values_f64()?;
        let mut out = vec![0.0; rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                let mut sum = 0.0;
                for k in 0..inner {
                    sum += left_values[row * inner + k] * right_values[k * cols + col];
                }
                out[row * cols + col] = sum;
            }
        }

        self.from_f64_like(vec![rows, cols], out)
    }
}

fn ensure_non_complex(number: &Number) -> Result<(), String> {
    if matches!(number, Number::Complex(_)) {
        Err("complex numbers are not supported in tensors".to_string())
    } else {
        Ok(())
    }
}

fn normalize_dtype_tag(dtype: &str) -> Option<&'static str> {
    if dtype == "f32" || dtype.ends_with("/float/32") {
        Some("f32")
    } else if dtype == "f64" || dtype.ends_with("/float/64") {
        Some("f64")
    } else if dtype == "u64" || dtype.ends_with("/uint/64") {
        Some("u64")
    } else {
        None
    }
}

fn can_broadcast_to(source: &[usize], target: &[usize]) -> bool {
    if source.len() > target.len() {
        return false;
    }

    source
        .iter()
        .rev()
        .zip(target.iter().rev())
        .all(|(left, right)| *left == 1 || left == right)
}

fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>, String> {
    let ndim = usize::max(left.len(), right.len());
    let mut out = vec![1usize; ndim];

    for i in 0..ndim {
        let l = left
            .len()
            .checked_sub(i + 1)
            .and_then(|idx| left.get(idx))
            .copied()
            .unwrap_or(1);
        let r = right
            .len()
            .checked_sub(i + 1)
            .and_then(|idx| right.get(idx))
            .copied()
            .unwrap_or(1);

        out[ndim - i - 1] = if l == r {
            l
        } else if l == 1 {
            r
        } else if r == 1 {
            l
        } else {
            return Err(format!(
                "tensor shapes {:?} and {:?} are not broadcast-compatible",
                left, right
            ));
        };
    }

    Ok(out)
}

fn project_broadcast_index(
    out_coord: &[usize],
    source_shape: &[usize],
) -> Result<Vec<usize>, String> {
    if source_shape.len() > out_coord.len() {
        return Err("source rank exceeds output rank for broadcast".to_string());
    }

    let mut source_coord = vec![0usize; source_shape.len()];
    let offset = out_coord.len() - source_shape.len();

    for (axis, dim) in source_shape.iter().copied().enumerate() {
        source_coord[axis] = if dim == 1 {
            0
        } else {
            out_coord[offset + axis]
        };
    }

    Ok(source_coord)
}

fn unravel_index(mut linear_idx: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut coord = vec![0usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        coord[axis] = linear_idx % dim;
        linear_idx /= dim;
    }

    coord
}

fn coord_offset_usize(shape: &[usize], coord: &[usize]) -> Result<usize, String> {
    if coord.len() != shape.len() {
        return Err("incorrect number of coordinates".to_string());
    }

    let mut offset = 0usize;
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        let value = coord[axis];
        if value >= dim {
            return Err(format!("coordinate at axis {axis} is out of bounds"));
        }
        offset += value * stride;
        stride *= dim;
    }

    Ok(offset)
}

fn coord_offset(shape: &[usize], coord: &[u64]) -> Result<usize, String> {
    if coord.len() != shape.len() {
        return Err("incorrect number of coordinates".to_string());
    }

    let mut offset = 0usize;
    let mut stride = 1usize;

    for (axis, dim) in shape.iter().enumerate().rev() {
        let value = usize::try_from(coord[axis])
            .map_err(|_| format!("coordinate at axis {axis} overflows usize"))?;

        if value >= *dim {
            return Err(format!("coordinate at axis {axis} is out of bounds"));
        }

        offset += value * stride;
        stride *= *dim;
    }

    Ok(offset)
}

fn strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len() - 1).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

fn coord_from_offset(offset: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut remainder = offset;
    let mut coord = vec![0; shape.len()];
    for axis in 0..shape.len() {
        let stride = strides[axis];
        coord[axis] = remainder / stride;
        remainder %= stride;
    }
    coord
}

fn offset_from_coord(coord: &[usize], strides: &[usize]) -> usize {
    coord
        .iter()
        .zip(strides.iter())
        .map(|(value, stride)| value * stride)
        .sum()
}

#[derive(Clone, Copy, Debug, Default)]
struct ReduceAccum {
    initialized: bool,
    count: usize,
    sum: f64,
    sumsq: f64,
    product: f64,
    min: f64,
    max: f64,
}

impl ReduceAccum {
    fn update(&mut self, value: f64) {
        if !self.initialized {
            self.initialized = true;
            self.count = 1;
            self.sum = value;
            self.sumsq = value * value;
            self.product = value;
            self.min = value;
            self.max = value;
            return;
        }

        self.count += 1;
        self.sum += value;
        self.sumsq += value * value;
        self.product *= value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    fn finalize(self, op: &str) -> Result<f64, String> {
        if !self.initialized {
            return Err("cannot reduce an empty tensor".to_string());
        }

        match op {
            "max" => Ok(self.max),
            "min" => Ok(self.min),
            "mean" => Ok(self.sum / self.count as f64),
            "norm" => Ok(self.sumsq.sqrt()),
            "product" => Ok(self.product),
            "std" => {
                let mean = self.sum / self.count as f64;
                Ok((self.sumsq / self.count as f64 - mean * mean)
                    .max(0.0)
                    .sqrt())
            }
            "sum" => Ok(self.sum),
            other => Err(format!("unsupported tensor reduction {other}")),
        }
    }
}

/// Temporary collection enum.
#[derive(Clone, Debug)]
pub enum Collection {
    /// Tensor data stored entirely in memory. Variants cover f32 and u64 element types.
    Tensor(Tensor),
}

impl From<ArrayBuf<f32, Buffer<f32>>> for Collection {
    fn from(tensor: ArrayBuf<f32, Buffer<f32>>) -> Self {
        Collection::Tensor(Tensor::F32(Box::new(tensor)))
    }
}

impl From<ArrayBuf<u64, Buffer<u64>>> for Collection {
    fn from(tensor: ArrayBuf<u64, Buffer<u64>>) -> Self {
        Collection::Tensor(Tensor::U64(Box::new(tensor)))
    }
}

impl<'en> en::IntoStream<'en> for Collection {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let mut map = encoder.encode_map(Some(1))?;
        match self {
            Collection::Tensor(tensor) => {
                let tensor_path = TensorType.path().to_string();
                map.encode_entry(tensor_path, tensor)?;
            }
        }
        map.end()
    }
}

impl<'en> en::ToStream<'en> for Collection {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        self.clone().into_stream(encoder)
    }
}

impl TryCastFrom<Collection> for Tensor {
    fn can_cast_from(collection: &Collection) -> bool {
        matches!(collection, Collection::Tensor(_))
    }

    fn opt_cast_from(collection: Collection) -> Option<Self> {
        match collection {
            Collection::Tensor(tensor) => Some(tensor),
        }
    }
}

impl<'en> en::IntoStream<'en> for Tensor {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let mut seq = encoder.encode_seq(Some(2))?;
        match self {
            Tensor::F32(array) => {
                let schema = (
                    number_type_path(&NumberType::Float(FloatType::F32)).to_string(),
                    array
                        .shape()
                        .iter()
                        .map(|dim| *dim as u64)
                        .collect::<Vec<_>>(),
                );
                seq.encode_element(schema)?;
                let values = array
                    .buffer()
                    .map_err(E::Error::custom)?
                    .to_slice()
                    .map_err(E::Error::custom)?
                    .into_vec();
                seq.encode_element(values)?;
            }
            Tensor::F64(array) => {
                let schema = (
                    number_type_path(&NumberType::Float(FloatType::F64)).to_string(),
                    array
                        .shape()
                        .iter()
                        .map(|dim| *dim as u64)
                        .collect::<Vec<_>>(),
                );
                seq.encode_element(schema)?;
                let values = array
                    .buffer()
                    .map_err(E::Error::custom)?
                    .to_slice()
                    .map_err(E::Error::custom)?
                    .into_vec();
                seq.encode_element(values)?;
            }
            Tensor::U64(array) => {
                let schema = (
                    number_type_path(&NumberType::UInt(UIntType::U64)).to_string(),
                    array
                        .shape()
                        .iter()
                        .map(|dim| *dim as u64)
                        .collect::<Vec<_>>(),
                );
                seq.encode_element(schema)?;
                let values = array
                    .buffer()
                    .map_err(E::Error::custom)?
                    .to_slice()
                    .map_err(E::Error::custom)?
                    .into_vec();
                seq.encode_element(values)?;
            }
        }
        seq.end()
    }
}

impl<'en> en::ToStream<'en> for Tensor {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        self.clone().into_stream(encoder)
    }
}

impl de::FromStream for Tensor {
    type Context = Arc<dyn Transaction>;

    async fn from_stream<D: de::Decoder>(
        _context: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct TensorVisitor;

        impl de::Visitor for TensorVisitor {
            type Value = Tensor;

            fn expecting() -> &'static str {
                "a TinyChain tensor payload"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let (dtype_path, shape): (String, Vec<u64>) = seq
                    .next_element(())
                    .await?
                    .ok_or_else(|| de::Error::custom("missing tensor schema"))?;
                let dtype_path_buf = dtype_path
                    .parse::<PathBuf>()
                    .map_err(|err| de::Error::custom(err.to_string()))?;
                let dtype = number_type_from_path(&dtype_path_buf).ok_or_else(|| {
                    de::Error::invalid_value(
                        dtype_path,
                        "a TinyChain numeric type path for tensor dtype",
                    )
                })?;

                let shape = coerce_shape(shape).map_err(de::Error::custom)?;

                let values = seq
                    .next_element::<Vec<Number>>(())
                    .await?
                    .ok_or_else(|| de::Error::custom("missing tensor values"))?;

                tensor_from_parts(dtype, shape, values).map_err(de::Error::custom)
            }
        }

        decoder.decode_seq(TensorVisitor).await
    }
}

#[derive(Clone)]
struct NullTransaction {
    id: TxnId,
    claim: Claim,
}

impl Default for NullTransaction {
    fn default() -> Self {
        let id = TxnId::from_parts(NetworkTime::from_nanos(0), 0);
        let claim = Claim::new(
            Link::from_str("/lib/default").expect("default claim link"),
            umask::Mode::all(),
        );
        Self { id, claim }
    }
}

impl Transaction for NullTransaction {
    fn id(&self) -> TxnId {
        self.id
    }

    fn timestamp(&self) -> NetworkTime {
        self.id.timestamp()
    }

    fn claim(&self) -> &Claim {
        &self.claim
    }
}

/// Return a placeholder transaction context for decoding state without a transaction.
pub fn null_transaction() -> Arc<dyn Transaction> {
    Arc::new(NullTransaction::default())
}

/// Transitional TinyChain state enum.
#[derive(Clone, Debug)]
pub enum State {
    None,
    Scalar(Scalar),
    Map(Map<State>),
    Tuple(Vec<State>),
    Collection(Collection),
}

struct StateSeq(Vec<State>);

impl de::FromStream for StateSeq {
    type Context = Arc<dyn Transaction>;

    async fn from_stream<D: de::Decoder>(
        context: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct StateSeqVisitor {
            context: Arc<dyn Transaction>,
        }

        impl de::Visitor for StateSeqVisitor {
            type Value = Vec<State>;

            fn expecting() -> &'static str {
                "a TinyChain state tuple"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut items: Vec<State> = if let Some(size) = seq.size_hint() {
                    Vec::with_capacity(size)
                } else {
                    Vec::new()
                };

                while let Some(value) = seq.next_element::<State>(self.context.clone()).await? {
                    items.push(value);
                }

                Ok(items)
            }
        }

        decoder
            .decode_seq(StateSeqVisitor { context })
            .await
            .map(StateSeq)
    }
}

impl State {
    pub fn is_none(&self) -> bool {
        match self {
            State::None => true,
            State::Scalar(Scalar::Value(Value::None)) => true,
            State::Tuple(items) => items.is_empty(),
            _ => false,
        }
    }
}

impl Default for State {
    fn default() -> Self {
        State::Scalar(Scalar::default())
    }
}

impl From<Value> for State {
    fn from(value: Value) -> Self {
        State::Scalar(Scalar::from(value))
    }
}

impl From<Collection> for State {
    fn from(collection: Collection) -> Self {
        State::Collection(collection)
    }
}

impl de::FromStream for State {
    type Context = Arc<dyn Transaction>;

    async fn from_stream<D: de::Decoder>(
        context: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct StateVisitor {
            context: Arc<dyn Transaction>,
        }

        impl de::Visitor for StateVisitor {
            type Value = State;

            fn expecting() -> &'static str {
                "a TinyChain state placeholder"
            }

            fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(State::None)
            }

            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(State::None)
            }

            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(State::from(Number::from(value)))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(State::from(Number::from(value)))
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(State::from(Number::from(value)))
            }

            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
                Ok(State::from(Number::from(value)))
            }

            fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(State::from(Value::from(value)))
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut items: Vec<State> = if let Some(size) = seq.size_hint() {
                    Vec::with_capacity(size)
                } else {
                    Vec::new()
                };

                while let Some(value) = seq.next_element::<State>(self.context.clone()).await? {
                    items.push(value);
                }

                Ok(State::Tuple(items))
            }

            async fn visit_map<A: de::MapAccess>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let Some(key) = map.next_key::<String>(()).await? else {
                    return Ok(State::Map(Map::new()));
                };

                if !key.starts_with('/') {
                    let mut out = Map::<State>::new();
                    let value = map.next_value::<State>(self.context.clone()).await?;
                    let id: Id = key
                        .parse::<Id>()
                        .map_err(|err| de::Error::custom(err.to_string()))?;
                    out.insert(id, value);

                    while let Some(key) = map.next_key::<String>(()).await? {
                        let value = map.next_value::<State>(self.context.clone()).await?;
                        let id: Id = key
                            .parse::<Id>()
                            .map_err(|err| de::Error::custom(err.to_string()))?;
                        out.insert(id, value);
                    }

                    return Ok(State::Map(out));
                }

                let path = key
                    .parse::<PathBuf>()
                    .map_err(|err| de::Error::custom(err.to_string()))?;

                let state_type = StateType::from_path(&path).ok_or_else(|| {
                    de::Error::invalid_value(path.to_string(), "a known TinyChain state type path")
                })?;

                match state_type {
                    StateType::Tuple => {
                        let StateSeq(tuple) =
                            map.next_value::<StateSeq>(self.context.clone()).await?;
                        drain_remaining_entries(&mut map).await?;
                        Ok(State::Tuple(tuple))
                    }
                    StateType::Collection(CollectionType::Tensor(_)) => {
                        let tensor = map.next_value::<Tensor>(self.context.clone()).await?;
                        drain_remaining_entries(&mut map).await?;
                        Ok(State::Collection(Collection::Tensor(tensor)))
                    }
                    StateType::Scalar(value_type) => {
                        let value = decode_value_entry(value_type, &mut map).await?;
                        drain_remaining_entries(&mut map).await?;
                        Ok(State::Scalar(Scalar::from(value)))
                    }
                }
            }
        }

        decoder.decode_any(StateVisitor { context }).await
    }
}

impl<'en> en::IntoStream<'en> for State {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            State::None => encoder.encode_unit(),
            State::Scalar(scalar) => scalar.into_stream(encoder),
            State::Map(map) => map.into_stream(encoder),
            State::Tuple(items) => items.into_stream(encoder),
            State::Collection(collection) => collection.into_stream(encoder),
        }
    }
}

impl From<Number> for State {
    fn from(number: Number) -> Self {
        State::from(Value::from(number))
    }
}

async fn decode_value_entry<A: de::MapAccess>(
    value_type: ValueType,
    map: &mut A,
) -> Result<Value, A::Error> {
    match value_type {
        ValueType::Link => {
            let link_raw = map.next_value::<String>(()).await?;
            let link =
                Link::from_str(&link_raw).map_err(|err| de::Error::custom(err.to_string()))?;
            Ok(Value::Link(link))
        }
        ValueType::Map => map
            .next_value::<std::collections::BTreeMap<String, Value>>(())
            .await
            .map(Value::Map),
        ValueType::Number => map.next_value::<Number>(()).await.map(Value::Number),
        ValueType::None => {
            let _ = map.next_value::<de::IgnoredAny>(()).await?;
            Ok(Value::None)
        }
        ValueType::String => map.next_value::<String>(()).await.map(Value::String),
        ValueType::Tuple => map.next_value::<Vec<Value>>(()).await.map(Value::Tuple),
    }
}

async fn drain_remaining_entries<A: de::MapAccess>(map: &mut A) -> Result<(), A::Error> {
    while map.next_key::<de::IgnoredAny>(()).await?.is_some() {
        let _ = map.next_value::<de::IgnoredAny>(()).await?;
    }
    Ok(())
}

fn tensor_from_parts(
    dtype: NumberType,
    shape: Vec<usize>,
    values: Vec<Number>,
) -> Result<Tensor, String> {
    match dtype {
        NumberType::Float(FloatType::F32) => {
            let values = numbers_to_f32(values)?;
            Tensor::dense_f32(shape, values)
        }
        NumberType::Float(FloatType::F64) => {
            let values = numbers_to_f64(values)?;
            Tensor::dense_f64(shape, values)
        }
        NumberType::UInt(UIntType::U64) => {
            let values = numbers_to_u64(values)?;
            Tensor::dense_u64(shape, values)
        }
        other => Err(format!("unsupported tensor dtype {other}")),
    }
}

fn coerce_shape(dims: Vec<u64>) -> Result<Vec<usize>, String> {
    dims.into_iter()
        .map(|dim| usize::try_from(dim).map_err(|_| format!("invalid dimension {dim}")))
        .collect()
}

fn numbers_to_f32(values: Vec<Number>) -> Result<Vec<f32>, String> {
    values
        .into_iter()
        .map(|number| {
            if matches!(number, Number::Complex(_)) {
                Err("complex numbers are not supported in tensors".into())
            } else {
                Ok(number.cast_into())
            }
        })
        .collect()
}

fn numbers_to_f64(values: Vec<Number>) -> Result<Vec<f64>, String> {
    values
        .into_iter()
        .map(|number| {
            if matches!(number, Number::Complex(_)) {
                Err("complex numbers are not supported in tensors".into())
            } else {
                Ok(number.cast_into())
            }
        })
        .collect()
}

fn numbers_to_u64(values: Vec<Number>) -> Result<Vec<u64>, String> {
    values
        .into_iter()
        .map(|number| {
            ensure_tensor_u64_component(&number)?;
            Ok(number.cast_into())
        })
        .collect()
}

fn ensure_tensor_u64_component(number: &Number) -> Result<(), String> {
    match number {
        Number::Bool(_) | Number::UInt(_) => Ok(()),
        Number::Int(int) => {
            let value = i64::from(*int);
            if value < 0 {
                Err("tensor values must be non-negative".into())
            } else {
                Ok(())
            }
        }
        Number::Float(float) => {
            let value = f64::from(*float);
            if !value.is_finite() {
                Err("tensor value must be finite".into())
            } else if value < 0.0 || value.fract() != 0.0 {
                Err(format!(
                    "expected a non-negative whole number but found {value}"
                ))
            } else {
                Ok(())
            }
        }
        Number::Complex(_) => Err("complex numbers are not supported in tensors".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use destream::{de, en};
    use futures::{executor::block_on, stream, TryStreamExt};
    use ha_ndarray::NDArray;

    fn encode_json<T>(value: T) -> Vec<u8>
    where
        T: for<'en> en::IntoStream<'en>,
    {
        block_on(
            destream_json::encode(value)
                .expect("encode json payload")
                .map_err(|err| err.to_string())
                .try_fold(Vec::new(), |mut acc, chunk| async move {
                    acc.extend_from_slice(&chunk);
                    Ok(acc)
                }),
        )
        .expect("collect json payload")
    }

    fn decode_json<T>(context: T::Context, bytes: Vec<u8>) -> T
    where
        T: de::FromStream,
    {
        let stream = stream::iter(vec![Ok::<Bytes, std::io::Error>(Bytes::from(bytes))]);
        block_on(destream_json::try_decode(context, stream)).expect("decode json payload")
    }

    #[test]
    fn tensor_variant_matches() {
        let shape = Vec::from([3usize]).into();
        let tensor = ArrayBuf::<f32, Buffer<f32>>::new(vec![1.0_f32, 2.0, 3.0].into(), shape)
            .expect("tensor buffer");
        let collection = Collection::from(tensor);
        assert!(Tensor::can_cast_from(&collection));
        match &collection {
            Collection::Tensor(Tensor::F32(buf)) => assert_eq!(buf.size(), 3),
            _ => panic!("expected F32 tensor"),
        }
    }

    #[test]
    fn scalar_numbers_round_trip() {
        let encoded = encode_json(true);
        let state: State = decode_json(null_transaction(), encoded);
        assert!(matches!(
            state,
            State::Scalar(Scalar::Value(Value::Number(_)))
        ));
    }

    #[test]
    fn tensor_round_trip() {
        let shape = Vec::from([2usize, 2usize]).into();
        let tensor = ArrayBuf::<f32, Buffer<f32>>::new(vec![1.0, 2.0, 3.0, 4.0].into(), shape)
            .expect("tensor buffer");
        let state = State::Collection(Collection::from(tensor));

        let encoded = encode_json(state.clone());
        let decoded: State = decode_json(null_transaction(), encoded);

        match decoded {
            State::Collection(Collection::Tensor(Tensor::F32(buf))) => {
                assert_eq!(buf.size(), 4);
            }
            other => panic!("unexpected state {other:?}"),
        }
    }

    #[test]
    fn tensor_f64_round_trip() {
        let tensor = Tensor::dense_f64(vec![2], vec![1.5, 2.5]).expect("dense tensor");
        let bytes = encode_json(tensor.clone());
        let decoded: Tensor = decode_json(null_transaction(), bytes);
        match decoded {
            Tensor::F64(buf) => assert_eq!(buf.size(), 2),
            other => panic!("unexpected tensor variant {other:?}"),
        }
    }

    #[test]
    fn tensor_direct_round_trip() {
        let tensor = Tensor::dense_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("dense tensor");
        let bytes = encode_json(tensor.clone());
        let decoded: Tensor = decode_json(null_transaction(), bytes);
        match decoded {
            Tensor::F32(buf) => assert_eq!(buf.size(), 4),
            other => panic!("unexpected tensor variant {other:?}"),
        }
    }

    #[test]
    fn state_map_round_trip_uses_plain_json_object() {
        let mut map = Map::new();
        map.insert(
            "status".parse().expect("id"),
            State::from(Value::from("ok")),
        );
        map.insert(
            "count".parse().expect("id"),
            State::from(Value::from(7_u64)),
        );
        let state = State::Map(map);

        let encoded = encode_json(state);
        let text = String::from_utf8(encoded.clone()).expect("utf-8");
        assert!(text.starts_with('{'));
        assert!(!text.contains("/state/scalar/map"));
        assert!(text.contains("\"status\""));
        assert!(text.contains("\"count\""));

        let decoded: State = decode_json(null_transaction(), encoded);
        assert!(matches!(decoded, State::Map(_)));
    }

    #[test]
    fn state_scalar_ref_serializes() {
        let state = State::Scalar(Scalar::from(tc_ir::TCRef::Id(
            "$foo".parse().expect("IdRef"),
        )));

        let encoded = encode_json(state);
        let text = String::from_utf8(encoded).expect("utf-8");
        assert_eq!(text, r#"{"$foo":[]}"#);
    }

    #[test]
    fn tensor_facade_read_write_value_roundtrip() {
        let mut tensor = Tensor::dense_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("tensor");

        let value = tensor.read_value(&[1, 0]).expect("read value");
        let value: f64 = value.cast_into();
        assert_eq!(value, 3.0);

        tensor
            .write_value(&[0, 1], Number::from(9.5_f64))
            .expect("write value");

        assert_eq!(
            tensor.flattened_f32().expect("values"),
            vec![1.0, 9.5, 3.0, 4.0]
        );
    }

    #[test]
    fn tensor_facade_transpose_2d() {
        let tensor = Tensor::dense_u64(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).expect("tensor");

        let transposed = tensor.transpose(None).expect("transpose");
        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(
            transposed.flattened_u64().expect("values"),
            vec![1, 4, 2, 5, 3, 6]
        );
    }

    #[test]
    fn tensor_facade_binary_and_reduce() {
        let left = Tensor::dense_f64(vec![2], vec![1.0, 3.0]).expect("left");
        let right = Tensor::dense_f64(vec![2], vec![2.0, 4.0]).expect("right");

        let added = left.binary_op(&right, "add").expect("binary add");
        assert_eq!(added.flattened_f64().expect("values"), vec![3.0, 7.0]);

        let reduced = added.reduce("sum").expect("reduce sum");
        let reduced: f64 = reduced.cast_into();
        assert_eq!(reduced, 10.0);
    }

    #[test]
    fn tensor_facade_reduce_axes_keepdims() {
        let tensor =
            Tensor::dense_f64(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("source tensor");

        let reduced = tensor
            .reduce_axes("sum", Some(vec![1]), true)
            .expect("reduce axes");

        match reduced {
            TensorReduceResult::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[2, 1]);
                assert_eq!(tensor.flattened_f64().expect("values"), vec![3.0, 7.0]);
            }
            other => panic!("expected reduced tensor but found {other:?}"),
        }
    }

    #[test]
    fn tensor_facade_cast_to_u64() {
        let tensor = Tensor::dense_f64(vec![3], vec![1.0, 2.0, 3.0]).expect("source tensor");
        let cast = tensor.cast("u64").expect("cast");

        assert_eq!(cast.dtype_tag(), "u64");
        assert_eq!(cast.flattened_u64().expect("values"), vec![1, 2, 3]);
    }

    #[test]
    fn tensor_facade_transpose_3d_permutation() {
        let tensor =
            Tensor::dense_u64(vec![2, 3, 2], (0_u64..12_u64).collect()).expect("source tensor");
        let transposed = tensor.transpose(Some(vec![2, 0, 1])).expect("transpose");

        assert_eq!(transposed.shape(), &[2, 2, 3]);
        assert_eq!(
            transposed.flattened_u64().expect("values"),
            vec![0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]
        );
    }

    #[test]
    fn tensor_facade_binary_broadcast() {
        let left =
            Tensor::dense_f64(vec![2, 1], vec![1.0, 2.0]).expect("left broadcast source tensor");
        let right = Tensor::dense_f64(vec![1, 3], vec![10.0, 20.0, 30.0]).expect("right tensor");

        let sum = left.binary_op(&right, "add").expect("broadcast add");
        assert_eq!(sum.shape(), &[2, 3]);
        assert_eq!(
            sum.flattened_f64().expect("values"),
            vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]
        );
    }

    #[test]
    fn tensor_facade_matmul_2d() {
        let left = Tensor::dense_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("left");
        let right = Tensor::dense_f32(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).expect("right");

        let out = left.matmul(&right).expect("matmul");
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(
            out.flattened_f32().expect("values"),
            vec![19.0, 22.0, 43.0, 50.0]
        );
    }

    #[test]
    fn tensor_facade_slice_range() {
        let tensor = Tensor::dense_u64(vec![3, 4], (0_u64..12_u64).collect()).expect("tensor");

        let mut range = Range::new();
        range.push(AxisRange::In(1, 3, 1));
        range.push(AxisRange::In(1, 3, 1));

        let sliced = tensor.slice(range).expect("slice");
        assert_eq!(sliced.shape(), &[2, 2]);
        assert_eq!(sliced.flattened_u64().expect("values"), vec![5, 6, 9, 10]);
    }
}
