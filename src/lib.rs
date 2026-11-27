#![forbid(unsafe_code)]

mod class;
mod codec;
mod gateway;
mod resolve;
mod route;
mod state;
mod view;

pub use class::{
    validate_classes, ClassBody, ClassDef, ClassError, ClassInstance, ClassParent, MemberSource,
    Object, ResolvedMember, MAX_INHERITANCE_DEPTH,
};
pub use codec::{
    decode_state, BTreeType, CollectionType, ObjectType, StateType, TableType, TensorType,
};
pub use gateway::Gateway;
pub use resolve::{resolve_ref, resolve_scalar as resolve};
pub use route::{route_scalar, BoundMethod, OpExecutor, Static};
pub use state::State;
pub use tc_collection::Collection;
pub use tc_collection::tensor::{AxisRange, Range, Tensor, TensorReduceResult};
pub use tc_value::class::{Class, NativeClass};
pub use view::{ObjectView, StateView};
