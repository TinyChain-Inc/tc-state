//! Native public routing for universal state and user-defined objects.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::future::Future;
use std::marker::PhantomData;

use number_general::{IntType, NumberClass, NumberInstance, UIntType};
use pathlink::{Link, PathSegment};
use safecast::TryCastFrom;
use tc_ir::{
    DeleteHandler, GetHandler, Handler, Map, OpDef, PostHandler, PutHandler, Route, Scalar,
};
use tc_value::Value;
use tc_value::class::NativeClass;

use super::{ClassDef, ClassInstance, Object, State};

/// Trusted host capability for executing a bound operation definition.
///
/// State routes bind methods; the server owns scheduling, admission, deadlines,
/// and cancellation. This capability does not independently grant permissions.
pub trait OpExecutor: super::Gateway {
    fn execute_op(
        &self,
        definition: OpDef,
        args: State<Self>,
        subject: Option<State<Self>>,
        declared_by: Option<Link>,
    ) -> impl Future<Output = tc_error::TCResult<State<Self>>> + Send;
}

/// The sole structural router for TinyChain's built-in `/state` namespace.
pub struct Static<Txn>(PhantomData<fn() -> Txn>);

impl<Txn> Default for Static<Txn> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Txn> Route<State<Txn>> for Static<Txn>
where
    Txn: OpExecutor,
{
    fn route<'a>(&'a self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        if path_eq(path, &["scalar", "value", "number", "add"]) {
            Some(Box::new(StaticAdd))
        } else if path_eq(path, &["scalar", "value", "number", "gt"]) {
            Some(Box::new(StaticGreater))
        } else {
            Reflection::from_segments(path)
                .map(|reflection| Box::new(Reflect(reflection)) as Box<_>)
        }
    }
}

fn path_eq(path: &[PathSegment], expected: &[&str]) -> bool {
    path.len() == expected.len()
        && path
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.as_str() == *expected)
}

struct StaticAdd;
struct StaticGreater;
struct Reflect(Reflection);

#[derive(Clone, Copy)]
enum Reflection {
    ScalarClass,
    ScalarRefParts,
    OpDefForm,
    OpDefLastId,
    OpDefScalars,
}

impl Reflection {
    fn from_segments(path: &[PathSegment]) -> Option<Self> {
        match path {
            path if path_eq(path, &["scalar", "reflect", "class"]) => Some(Self::ScalarClass),
            path if path_eq(path, &["scalar", "reflect", "ref_parts"]) => {
                Some(Self::ScalarRefParts)
            }
            path if path_eq(path, &["scalar", "op", "reflect", "form"]) => Some(Self::OpDefForm),
            path if path_eq(path, &["scalar", "op", "reflect", "last_id"]) => {
                Some(Self::OpDefLastId)
            }
            path if path_eq(path, &["scalar", "op", "reflect", "scalars"]) => {
                Some(Self::OpDefScalars)
            }
            _ => None,
        }
    }

    fn apply(self, scalar: &Scalar) -> tc_error::TCResult<Scalar> {
        Ok(match self {
            Self::ScalarClass => Scalar::Value(Value::Link(match &scalar {
                Scalar::Value(value) => class_link(value.class().path()),
                Scalar::Op(op) => class_from_opdef(op),
                Scalar::Ref(reference) => class_from_tcref(reference),
                Scalar::Map(_) => class_link(tc_ir::SCALAR_MAP),
                Scalar::Tuple(_) => class_link(tc_ir::SCALAR_TUPLE),
            })),
            Self::ScalarRefParts => Scalar::Tuple(match &scalar {
                Scalar::Ref(reference) => match reference.as_ref() {
                    tc_ir::TCRef::Cond(value) => vec![
                        Scalar::from(value.cond.clone()),
                        value.then.clone(),
                        value.or_else.clone(),
                    ],
                    tc_ir::TCRef::After(value) => vec![value.when.clone(), value.then.clone()],
                    tc_ir::TCRef::While(value) => vec![
                        value.cond.clone(),
                        value.closure.clone(),
                        value.state.clone(),
                    ],
                    tc_ir::TCRef::ForEach(value) => vec![
                        value.items.clone(),
                        value.op.clone(),
                        Scalar::Value(Value::String(value.item_name.to_string())),
                    ],
                    _ => Vec::new(),
                },
                _ => Vec::new(),
            }),
            Self::OpDefForm => Scalar::Tuple(
                opdef(scalar)?
                    .form()
                    .iter()
                    .map(|(id, scalar)| {
                        Scalar::Tuple(vec![
                            Scalar::Value(Value::String(id.to_string())),
                            scalar.clone(),
                        ])
                    })
                    .collect(),
            ),
            Self::OpDefLastId => Scalar::Value(
                opdef(scalar)?
                    .last_id()
                    .map(|id| Value::String(id.to_string()))
                    .unwrap_or(Value::None),
            ),
            Self::OpDefScalars => Scalar::Tuple(scalar_children(scalar)),
        })
    }
}

fn scalar_children(scalar: &Scalar) -> Vec<Scalar> {
    match &scalar {
        Scalar::Value(_) => Vec::new(),
        Scalar::Map(map) => map.values().cloned().collect(),
        Scalar::Tuple(tuple) => tuple.clone(),
        Scalar::Op(op) => op.form().iter().map(|(_, scalar)| scalar.clone()).collect(),
        Scalar::Ref(reference) => match reference.as_ref() {
            tc_ir::TCRef::Id(_) => Vec::new(),
            tc_ir::TCRef::Op(op) => match op {
                tc_ir::OpRef::Get((_, key)) | tc_ir::OpRef::Delete((_, key)) => vec![key.clone()],
                tc_ir::OpRef::Put((_, key, value)) => vec![key.clone(), value.clone()],
                tc_ir::OpRef::Post((_, params)) => params.values().cloned().collect(),
            },
            tc_ir::TCRef::Cond(value) => vec![
                Scalar::from(value.cond.clone()),
                value.then.clone(),
                value.or_else.clone(),
            ],
            tc_ir::TCRef::After(value) => vec![value.when.clone(), value.then.clone()],
            tc_ir::TCRef::While(value) => vec![
                value.cond.clone(),
                value.closure.clone(),
                value.state.clone(),
            ],
            tc_ir::TCRef::ForEach(value) => vec![value.items.clone(), value.op.clone()],
        },
    }
}

fn opdef(scalar: &Scalar) -> tc_error::TCResult<&OpDef> {
    match &scalar {
        Scalar::Op(opdef) => Ok(opdef),
        _ => Err(tc_error::TCError::bad_request(
            "expected OpDef scalar parameter",
        )),
    }
}

fn class_link(path: impl Into<pathlink::PathBuf>) -> Link {
    Link::from(path.into())
}

fn class_from_opdef(op: &OpDef) -> Link {
    class_link(match op {
        OpDef::Get(_) => tc_ir::OPDEF_GET,
        OpDef::Put(_) => tc_ir::OPDEF_PUT,
        OpDef::Post(_) => tc_ir::OPDEF_POST,
        OpDef::Delete(_) => tc_ir::OPDEF_DELETE,
    })
}

fn class_from_tcref(reference: &tc_ir::TCRef) -> Link {
    class_link(match &reference {
        tc_ir::TCRef::Cond(_) => tc_ir::TCREF_COND,
        tc_ir::TCRef::After(_) => tc_ir::TCREF_AFTER,
        tc_ir::TCRef::While(_) => tc_ir::TCREF_WHILE,
        tc_ir::TCRef::ForEach(_) => tc_ir::TCREF_FOR_EACH,
        tc_ir::TCRef::Id(_) => tc_ir::SCALAR_REF_PREFIX,
        tc_ir::TCRef::Op(op) => match op {
            tc_ir::OpRef::Get(_) => tc_ir::OPREF_GET,
            tc_ir::OpRef::Put(_) => tc_ir::OPREF_PUT,
            tc_ir::OpRef::Post(_) => tc_ir::OPREF_POST,
            tc_ir::OpRef::Delete(_) => tc_ir::OPREF_DELETE,
        },
    })
}

fn reflection_param(params: &Map<Scalar>) -> tc_error::TCResult<Scalar> {
    let scalar = "scalar".parse::<tc_ir::Id>().expect("static parameter id");
    let op = "op".parse::<tc_ir::Id>().expect("static parameter id");
    params
        .get(&scalar)
        .or_else(|| params.get(&op))
        .cloned()
        .ok_or_else(|| tc_error::TCError::bad_request("missing scalar parameter"))
}

impl<'a, Txn> Handler<'a, State<Txn>> for StaticAdd
where
    Txn: OpExecutor,
{
    fn get<'txn>(self: Box<Self>) -> Option<GetHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                static_number(key, |left, right| left.try_add(right)).map(State::from)
            })
        }))
    }

    fn post<'txn>(self: Box<Self>) -> Option<PostHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        Some(Box::new(|_txn, params| {
            Box::pin(async move {
                static_number(Scalar::Map(scalar_map(params)?), |left, right| {
                    left.try_add(right)
                })
                .map(State::from)
            })
        }))
    }
}

impl<'a, Txn> Handler<'a, State<Txn>> for StaticGreater
where
    Txn: OpExecutor,
{
    fn get<'txn>(self: Box<Self>) -> Option<GetHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                static_number(key, |left, right| {
                    Ok(number_general::Number::from(left > right))
                })
                .map(State::from)
            })
        }))
    }

    fn post<'txn>(self: Box<Self>) -> Option<PostHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        Some(Box::new(|_txn, params| {
            Box::pin(async move {
                static_number(Scalar::Map(scalar_map(params)?), |left, right| {
                    Ok(number_general::Number::from(left > right))
                })
                .map(State::from)
            })
        }))
    }
}

impl<'a, Txn> Handler<'a, State<Txn>> for Reflect
where
    Txn: OpExecutor,
{
    fn post<'txn>(self: Box<Self>) -> Option<PostHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |_txn, params| {
            Box::pin(async move {
                self.0
                    .apply(&reflection_param(&scalar_map(params)?)?)
                    .map(State::from)
            })
        }))
    }
}

fn scalar_map<Txn: tc_collection::StorageContext>(
    params: Map<State<Txn>>,
) -> tc_error::TCResult<Map<Scalar>> {
    params
        .into_iter()
        .map(|(id, state)| scalar_from_state(state).map(|scalar| (id, scalar)))
        .collect()
}

fn static_number(
    input: Scalar,
    apply: impl FnOnce(
        number_general::Number,
        number_general::Number,
    ) -> Result<number_general::Number, number_general::Error>,
) -> tc_error::TCResult<Scalar> {
    let params = match input {
        Scalar::Map(params) => params,
        Scalar::Tuple(items) if items.len() == 2 => [
            ("l".parse().expect("static id"), items[0].clone()),
            ("r".parse().expect("static id"), items[1].clone()),
        ]
        .into_iter()
        .collect(),
        _ => {
            return Err(tc_error::TCError::bad_request(
                "expected numeric parameters",
            ));
        }
    };

    let left = number_param(&params, "l")?;
    let right = number_param(&params, "r")?;
    Ok(Scalar::from(Value::Number(
        apply(left, right).map_err(tc_error::TCError::bad_request)?,
    )))
}

fn scalar_from_state<Txn: tc_collection::StorageContext>(
    state: State<Txn>,
) -> tc_error::TCResult<Scalar> {
    Scalar::try_cast_from(state, |_| {
        tc_error::TCError::bad_request("expected a scalar state request")
    })
}

fn number_param(params: &Map<Scalar>, name: &str) -> tc_error::TCResult<number_general::Number> {
    let id: tc_ir::Id = name.parse().expect("static parameter id");
    let scalar = params
        .get(&id)
        .ok_or_else(|| tc_error::TCError::bad_request(format!("missing {name} parameter")))?;

    match scalar {
        Scalar::Value(value) => number_general::Number::opt_cast_from(value),
        _ => None,
    }
    .ok_or_else(|| tc_error::TCError::bad_request(format!("expected {name} to be a number")))
}

impl<'a, Txn> Handler<'a, State<Txn>> for &'a ClassDef
where
    Txn: OpExecutor,
{
    fn get<'txn>(self: Box<Self>) -> Option<GetHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |_txn, parent| {
            Box::pin(async move {
                let instance = ClassInstance::new(State::from(parent), (*self).clone(), Map::new());
                Ok(State::Object(Box::new(Object::Instance(instance))))
            })
        }))
    }

    fn post<'txn>(self: Box<Self>) -> Option<PostHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |_txn, members| {
            Box::pin(async move {
                let instance = ClassInstance::new(State::default(), (*self).clone(), members);
                Ok(State::Object(Box::new(Object::Instance(instance))))
            })
        }))
    }
}

impl<Txn> Route<State<Txn>> for ClassDef
where
    Txn: OpExecutor,
{
    fn route<'a>(&'a self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        path.is_empty()
            .then(|| Box::new(self) as Box<dyn Handler<'a, State<Txn>>>)
    }
}

impl<Txn> Route<State<Txn>> for Object<Txn>
where
    Txn: OpExecutor,
{
    fn route<'a>(&'a self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        match self {
            Self::Class(class) => class.route(path),
            Self::Instance(instance) => route_instance(instance, path),
        }
    }
}

fn route_instance<'a, Txn>(
    instance: &'a ClassInstance<Txn>,
    path: &[PathSegment],
) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>>
where
    Txn: OpExecutor,
{
    let Some((_name, _suffix)) = path.split_first() else {
        return instance.parent().route(path);
    };

    Some(Box::new(InstanceHandler {
        instance: instance.clone(),
        path: path.to_vec(),
    }))
}

struct InstanceHandler<Txn: OpExecutor> {
    instance: ClassInstance<Txn>,
    path: Vec<PathSegment>,
}

enum OwnedMember<Txn: tc_collection::StorageContext> {
    Instance,
    State(State<Txn>),
    Method(Link, OpDef),
    Parent,
}

async fn instance_member<Txn: OpExecutor>(
    instance: &ClassInstance<Txn>,
    txn: &Txn,
    name: &tc_ir::Id,
) -> tc_error::TCResult<OwnedMember<Txn>> {
    let mut classes = BTreeMap::new();
    let mut class = instance.class();
    let mut visited = std::collections::BTreeSet::from([class.identity().clone()]);
    for _ in 0..super::MAX_INHERITANCE_DEPTH {
        let super::ClassParent::Class(parent) = class.parent() else {
            break;
        };

        if !visited.insert(parent.clone()) {
            return Err(tc_error::TCError::bad_request("Class inheritance cycle"));
        }

        let parent = txn.resolve_class(parent).await?;
        class = classes.entry(parent.identity().clone()).or_insert(parent);
    }

    if matches!(class.parent(), super::ClassParent::Class(_)) {
        return Err(tc_error::TCError::bad_request(
            "Class inheritance depth exceeded",
        ));
    }

    let member = match instance.resolve_member(name, &classes, |_, _, _| None) {
        Ok(super::ResolvedMember::State { .. }) => OwnedMember::Instance,
        Ok(super::ResolvedMember::Scalar { value, .. }) => {
            OwnedMember::State(State::from(value.clone()))
        }
        Ok(super::ResolvedMember::BoundMethod {
            declared_by,
            definition,
            ..
        }) => OwnedMember::Method(declared_by.clone(), definition.clone()),
        Err(super::ClassError::MissingMember { .. }) => OwnedMember::Parent,
        Err(error) => return Err(tc_error::TCError::bad_request(error)),
    };

    Ok(member)
}

// Structural selection owns no execution policy. Native leaves retain their
// concrete routes, and bound methods retain their original subject and origin.
enum Selected<Txn: tc_collection::StorageContext> {
    Native(State<Txn>, Vec<PathSegment>, String),
    Member(State<Txn>, String),
    Method(Box<BoundMethod<Txn>>, String),
}

impl<Txn: OpExecutor> InstanceHandler<Txn> {
    async fn select(self, txn: &Txn, observe_member: bool) -> tc_error::TCResult<Selected<Txn>> {
        let mut state = State::from(Object::Instance(self.instance));
        let mut path = self.path.as_slice();
        // Public routing preserves this path while Maps transparently consume
        // segments; an Instance delegation starts a new diagnostic boundary.
        let mut diagnostic = path;
        loop {
            match structural_step(&state, path) {
                StructuralStep::Parent(_) => state = take_parent(state),
                StructuralStep::Child(_) => {
                    state = take_member(state, path[0].as_str());
                    path = &path[1..];
                }
                StructuralStep::Member(member) => {
                    return Ok(Selected::Member(member.clone(), path_string(diagnostic)));
                }
                StructuralStep::Method(definition) => {
                    if path.len() != 1 {
                        return Err(tc_error::TCError::not_found(path_string(diagnostic)));
                    }

                    let definition = definition.clone();
                    return Ok(Selected::Method(
                        Box::new(BoundMethod::new(state, definition)),
                        path_string(diagnostic),
                    ));
                }
                StructuralStep::Instance(instance) => {
                    let name = path[0]
                        .as_str()
                        .parse::<tc_ir::Id>()
                        .map_err(tc_error::TCError::bad_request)?;

                    match instance_member(instance, txn, &name).await? {
                        OwnedMember::Parent => {
                            diagnostic = path;
                            state = take_parent(state);
                        }
                        OwnedMember::Instance => {
                            state = take_member(state, name.as_str());
                            path = &path[1..];
                            diagnostic = path;
                            if path.is_empty() && observe_member {
                                return Ok(Selected::Member(state, String::new()));
                            }
                        }
                        OwnedMember::State(member) => {
                            state = member;
                            path = &path[1..];
                            diagnostic = path;
                            if path.is_empty() && observe_member {
                                return Ok(Selected::Member(state, String::new()));
                            }
                        }
                        OwnedMember::Method(declared_by, definition) => {
                            if path.len() != 1 {
                                return Err(tc_error::TCError::not_found(path_string(path)));
                            }

                            return Ok(Selected::Method(
                                Box::new(BoundMethod {
                                    subject: state,
                                    definition,
                                    declared_by: Some(declared_by),
                                }),
                                "Class method".into(),
                            ));
                        }
                    }
                }
                StructuralStep::Native => {
                    return Ok(Selected::Native(
                        state,
                        path.to_vec(),
                        path_string(diagnostic),
                    ));
                }
            }
        }
    }

    async fn get(self, txn: &Txn, key: Scalar) -> tc_error::TCResult<State<Txn>> {
        match self.select(txn, true).await? {
            Selected::Member(state, _) => Ok(state),
            Selected::Native(state, path, diagnostic) => {
                let handler = state
                    .route(&path)
                    .ok_or_else(|| tc_error::TCError::not_found(diagnostic.clone()))?;

                let get = handler.get().ok_or_else(|| {
                    tc_error::TCError::method_not_allowed(tc_ir::Method::Get, diagnostic)
                })?;

                get(txn, key).await
            }
            Selected::Method(method, path) => {
                let get = method.get().ok_or_else(|| {
                    tc_error::TCError::method_not_allowed(tc_ir::Method::Get, path)
                })?;

                get(txn, key).await
            }
        }
    }

    async fn put(self, txn: &Txn, key: Scalar, value: State<Txn>) -> tc_error::TCResult<()> {
        match self.select(txn, false).await? {
            Selected::Member(_, path) => Err(tc_error::TCError::method_not_allowed(
                tc_ir::Method::Put,
                path,
            )),
            Selected::Native(state, path, diagnostic) => {
                let handler = state
                    .route(&path)
                    .ok_or_else(|| tc_error::TCError::not_found(diagnostic.clone()))?;

                let put = handler.put().ok_or_else(|| {
                    tc_error::TCError::method_not_allowed(tc_ir::Method::Put, diagnostic)
                })?;

                put(txn, key, value).await
            }
            Selected::Method(method, path) => {
                let put = method.put().ok_or_else(|| {
                    tc_error::TCError::method_not_allowed(tc_ir::Method::Put, path)
                })?;

                put(txn, key, value).await
            }
        }
    }

    async fn post(self, txn: &Txn, params: Map<State<Txn>>) -> tc_error::TCResult<State<Txn>> {
        match self.select(txn, false).await? {
            Selected::Member(_, path) => Err(tc_error::TCError::method_not_allowed(
                tc_ir::Method::Post,
                path,
            )),
            Selected::Native(state, path, diagnostic) => {
                let handler = state
                    .route(&path)
                    .ok_or_else(|| tc_error::TCError::not_found(diagnostic.clone()))?;

                let post = handler.post().ok_or_else(|| {
                    tc_error::TCError::method_not_allowed(tc_ir::Method::Post, diagnostic)
                })?;

                post(txn, params).await
            }
            Selected::Method(method, path) => {
                let post = method.post().ok_or_else(|| {
                    tc_error::TCError::method_not_allowed(tc_ir::Method::Post, path)
                })?;

                post(txn, params).await
            }
        }
    }

    async fn delete(self, txn: &Txn, key: Scalar) -> tc_error::TCResult<()> {
        match self.select(txn, false).await? {
            Selected::Member(_, path) => Err(tc_error::TCError::method_not_allowed(
                tc_ir::Method::Delete,
                path,
            )),
            Selected::Native(state, path, diagnostic) => {
                let handler = state
                    .route(&path)
                    .ok_or_else(|| tc_error::TCError::not_found(diagnostic.clone()))?;

                let delete = handler.delete().ok_or_else(|| {
                    tc_error::TCError::method_not_allowed(tc_ir::Method::Delete, diagnostic)
                })?;

                delete(txn, key).await
            }
            Selected::Method(method, path) => {
                let delete = method.delete().ok_or_else(|| {
                    tc_error::TCError::method_not_allowed(tc_ir::Method::Delete, path)
                })?;

                delete(txn, key).await
            }
        }
    }
}

fn take_parent<Txn: tc_collection::StorageContext>(state: State<Txn>) -> State<Txn> {
    let State::Object(object) = state else {
        unreachable!("selected instance")
    };

    let Object::Instance(instance) = *object else {
        unreachable!("selected instance")
    };

    instance.into_parts().0
}

fn take_member<Txn: tc_collection::StorageContext>(state: State<Txn>, name: &str) -> State<Txn> {
    let mut members = match state {
        State::Map(members) => members,
        State::Object(object) => {
            let Object::Instance(instance) = *object else {
                unreachable!("selected instance")
            };

            instance.into_parts().2
        }
        _ => unreachable!("selected member owner"),
    };

    members.remove(name).expect("selected member")
}

impl<'a, Txn: OpExecutor> Handler<'a, State<Txn>> for InstanceHandler<Txn> {
    fn get<'txn>(self: Box<Self>) -> Option<GetHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, key| Box::pin((*self).get(txn, key))))
    }

    fn put<'txn>(self: Box<Self>) -> Option<PutHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, key, value| {
            Box::pin((*self).put(txn, key, value))
        }))
    }

    fn post<'txn>(self: Box<Self>) -> Option<PostHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, params| {
            Box::pin((*self).post(txn, params))
        }))
    }

    fn delete<'txn>(self: Box<Self>) -> Option<DeleteHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, key| Box::pin((*self).delete(txn, key))))
    }
}

fn path_string(path: &[PathSegment]) -> String {
    path.iter().fold(String::new(), |mut path, segment| {
        path.push('/');
        path.push_str(segment.as_str());
        path
    })
}

// Both borrowed routing and transaction-dependent Class lookup use these same
// structural choices. Only Class ancestry resolution needs an asynchronous step.
enum StructuralStep<'a, Txn: tc_collection::StorageContext> {
    Parent(&'a State<Txn>),
    Child(&'a State<Txn>),
    Member(&'a State<Txn>),
    Method(&'a OpDef),
    Instance(&'a ClassInstance<Txn>),
    Native,
}

fn structural_step<'a, Txn: tc_collection::StorageContext>(
    state: &'a State<Txn>,
    path: &[PathSegment],
) -> StructuralStep<'a, Txn> {
    match &state {
        State::Object(object) => match object.as_ref() {
            Object::Instance(instance) if path.is_empty() => {
                StructuralStep::Parent(instance.parent())
            }
            Object::Instance(instance) => StructuralStep::Instance(instance),
            Object::Class(_) => StructuralStep::Native,
        },
        State::Map(members) => {
            let Some((name, suffix)) = path.split_first() else {
                return StructuralStep::Native;
            };

            let Some(member) = members.get(name.as_str()) else {
                return StructuralStep::Native;
            };

            if let State::Scalar(Scalar::Op(definition)) = member {
                return StructuralStep::Method(definition);
            }

            if suffix.is_empty() && !matches!(&member, State::Chain(_) | State::Collection(_)) {
                StructuralStep::Member(member)
            } else {
                StructuralStep::Child(member)
            }
        }
        _ => StructuralStep::Native,
    }
}

impl<Txn> Route<State<Txn>> for State<Txn>
where
    Txn: OpExecutor,
{
    fn route<'a>(&'a self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        let mut state = self;
        let mut remaining = path;
        loop {
            match structural_step(state, remaining) {
                StructuralStep::Parent(parent) => state = parent,
                StructuralStep::Child(member) => {
                    state = member;
                    remaining = &remaining[1..];
                }
                StructuralStep::Member(member) => {
                    return Some(Box::new(MemberValue(member.clone())));
                }
                StructuralStep::Method(definition) => {
                    return (remaining.len() == 1).then(|| {
                        Box::new(BoundMethod::new(state.clone(), definition.clone()))
                            as Box<dyn Handler<'a, State<Txn>>>
                    });
                }
                StructuralStep::Instance(instance) => return route_instance(instance, remaining),
                StructuralStep::Native => {
                    return match &state {
                        State::Collection(collection) => collection.route(remaining),
                        State::Chain(chain) => chain.route(remaining),
                        State::Object(object) => match object.as_ref() {
                            Object::Class(class) => class.route(remaining),
                            Object::Instance(_) => unreachable!("instance selected above"),
                        },
                        State::None => None,
                        State::Scalar(scalar) => match &scalar {
                            Scalar::Tuple(_) => TupleRoute(state).handler(remaining),
                            Scalar::Map(_) => MapRoute(state).handler(remaining),
                            Scalar::Value(value) => match &value {
                                Value::Number(_) => NumberRoute(state).handler(remaining),
                                Value::String(_) => StringRoute(state).handler(remaining),
                                _ => ValueRoute(state).handler(remaining),
                            },
                            _ => None,
                        },
                        State::Map(_) => MapRoute(state).handler(remaining),
                        State::Tuple(_) => TupleRoute(state).handler(remaining),
                    };
                }
            }
        }
    }
}

macro_rules! route_handler {
    ($handler:ident, $subject:expr) => {
        Some(Box::new($handler($subject)) as Box<dyn Handler<'_, State<Txn>>>)
    };
}

struct MemberValue<Txn: tc_collection::StorageContext>(State<Txn>);

impl<'a, Txn: OpExecutor> Handler<'a, State<Txn>> for MemberValue<Txn> {
    fn get<'txn>(self: Box<Self>) -> Option<GetHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |_, _| Box::pin(async move { Ok(self.0) })))
    }
}

/// Route a scalar application attribute, constructing its native `$self` only
/// when binding an operation definition. Operation references evaluate on GET;
/// other references remain literal values. Native map observations keep their
/// own value-returning semantics.
pub fn route_scalar<'a, Txn: OpExecutor>(
    scalar: &Scalar,
    subject: impl FnOnce() -> State<Txn>,
) -> Box<dyn Handler<'a, State<Txn>> + 'a> {
    match scalar.clone() {
        Scalar::Op(definition) => Box::new(BoundMethod::new(subject(), definition)),
        Scalar::Ref(reference) => match *reference {
            tc_ir::TCRef::Op(op) => Box::new(RefHandler(op)),
            reference => Box::new(MemberValue(State::from(Scalar::from(reference)))),
        },
        value => Box::new(MemberValue(State::from(value))),
    }
}

struct RefHandler(tc_ir::OpRef);

impl<'a, Txn: OpExecutor> Handler<'a, State<Txn>> for RefHandler {
    fn get<'txn>(self: Box<Self>) -> Option<GetHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, _key| {
            Box::pin(super::resolve_ref(tc_ir::TCRef::Op(self.0), txn, None))
        }))
    }
}

/// An immutable operation definition bound to its native `$self` value.
#[derive(Clone)]
pub struct BoundMethod<Txn: tc_collection::StorageContext> {
    subject: State<Txn>,
    definition: OpDef,
    declared_by: Option<Link>,
}

impl<Txn: tc_collection::StorageContext> BoundMethod<Txn> {
    pub fn new(subject: State<Txn>, definition: OpDef) -> Self {
        Self {
            subject,
            definition,
            declared_by: None,
        }
    }
}

impl<'a, Txn: OpExecutor> Handler<'a, State<Txn>> for BoundMethod<Txn> {
    fn get<'txn>(self: Box<Self>) -> Option<GetHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        matches!(self.definition, OpDef::Get(_)).then(|| {
            Box::new(move |txn, key| {
                Box::pin(async move {
                    OpExecutor::execute_op(
                        txn,
                        self.definition,
                        State::from(key),
                        Some(self.subject),
                        self.declared_by,
                    )
                    .await
                }) as tc_ir::HandlerFuture<'a, State<Txn>>
            }) as GetHandler<'a, 'txn, State<Txn>>
        })
    }

    fn put<'txn>(self: Box<Self>) -> Option<PutHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        matches!(self.definition, OpDef::Put(_)).then(|| {
            Box::new(move |txn, key, value| {
                Box::pin(async move {
                    OpExecutor::execute_op(
                        txn,
                        self.definition,
                        State::Tuple(vec![State::from(key), value]),
                        Some(self.subject),
                        self.declared_by,
                    )
                    .await
                    .map(|_| ())
                }) as tc_ir::HandlerFuture<'a, ()>
            }) as PutHandler<'a, 'txn, State<Txn>>
        })
    }

    fn post<'txn>(self: Box<Self>) -> Option<PostHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        matches!(self.definition, OpDef::Post(_)).then(|| {
            Box::new(move |txn, params| {
                Box::pin(async move {
                    OpExecutor::execute_op(
                        txn,
                        self.definition,
                        State::Map(params),
                        Some(self.subject),
                        self.declared_by,
                    )
                    .await
                }) as tc_ir::HandlerFuture<'a, State<Txn>>
            }) as PostHandler<'a, 'txn, State<Txn>>
        })
    }

    fn delete<'txn>(self: Box<Self>) -> Option<DeleteHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        matches!(self.definition, OpDef::Delete(_)).then(|| {
            Box::new(move |txn, key| {
                Box::pin(async move {
                    OpExecutor::execute_op(
                        txn,
                        self.definition,
                        State::from(key),
                        Some(self.subject),
                        self.declared_by,
                    )
                    .await
                    .map(|_| ())
                }) as tc_ir::HandlerFuture<'a, ()>
            }) as DeleteHandler<'a, 'txn, State<Txn>>
        })
    }
}

struct ValueRoute<'a, Txn: tc_collection::StorageContext>(&'a State<Txn>);
struct NumberRoute<'a, Txn: tc_collection::StorageContext>(&'a State<Txn>);
struct StringRoute<'a, Txn: tc_collection::StorageContext>(&'a State<Txn>);
struct TupleRoute<'a, Txn: tc_collection::StorageContext>(&'a State<Txn>);
struct MapRoute<'a, Txn: tc_collection::StorageContext>(&'a State<Txn>);

impl<'a, Txn: OpExecutor> ValueRoute<'a, Txn> {
    fn handler(self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        let [method] = path else { return None };
        match method.as_str() {
            "eq" => route_handler!(Equal, self.0),
            _ => None,
        }
    }
}

impl<'a, Txn: OpExecutor> NumberRoute<'a, Txn> {
    fn handler(self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        let [method] = path else { return None };
        match method.as_str() {
            "add" => route_handler!(Add, self.0),
            "gt" => route_handler!(Greater, self.0),
            "and" => route_handler!(And, self.0),
            "or" => route_handler!(Or, self.0),
            "xor" => route_handler!(Xor, self.0),
            "not" => route_handler!(Not, self.0),
            _ => ValueRoute(self.0).handler(path),
        }
    }
}

impl<'a, Txn: OpExecutor> StringRoute<'a, Txn> {
    fn handler(self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        let [method] = path else { return None };
        match method.as_str() {
            "concat" => route_handler!(Concat, self.0),
            "render" => route_handler!(Render, self.0),
            _ => ValueRoute(self.0).handler(path),
        }
    }
}

impl<'a, Txn: OpExecutor> MapRoute<'a, Txn> {
    fn handler(self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        let [method] = path else { return None };
        match method.as_str() {
            "len" => route_handler!(Length, self.0),
            "get" => route_handler!(Lookup, self.0),
            "reduce" | "fold" => route_handler!(Fold, self.0),
            _ => None,
        }
    }
}

impl<'a, Txn: OpExecutor> TupleRoute<'a, Txn> {
    fn handler(self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        let [method] = path else { return None };
        match method.as_str() {
            "len" => route_handler!(Length, self.0),
            "get" => route_handler!(Lookup, self.0),
            "reduce" | "fold" => route_handler!(Fold, self.0),
            "slice" => route_handler!(Slice, self.0),
            "head" => route_handler!(Head, self.0),
            "tail" => route_handler!(Tail, self.0),
            "concat" => route_handler!(Concat, self.0),
            _ => None,
        }
    }
}

macro_rules! native_handler {
    ($handler:ident, $execute:ident) => {
        struct $handler<'a, Txn: OpExecutor>(&'a State<Txn>);

        impl<'a, Txn: OpExecutor> Handler<'a, State<Txn>> for $handler<'a, Txn> {
            fn post<'txn>(self: Box<Self>) -> Option<PostHandler<'a, 'txn, State<Txn>>>
            where
                'txn: 'a,
            {
                Some(Box::new(move |txn, params| {
                    Box::pin($execute(self.0, txn, params))
                }))
            }
        }
    };
}

native_handler!(Add, add);

native_handler!(Greater, greater);

native_handler!(Equal, equal);

native_handler!(And, and);

native_handler!(Or, or);

native_handler!(Xor, xor);

native_handler!(Not, not);

native_handler!(Length, length);

native_handler!(Head, head);

native_handler!(Tail, tail);

native_handler!(Slice, slice);

native_handler!(Concat, concat);

native_handler!(Render, render);

native_handler!(Lookup, get);

native_handler!(Fold, fold);

async fn add<Txn: OpExecutor>(
    subject: &State<Txn>,
    _: &Txn,
    mut params: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    Ok(State::from(Value::Number(
        number(subject, "subject")?
            .try_add(number(&take(&mut params, "r")?, "r")?)
            .map_err(tc_error::TCError::bad_request)?,
    )))
}

async fn greater<Txn: OpExecutor>(
    subject: &State<Txn>,
    _: &Txn,
    mut params: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    Ok(State::from(Value::Number(number_general::Number::from(
        number(subject, "subject")? > number(&take(&mut params, "r")?, "r")?,
    ))))
}

async fn equal<Txn: OpExecutor>(
    subject: &State<Txn>,
    _: &Txn,
    mut params: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    let left = value(subject, "subject")?;
    let right = take(&mut params, "r")?;
    let right = value(&right, "r")?;
    let equal = left == right;
    Ok(State::from(Value::Number(number_general::Number::from(
        equal,
    ))))
}

macro_rules! boolean {
    ($name:ident, $op:tt) => {
        async fn $name<Txn: OpExecutor>(subject: &State<Txn>, _: &Txn, mut params: Map<State<Txn>>) -> tc_error::TCResult<State<Txn>> {
            let zero = number_general::Number::from(0);
            let left = number(subject, "subject")? != zero;
            let right = number(&take(&mut params, "r")?, "r")? != zero;
            Ok(State::from(Value::Number(number_general::Number::from(left $op right))))
        }
    };
}

boolean!(and, &&);

boolean!(or, ||);

boolean!(xor, ^);

async fn not<Txn: OpExecutor>(
    subject: &State<Txn>,
    _: &Txn,
    _: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    Ok(State::from(Value::Number(number_general::Number::from(
        number(subject, "subject")? == number_general::Number::from(0),
    ))))
}

async fn length<Txn: OpExecutor>(
    subject: &State<Txn>,
    _: &Txn,
    _: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    Ok(State::from(Value::Number(number_general::Number::from(
        match subject {
            State::Tuple(items) => items.len(),
            State::Map(items) => items.len(),
            State::Scalar(Scalar::Tuple(items)) => items.len(),
            State::Scalar(Scalar::Map(items)) => items.len(),
            _ => {
                return Err(tc_error::TCError::bad_request(
                    "expected tuple or map for len",
                ))
            }
        } as u64,
    ))))
}

async fn head<Txn: OpExecutor>(
    subject: &State<Txn>,
    _: &Txn,
    _: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    items(subject.clone(), "head")?
        .into_iter()
        .next()
        .ok_or_else(|| tc_error::TCError::bad_request("cannot take head of empty tuple"))
}

async fn tail<Txn: OpExecutor>(
    subject: &State<Txn>,
    _: &Txn,
    _: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    Ok(State::Tuple(
        items(subject.clone(), "tail")?
            .into_iter()
            .skip(1)
            .collect(),
    ))
}

async fn slice<Txn: OpExecutor>(
    subject: &State<Txn>,
    _: &Txn,
    mut params: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    let items = items(subject.clone(), "slice")?;
    let len = items.len() as i64;
    let mut start = i64::from(
        IntType::I64
            .try_cast(number(&take(&mut params, "start")?, "start")?)
            .map_err(tc_error::TCError::bad_request)?,
    );

    let mut stop = i64::from(
        IntType::I64
            .try_cast(number(&take(&mut params, "stop")?, "stop")?)
            .map_err(tc_error::TCError::bad_request)?,
    );

    if start < 0 {
        start += len;
    }

    if stop < 0 {
        stop += len;
    }

    start = start.clamp(0, len);
    stop = stop.clamp(start, len);
    Ok(State::Tuple(
        items
            .into_iter()
            .skip(start as usize)
            .take((stop - start) as usize)
            .collect(),
    ))
}

async fn concat<Txn: OpExecutor>(
    subject: &State<Txn>,
    _: &Txn,
    mut params: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    let right = take(&mut params, "r")?;
    if let Some(left) = string(subject) {
        let right = string(&right)
            .ok_or_else(|| tc_error::TCError::bad_request("expected string concat parameter r"))?;

        return Ok(State::from(Value::String(format!("{left}{right}"))));
    }

    let mut left = items(subject.clone(), "concat")?;
    left.extend(items(right, "concat")?);
    Ok(State::Tuple(left))
}

async fn render<Txn: OpExecutor>(
    subject: &State<Txn>,
    _: &Txn,
    params: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    let mut rendered = string(subject)
        .ok_or_else(|| tc_error::TCError::bad_request("expected a string template"))?
        .to_string();

    for (key, state) in params {
        let replacement =
            Cow::<str>::opt_cast_from(value(&state, key.as_str())?).ok_or_else(|| {
                tc_error::TCError::bad_request("cannot render this value as a string parameter")
            })?;

        rendered = rendered.replace(&format!("{{{{{}}}}}", key.as_str()), &replacement);
    }

    Ok(State::from(Value::String(rendered)))
}

async fn get<Txn: OpExecutor>(
    subject: &State<Txn>,
    _: &Txn,
    mut params: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    lookup(subject, value(&take(&mut params, "i")?, "i")?)
}

async fn fold<Txn: OpExecutor>(
    subject: &State<Txn>,
    txn: &Txn,
    mut params: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    let item_name_state = take(&mut params, "item_name")?;
    let name = string(&item_name_state)
        .ok_or_else(|| tc_error::TCError::bad_request("expected item_name to be a string"))?;

    let item_name: tc_ir::Id = name
        .parse()
        .map_err(|error| tc_error::TCError::bad_request(format!("invalid item_name: {error}")))?;

    let op_state = take(&mut params, "op")?;
    let op = match op_state {
        State::Scalar(Scalar::Op(op)) => op,
        _ => return Err(tc_error::TCError::bad_request("expected op to be an OpDef")),
    };

    let mut state = take(&mut params, "value")?;
    for item in items(subject.clone(), "fold")? {
        let mut call = state_params(state)?;
        call.insert(item_name.clone(), item);
        state = txn
            .execute_op(op.clone(), State::Map(call), Some(subject.clone()), None)
            .await?;
    }

    Ok(state)
}

fn take<Txn: tc_collection::StorageContext>(
    params: &mut Map<State<Txn>>,
    name: &str,
) -> tc_error::TCResult<State<Txn>> {
    params
        .remove(&name.parse::<tc_ir::Id>().expect("static native parameter"))
        .ok_or_else(|| tc_error::TCError::bad_request(format!("missing {name} parameter")))
}

fn value<'a, Txn: tc_collection::StorageContext>(
    state: &'a State<Txn>,
    name: &str,
) -> tc_error::TCResult<&'a Value> {
    if let State::Scalar(Scalar::Value(value)) = state {
        return Ok(value);
    }

    Err(tc_error::TCError::bad_request(format!(
        "expected {name} to be a scalar value"
    )))
}

fn number<Txn: tc_collection::StorageContext>(
    state: &State<Txn>,
    name: &str,
) -> tc_error::TCResult<number_general::Number> {
    number_general::Number::opt_cast_from(value(state, name)?)
        .ok_or_else(|| tc_error::TCError::bad_request(format!("expected {name} to be a number")))
}

fn string<Txn: tc_collection::StorageContext>(state: &State<Txn>) -> Option<&str> {
    match state {
        State::Scalar(Scalar::Value(value)) => <&str>::opt_cast_from(value),
        _ => None,
    }
}

fn items<Txn: tc_collection::StorageContext>(
    state: State<Txn>,
    operation: &str,
) -> tc_error::TCResult<Vec<State<Txn>>> {
    match state {
        State::Tuple(items) => return Ok(items),
        State::Map(map) => {
            return Ok(map
                .into_iter()
                .map(|(id, _)| State::from(Value::from(id.to_string())))
                .collect());
        }
        State::Scalar(scalar) => match scalar {
            Scalar::Tuple(items) => {
                return Ok(items
                    .into_iter()
                    .map(|scalar| State::Scalar(scalar))
                    .collect());
            }
            Scalar::Map(map) => {
                return Ok(map
                    .into_iter()
                    .map(|(id, _)| State::from(Value::from(id.to_string())))
                    .collect());
            }
            _ => {}
        },
        _ => {}
    }

    Err(tc_error::TCError::bad_request(format!(
        "expected tuple or map for {operation}"
    )))
}

fn tuple_index(index: number_general::Number) -> tc_error::TCResult<usize> {
    let index = UIntType::U64
        .try_cast(index)
        .map_err(tc_error::TCError::bad_request)?;

    usize::try_from(u64::from(index)).map_err(tc_error::TCError::bad_request)
}

fn lookup<Txn: tc_collection::StorageContext>(
    state: &State<Txn>,
    key: &Value,
) -> tc_error::TCResult<State<Txn>> {
    match (&state, key) {
        (State::Tuple(items), Value::Number(index)) => {
            let index = tuple_index(*index)?;
            return items
                .get(index)
                .cloned()
                .ok_or_else(|| tc_error::TCError::bad_request("tuple index out of bounds"));
        }
        (State::Map(map), Value::String(key)) => {
            return map
                .get(&key.parse::<tc_ir::Id>().map_err(|error| {
                    tc_error::TCError::bad_request(format!("invalid map key: {error}"))
                })?)
                .cloned()
                .ok_or_else(|| tc_error::TCError::bad_request("map key not found"));
        }
        (State::Scalar(scalar), key) => match (&scalar, key) {
            (Scalar::Tuple(items), Value::Number(index)) => {
                let index = tuple_index(*index)?;
                return items
                    .get(index)
                    .cloned()
                    .map(|scalar| State::Scalar(scalar))
                    .ok_or_else(|| tc_error::TCError::bad_request("tuple index out of bounds"));
            }
            (Scalar::Map(map), Value::String(key)) => {
                return map
                    .get(&key.parse::<tc_ir::Id>().map_err(|error| {
                        tc_error::TCError::bad_request(format!("invalid map key: {error}"))
                    })?)
                    .cloned()
                    .map(|scalar| State::Scalar(scalar))
                    .ok_or_else(|| tc_error::TCError::bad_request("map key not found"));
            }
            _ => {}
        },
        _ => {}
    }

    Err(tc_error::TCError::bad_request("invalid native lookup"))
}

fn state_params<Txn: tc_collection::StorageContext>(
    state: State<Txn>,
) -> tc_error::TCResult<Map<State<Txn>>> {
    match state {
        State::Map(map) => Ok(map),
        State::Scalar(scalar) => match scalar {
            Scalar::Map(map) => Ok(map
                .into_iter()
                .map(|(id, scalar)| (id, State::Scalar(scalar)))
                .collect()),
            kind => Ok([(
                "state".parse().expect("static state id"),
                State::Scalar(kind),
            )]
            .into_iter()
            .collect()),
        },
        _ => Err(tc_error::TCError::bad_request(
            "expected map for reduce state",
        )),
    }
}

#[cfg(test)]
#[path = "../tests/unit/route.rs"]
mod tests;
