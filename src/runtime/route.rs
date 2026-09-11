//! Native public routing for universal state and user-defined objects.

use std::collections::BTreeMap;
use std::future::Future;
use std::marker::PhantomData;
use std::str::FromStr;

use pathlink::Link;
use pathlink::PathSegment;
use safecast::CastInto;
use tc_ir::{
    DeleteHandler, GetHandler, Handler, Map, OpDef, PostHandler, Public, PutHandler, Route, Scalar,
};
use tc_value::class::NativeClass;
use tc_value::Value;

use super::{ClassDef, ClassInstance, Object, State};

/// Host capability required to execute a bound Class method.
///
/// The transaction remains borrowed and host-owned. `tc-state` only performs
/// routing and binding; the kernel evaluates the [`OpDef`].
pub trait StateExecutor: tc_collection::StorageContext + Sized + 'static {
    fn resolve_class(
        &self,
        identity: &Link,
    ) -> impl Future<Output = tc_error::TCResult<ClassDef>> + Send;

    fn get(
        &self,
        target: Link,
        key: Scalar,
    ) -> impl Future<Output = tc_error::TCResult<State<Self>>> + Send;

    fn put(
        &self,
        target: Link,
        key: Scalar,
        value: State<Self>,
    ) -> impl Future<Output = tc_error::TCResult<()>> + Send;

    fn post(
        &self,
        target: Link,
        params: Map<State<Self>>,
    ) -> impl Future<Output = tc_error::TCResult<State<Self>>> + Send;

    fn delete(
        &self,
        target: Link,
        key: Scalar,
    ) -> impl Future<Output = tc_error::TCResult<()>> + Send;

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
    Txn: StateExecutor,
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
        let segments = path.iter().map(PathSegment::as_str).collect::<Vec<_>>();
        match segments.as_slice() {
            ["scalar", "reflect", "class"] => Some(Self::ScalarClass),
            ["scalar", "reflect", "ref_parts"] => Some(Self::ScalarRefParts),
            ["scalar", "op", "reflect", "form"] => Some(Self::OpDefForm),
            ["scalar", "op", "reflect", "last_id"] => Some(Self::OpDefLastId),
            ["scalar", "op", "reflect", "scalars"] => Some(Self::OpDefScalars),
            _ => None,
        }
    }

    fn apply(self, scalar: &Scalar) -> tc_error::TCResult<Scalar> {
        Ok(match self {
            Self::ScalarClass => Scalar::Value(Value::Link(match scalar {
                Scalar::Value(value) => class_link(value.class().path()),
                Scalar::Op(op) => class_from_opdef(op),
                Scalar::Ref(reference) => class_from_tcref(reference),
                Scalar::Map(_) => class_link(tc_ir::SCALAR_MAP),
                Scalar::Tuple(_) => class_link(tc_ir::SCALAR_TUPLE),
            })),
            Self::ScalarRefParts => Scalar::Tuple(match scalar {
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
    match scalar {
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
    match scalar {
        Scalar::Op(opdef) => Ok(opdef),
        _ => Err(tc_error::TCError::bad_request(
            "expected OpDef scalar parameter",
        )),
    }
}

fn class_link(path: impl Into<pathlink::PathBuf>) -> Link {
    Link::from_str(&path.into().to_string()).expect("IR class link")
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
    class_link(match reference {
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
    Txn: StateExecutor,
{
    fn get<'txn>(self: Box<Self>) -> Option<GetHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                static_number(key, |left, right| left + right).map(State::from_scalar)
            })
        }))
    }

    fn post<'txn>(self: Box<Self>) -> Option<PostHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        Some(Box::new(|_txn, params| {
            Box::pin(async move {
                static_number(Scalar::Map(scalar_map(params)?), |left, right| left + right)
                    .map(State::from_scalar)
            })
        }))
    }
}

impl<'a, Txn> Handler<'a, State<Txn>> for StaticGreater
where
    Txn: StateExecutor,
{
    fn get<'txn>(self: Box<Self>) -> Option<GetHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                static_number(key, |left, right| {
                    number_general::Number::from(left > right)
                })
                .map(State::from_scalar)
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
                    number_general::Number::from(left > right)
                })
                .map(State::from_scalar)
            })
        }))
    }
}

impl<'a, Txn> Handler<'a, State<Txn>> for Reflect
where
    Txn: StateExecutor,
{
    fn post<'txn>(self: Box<Self>) -> Option<PostHandler<'a, 'txn, State<Txn>>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |_txn, params| {
            Box::pin(async move {
                self.0
                    .apply(&reflection_param(&scalar_map(params)?)?)
                    .map(State::from_scalar)
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
    apply: impl FnOnce(number_general::Number, number_general::Number) -> number_general::Number,
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
    Ok(Scalar::from(Value::Number(apply(left, right))))
}

fn scalar_from_state<Txn: tc_collection::StorageContext>(
    state: State<Txn>,
) -> tc_error::TCResult<Scalar> {
    match state {
        State::None => Ok(Scalar::default()),
        State::Scalar(scalar) => Ok(scalar),
        State::Map(map) => map
            .into_iter()
            .map(|(id, state)| scalar_from_state(state).map(|scalar| (id, scalar)))
            .collect::<tc_error::TCResult<Map<_>>>()
            .map(Scalar::Map),
        State::Tuple(items) => items
            .into_iter()
            .map(scalar_from_state)
            .collect::<tc_error::TCResult<Vec<_>>>()
            .map(Scalar::Tuple),
        State::Collection(_) | State::Object(_) => Err(tc_error::TCError::bad_request(
            "expected a scalar state request",
        )),
    }
}

fn number_param(params: &Map<Scalar>, name: &str) -> tc_error::TCResult<number_general::Number> {
    let id: tc_ir::Id = name.parse().expect("static parameter id");
    match params.get(&id) {
        Some(Scalar::Value(Value::Number(value))) => Ok(*value),
        Some(_) => Err(tc_error::TCError::bad_request(format!(
            "expected {name} to be a number"
        ))),
        None => Err(tc_error::TCError::bad_request(format!(
            "missing {name} parameter"
        ))),
    }
}

impl<'a, Txn> Handler<'a, State<Txn>> for &'a ClassDef
where
    Txn: StateExecutor,
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
    Txn: StateExecutor,
{
    fn route<'a>(&'a self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        path.is_empty()
            .then(|| Box::new(self) as Box<dyn Handler<'a, State<Txn>>>)
    }
}

impl<Txn> Route<State<Txn>> for Object<Txn>
where
    Txn: StateExecutor,
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
    Txn: StateExecutor,
{
    let Some((_name, _suffix)) = path.split_first() else {
        return instance.parent().route(path);
    };

    Some(Box::new(InstanceHandler {
        instance: instance.clone(),
        path: path.to_vec(),
    }))
}

struct InstanceHandler<Txn: StateExecutor> {
    instance: ClassInstance<Txn>,
    path: Vec<PathSegment>,
}

enum OwnedMember<Txn: tc_collection::StorageContext> {
    State(State<Txn>),
    Method(Link, OpDef),
    Parent,
}

impl<Txn: StateExecutor> InstanceHandler<Txn> {
    async fn member(&self, txn: &Txn) -> tc_error::TCResult<(OwnedMember<Txn>, &[PathSegment])> {
        let (name, suffix) = self
            .path
            .split_first()
            .ok_or_else(|| tc_error::TCError::not_found("empty instance route"))?;
        let name = name
            .as_str()
            .parse::<tc_ir::Id>()
            .map_err(|err| tc_error::TCError::bad_request(err.to_string()))?;
        let mut classes = BTreeMap::new();
        let mut class = self.instance.class();
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

        let member = match self
            .instance
            .resolve_member(&name, &classes, |_, _, _| None)
        {
            Ok(super::ResolvedMember::State { value, .. }) => OwnedMember::State(value.clone()),
            Ok(super::ResolvedMember::Scalar { value, .. }) => {
                OwnedMember::State(State::from_scalar(value.clone()))
            }
            Ok(super::ResolvedMember::BoundMethod {
                declared_by,
                definition,
                ..
            }) => OwnedMember::Method(declared_by.clone(), definition.clone()),
            Err(super::ClassError::MissingMember { .. }) => OwnedMember::Parent,
            Err(error) => return Err(tc_error::TCError::bad_request(error.to_string())),
        };
        Ok((member, suffix))
    }

    fn subject(&self) -> State<Txn> {
        State::Object(Box::new(Object::Instance(self.instance.clone())))
    }

    async fn get(self, txn: &Txn, key: Scalar) -> tc_error::TCResult<State<Txn>> {
        let (member, suffix) = self.member(txn).await?;
        match member {
            OwnedMember::State(state) if suffix.is_empty() => Ok(state),
            OwnedMember::State(state) => state.get(txn, suffix, key).await,
            OwnedMember::Method(declared_by, definition)
                if suffix.is_empty() && matches!(definition, OpDef::Get(_)) =>
            {
                txn.execute_op(
                    definition,
                    State::from_scalar(key),
                    Some(self.subject()),
                    Some(declared_by),
                )
                .await
            }
            OwnedMember::Method(_, _) if suffix.is_empty() => Err(
                tc_error::TCError::method_not_allowed(tc_ir::Method::Get, "Class method"),
            ),
            OwnedMember::Method(_, _) => Err(tc_error::TCError::not_found(path_string(&self.path))),
            OwnedMember::Parent => self.instance.parent().get(txn, &self.path, key).await,
        }
    }

    async fn put(self, txn: &Txn, key: Scalar, value: State<Txn>) -> tc_error::TCResult<()> {
        let (member, suffix) = self.member(txn).await?;
        match member {
            OwnedMember::State(state) => state.put(txn, suffix, key, value).await,
            OwnedMember::Method(declared_by, definition)
                if suffix.is_empty() && matches!(definition, OpDef::Put(_)) =>
            {
                txn.execute_op(
                    definition,
                    State::Tuple(vec![State::from_scalar(key), value]),
                    Some(self.subject()),
                    Some(declared_by),
                )
                .await
                .map(|_| ())
            }
            OwnedMember::Method(_, _) if suffix.is_empty() => Err(
                tc_error::TCError::method_not_allowed(tc_ir::Method::Put, "Class method"),
            ),
            OwnedMember::Method(_, _) => Err(tc_error::TCError::not_found(path_string(&self.path))),
            OwnedMember::Parent => {
                self.instance
                    .parent()
                    .put(txn, &self.path, key, value)
                    .await
            }
        }
    }

    async fn post(self, txn: &Txn, params: Map<State<Txn>>) -> tc_error::TCResult<State<Txn>> {
        let (member, suffix) = self.member(txn).await?;
        match member {
            OwnedMember::State(state) => state.post(txn, suffix, params).await,
            OwnedMember::Method(declared_by, definition)
                if suffix.is_empty() && matches!(definition, OpDef::Post(_)) =>
            {
                txn.execute_op(
                    definition,
                    State::Map(params),
                    Some(self.subject()),
                    Some(declared_by),
                )
                .await
            }
            OwnedMember::Method(_, _) if suffix.is_empty() => Err(
                tc_error::TCError::method_not_allowed(tc_ir::Method::Post, "Class method"),
            ),
            OwnedMember::Method(_, _) => Err(tc_error::TCError::not_found(path_string(&self.path))),
            OwnedMember::Parent => self.instance.parent().post(txn, &self.path, params).await,
        }
    }

    async fn delete(self, txn: &Txn, key: Scalar) -> tc_error::TCResult<()> {
        let (member, suffix) = self.member(txn).await?;
        match member {
            OwnedMember::State(state) => state.delete(txn, suffix, key).await,
            OwnedMember::Method(declared_by, definition)
                if suffix.is_empty() && matches!(definition, OpDef::Delete(_)) =>
            {
                txn.execute_op(
                    definition,
                    State::from_scalar(key),
                    Some(self.subject()),
                    Some(declared_by),
                )
                .await
                .map(|_| ())
            }
            OwnedMember::Method(_, _) if suffix.is_empty() => Err(
                tc_error::TCError::method_not_allowed(tc_ir::Method::Delete, "Class method"),
            ),
            OwnedMember::Method(_, _) => Err(tc_error::TCError::not_found(path_string(&self.path))),
            OwnedMember::Parent => self.instance.parent().delete(txn, &self.path, key).await,
        }
    }
}

impl<'a, Txn: StateExecutor> Handler<'a, State<Txn>> for InstanceHandler<Txn> {
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

impl<Txn> Route<State<Txn>> for State<Txn>
where
    Txn: StateExecutor,
{
    fn route<'a>(&'a self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        match self {
            Self::Collection(collection) => collection.route(path),
            Self::Object(object) => object.route(path),
            Self::None => None,
            Self::Scalar(Scalar::Tuple(_)) => TupleRoute(self.clone()).handler(path),
            Self::Scalar(Scalar::Map(_)) => MapRoute(self.clone()).handler(path),
            Self::Scalar(Scalar::Value(Value::Number(_))) => {
                NumberRoute(self.clone()).handler(path)
            }
            Self::Scalar(Scalar::Value(Value::String(_))) => {
                StringRoute(self.clone()).handler(path)
            }
            Self::Scalar(Scalar::Value(_)) => ValueRoute(self.clone()).handler(path),
            Self::Scalar(_) => None,
            Self::Map(_) => MapRoute(self.clone()).handler(path),
            Self::Tuple(_) => TupleRoute(self.clone()).handler(path),
        }
    }
}

macro_rules! route_handler {
    ($handler:ident, $subject:expr) => {
        Some(Box::new($handler($subject)) as Box<dyn Handler<'_, State<Txn>>>)
    };
}

struct ValueRoute<Txn: tc_collection::StorageContext>(State<Txn>);
struct NumberRoute<Txn: tc_collection::StorageContext>(State<Txn>);
struct StringRoute<Txn: tc_collection::StorageContext>(State<Txn>);
struct TupleRoute<Txn: tc_collection::StorageContext>(State<Txn>);
struct MapRoute<Txn: tc_collection::StorageContext>(State<Txn>);

impl<Txn: StateExecutor> ValueRoute<Txn> {
    fn handler<'a>(self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        let [method] = path else { return None };
        match method.as_str() {
            "eq" => route_handler!(Equal, self.0),
            _ => None,
        }
    }
}

impl<Txn: StateExecutor> NumberRoute<Txn> {
    fn handler<'a>(self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
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

impl<Txn: StateExecutor> StringRoute<Txn> {
    fn handler<'a>(self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        let [method] = path else { return None };
        match method.as_str() {
            "concat" => route_handler!(Concat, self.0),
            "render" => route_handler!(Render, self.0),
            _ => ValueRoute(self.0).handler(path),
        }
    }
}

impl<Txn: StateExecutor> MapRoute<Txn> {
    fn handler<'a>(self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        let [method] = path else { return None };
        match method.as_str() {
            "len" => route_handler!(Length, self.0),
            "get" => route_handler!(Lookup, self.0),
            "reduce" | "fold" => route_handler!(Fold, self.0),
            _ => None,
        }
    }
}

impl<Txn: StateExecutor> TupleRoute<Txn> {
    fn handler<'a>(self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
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
        struct $handler<Txn: StateExecutor>(State<Txn>);

        impl<'a, Txn: StateExecutor> Handler<'a, State<Txn>> for $handler<Txn> {
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

async fn add<Txn: StateExecutor>(
    subject: State<Txn>,
    _: &Txn,
    mut params: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    Ok(State::from(Value::Number(
        number(&subject, "subject")? + number(&take(&mut params, "r")?, "r")?,
    )))
}

async fn greater<Txn: StateExecutor>(
    subject: State<Txn>,
    _: &Txn,
    mut params: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    Ok(State::from(Value::Number(number_general::Number::from(
        number(&subject, "subject")? > number(&take(&mut params, "r")?, "r")?,
    ))))
}

async fn equal<Txn: StateExecutor>(
    subject: State<Txn>,
    _: &Txn,
    mut params: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    let left = value(&subject, "subject")?;
    let right = value(&take(&mut params, "r")?, "r")?;
    let equal = match (&left, &right) {
        (Value::Link(left), Value::String(right)) => left == right,
        (Value::String(left), Value::Link(right)) => right == left,
        _ => left == right,
    };
    Ok(State::from(Value::Number(number_general::Number::from(
        equal,
    ))))
}

macro_rules! boolean {
    ($name:ident, $op:tt) => {
        async fn $name<Txn: StateExecutor>(subject: State<Txn>, _: &Txn, mut params: Map<State<Txn>>) -> tc_error::TCResult<State<Txn>> {
            let zero = number_general::Number::from(0);
            let left = number(&subject, "subject")? != zero;
            let right = number(&take(&mut params, "r")?, "r")? != zero;
            Ok(State::from(Value::Number(number_general::Number::from(left $op right))))
        }
    };
}
boolean!(and, &&);
boolean!(or, ||);
boolean!(xor, ^);

async fn not<Txn: StateExecutor>(
    subject: State<Txn>,
    _: &Txn,
    _: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    Ok(State::from(Value::Number(number_general::Number::from(
        number(&subject, "subject")? == number_general::Number::from(0),
    ))))
}

async fn length<Txn: StateExecutor>(
    subject: State<Txn>,
    _: &Txn,
    _: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    Ok(State::from(Value::Number(number_general::Number::from(
        items(subject, "len")?.len() as u64,
    ))))
}

async fn head<Txn: StateExecutor>(
    subject: State<Txn>,
    _: &Txn,
    _: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    items(subject, "head")?
        .into_iter()
        .next()
        .ok_or_else(|| tc_error::TCError::bad_request("cannot take head of empty tuple"))
}

async fn tail<Txn: StateExecutor>(
    subject: State<Txn>,
    _: &Txn,
    _: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    Ok(State::Tuple(
        items(subject, "tail")?.into_iter().skip(1).collect(),
    ))
}

async fn slice<Txn: StateExecutor>(
    subject: State<Txn>,
    _: &Txn,
    mut params: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    let items = items(subject, "slice")?;
    let len = items.len() as i64;
    let mut start: i64 = number(&take(&mut params, "start")?, "start")?.cast_into();
    let mut stop: i64 = number(&take(&mut params, "stop")?, "stop")?.cast_into();
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

async fn concat<Txn: StateExecutor>(
    subject: State<Txn>,
    _: &Txn,
    mut params: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    let right = take(&mut params, "r")?;
    if let Some(left) = string(&subject) {
        let right = string(&right)
            .ok_or_else(|| tc_error::TCError::bad_request("expected string concat parameter r"))?;
        return Ok(State::from(Value::String(format!("{left}{right}"))));
    }
    let mut left = items(subject.clone(), "concat")?;
    left.extend(items(right, "concat")?);
    Ok(State::Tuple(left))
}

async fn render<Txn: StateExecutor>(
    subject: State<Txn>,
    _: &Txn,
    params: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    let mut rendered = string(&subject)
        .ok_or_else(|| tc_error::TCError::bad_request("expected a string template"))?
        .to_string();
    for (key, state) in params {
        rendered = rendered.replace(
            &format!("{{{{{}}}}}", key.as_str()),
            &render_value(value(&state, key.as_str())?)?,
        );
    }
    Ok(State::from(Value::String(rendered)))
}

async fn get<Txn: StateExecutor>(
    subject: State<Txn>,
    _: &Txn,
    mut params: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    lookup(&subject, value(&take(&mut params, "i")?, "i")?)
}

async fn fold<Txn: StateExecutor>(
    subject: State<Txn>,
    txn: &Txn,
    mut params: Map<State<Txn>>,
) -> tc_error::TCResult<State<Txn>> {
    let item_name: tc_ir::Id = match take(&mut params, "item_name")? {
        State::Scalar(Scalar::Value(Value::String(name))) => name.parse().map_err(|error| {
            tc_error::TCError::bad_request(format!("invalid item_name: {error}"))
        })?,
        _ => {
            return Err(tc_error::TCError::bad_request(
                "expected item_name to be a string",
            ));
        }
    };
    let op = match take(&mut params, "op")? {
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

fn value<Txn: tc_collection::StorageContext>(
    state: &State<Txn>,
    name: &str,
) -> tc_error::TCResult<Value> {
    match state {
        State::Scalar(Scalar::Value(value)) => Ok(value.clone()),
        _ => Err(tc_error::TCError::bad_request(format!(
            "expected {name} to be a scalar value"
        ))),
    }
}

fn number<Txn: tc_collection::StorageContext>(
    state: &State<Txn>,
    name: &str,
) -> tc_error::TCResult<number_general::Number> {
    match value(state, name)? {
        Value::Number(number) => Ok(number),
        _ => Err(tc_error::TCError::bad_request(format!(
            "expected {name} to be a number"
        ))),
    }
}

fn string<Txn: tc_collection::StorageContext>(state: &State<Txn>) -> Option<&str> {
    match state {
        State::Scalar(Scalar::Value(Value::String(value))) => Some(value),
        _ => None,
    }
}

fn items<Txn: tc_collection::StorageContext>(
    state: State<Txn>,
    operation: &str,
) -> tc_error::TCResult<Vec<State<Txn>>> {
    match state {
        State::Tuple(items) => Ok(items),
        State::Scalar(Scalar::Tuple(items)) => Ok(items.into_iter().map(State::Scalar).collect()),
        State::Map(map) => Ok(map
            .into_iter()
            .map(|(id, _)| State::from(Value::String(id.to_string())))
            .collect()),
        State::Scalar(Scalar::Map(map)) => Ok(map
            .into_iter()
            .map(|(id, _)| State::from(Value::String(id.to_string())))
            .collect()),
        _ => Err(tc_error::TCError::bad_request(format!(
            "expected tuple or map for {operation}"
        ))),
    }
}

fn lookup<Txn: tc_collection::StorageContext>(
    state: &State<Txn>,
    key: Value,
) -> tc_error::TCResult<State<Txn>> {
    match (state, key) {
        (State::Tuple(items), Value::Number(index)) => {
            let index: usize = index.cast_into();
            items
                .get(index)
                .cloned()
                .ok_or_else(|| tc_error::TCError::bad_request("tuple index out of bounds"))
        }
        (State::Scalar(Scalar::Tuple(items)), Value::Number(index)) => {
            let index: usize = index.cast_into();
            items
                .get(index)
                .cloned()
                .map(State::Scalar)
                .ok_or_else(|| tc_error::TCError::bad_request("tuple index out of bounds"))
        }
        (State::Map(map), Value::String(key)) => map
            .get(&key.parse::<tc_ir::Id>().map_err(|error| {
                tc_error::TCError::bad_request(format!("invalid map key: {error}"))
            })?)
            .cloned()
            .ok_or_else(|| tc_error::TCError::bad_request("map key not found")),
        (State::Scalar(Scalar::Map(map)), Value::String(key)) => map
            .get(&key.parse::<tc_ir::Id>().map_err(|error| {
                tc_error::TCError::bad_request(format!("invalid map key: {error}"))
            })?)
            .cloned()
            .map(State::Scalar)
            .ok_or_else(|| tc_error::TCError::bad_request("map key not found")),
        _ => Err(tc_error::TCError::bad_request("invalid native lookup")),
    }
}

fn render_value(value: Value) -> tc_error::TCResult<String> {
    match value {
        Value::String(value) => Ok(value),
        Value::Number(value) => Ok(value.to_string()),
        Value::Link(value) => Ok(value.to_string()),
        Value::Bytes(_) | Value::Tuple(_) | Value::None => Err(tc_error::TCError::bad_request(
            "cannot render this value as a string parameter",
        )),
    }
}

fn state_params<Txn: tc_collection::StorageContext>(
    state: State<Txn>,
) -> tc_error::TCResult<Map<State<Txn>>> {
    match state {
        State::Map(map) => Ok(map),
        State::Scalar(Scalar::Map(map)) => Ok(map
            .into_iter()
            .map(|(id, value)| (id, State::Scalar(value)))
            .collect()),
        State::Scalar(scalar) => Ok([(
            "state".parse().expect("static state id"),
            State::Scalar(scalar),
        )]
        .into_iter()
        .collect()),
        _ => Err(tc_error::TCError::bad_request(
            "expected map for reduce state",
        )),
    }
}

#[cfg(test)]
mod tests {
    use tc_ir::Route;
    use tc_value::Value;

    use super::*;
    use crate::runtime::tests::TestTxn;

    impl StateExecutor for TestTxn {
        async fn resolve_class(&self, _identity: &Link) -> tc_error::TCResult<ClassDef> {
            Err(tc_error::TCError::not_found("test Class"))
        }

        async fn get(&self, _target: Link, _key: Scalar) -> tc_error::TCResult<State<Self>> {
            Err(tc_error::TCError::not_found("test target"))
        }

        async fn put(
            &self,
            _target: Link,
            _key: Scalar,
            _value: State<Self>,
        ) -> tc_error::TCResult<()> {
            Err(tc_error::TCError::not_found("test target"))
        }

        async fn post(
            &self,
            _target: Link,
            _params: Map<State<Self>>,
        ) -> tc_error::TCResult<State<Self>> {
            Err(tc_error::TCError::not_found("test target"))
        }

        async fn delete(&self, _target: Link, _key: Scalar) -> tc_error::TCResult<()> {
            Err(tc_error::TCError::not_found("test target"))
        }

        async fn execute_op(
            &self,
            definition: OpDef,
            _args: State<Self>,
            _subject: Option<State<Self>>,
            _declared_by: Option<Link>,
        ) -> tc_error::TCResult<State<Self>> {
            Ok(State::Scalar(Scalar::Op(definition)))
        }
    }

    #[tokio::test]
    async fn static_routes_native_number_operations() {
        let routes = Static::<TestTxn>::default();
        let path: pathlink::PathBuf = "/scalar/value/number/add".parse().unwrap();
        let handler = routes.route(&path).expect("number route");
        let get = handler.get().expect("GET handler");
        let txn = TestTxn::new();
        let result = get(
            &txn,
            Scalar::Tuple(vec![
                Scalar::from(Value::Number(2.into())),
                Scalar::from(Value::Number(3.into())),
            ]),
        )
        .await
        .expect("add");
        assert!(matches!(
            result,
            State::Scalar(Scalar::Value(Value::Number(number))) if number == 5.into()
        ));
    }

    #[tokio::test]
    async fn native_values_own_their_operations() {
        let txn = TestTxn::new();
        let number = State::from(Value::Number(2.into()));
        let mut params = Map::new();
        params.insert("r".parse().unwrap(), State::from(Value::Number(3.into())));
        let add_path = ["add".parse().unwrap()];
        let handler = number.route(&add_path).expect("number handler");
        let sum = handler.post().expect("POST handler")(&txn, params)
            .await
            .expect("add");
        assert!(matches!(sum, State::Scalar(Scalar::Value(Value::Number(n))) if n == 5.into()));

        let tuple = State::Tuple(vec![
            State::from(1_u64),
            State::from(2_u64),
            State::from(3_u64),
        ]);
        let mut params = Map::new();
        params.insert("start".parse().unwrap(), State::from(1_u64));
        params.insert("stop".parse().unwrap(), State::from(3_u64));
        let slice_path = ["slice".parse().unwrap()];
        let handler = tuple.route(&slice_path).expect("tuple handler");
        let slice = handler.post().expect("POST handler")(&txn, params)
            .await
            .expect("slice");
        assert!(matches!(slice, State::Tuple(items) if items.len() == 2));

        let map = State::Map(
            [("answer".parse().unwrap(), State::from(42_u64))]
                .into_iter()
                .collect(),
        );
        let mut params = Map::new();
        params.insert(
            "i".parse().unwrap(),
            State::from(Value::String("answer".into())),
        );
        let get_path = ["get".parse().unwrap()];
        let handler = map.route(&get_path).expect("map handler");
        let answer = handler.post().expect("POST handler")(&txn, params)
            .await
            .expect("get");
        assert!(matches!(answer, State::Scalar(Scalar::Value(Value::Number(n))) if n == 42.into()));
    }

    #[test]
    fn class_and_instance_are_native_state_routes() {
        let identity: Link = "/class/example-devco/example/1.0.0"
            .parse()
            .expect("class identity");
        let mut prototype = Map::new();
        prototype.insert(
            "call".parse().expect("method name"),
            Scalar::Op(OpDef::Post(vec![(
                "result".parse().unwrap(),
                Scalar::default(),
            )])),
        );
        let class = ClassDef::new(
            identity,
            [1; 32],
            super::super::ClassParent::Native(super::super::StateType::Tuple),
            prototype,
        );
        let class_state = State::<TestTxn>::Object(Box::new(Object::Class(class.clone())));
        assert!(class_state.route(&[]).is_some());

        let instance = ClassInstance::new(State::None, class, Map::new());
        let instance_state = State::<TestTxn>::Object(Box::new(Object::Instance(instance)));
        assert!(instance_state.route(&[]).is_none());
        assert!(instance_state
            .route(&["call".parse().expect("method path")])
            .is_some());
    }
}
