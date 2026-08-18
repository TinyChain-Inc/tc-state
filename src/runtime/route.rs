//! Native public routing for universal state and user-defined objects.

use pathlink::{Link, PathSegment};
use tc_ir::{Handler, Map, Method, OpDef, Route, Scalar};

use super::{ClassDef, ClassInstance, Object, State};

/// A bound-method request passed to the host graph executor.
pub enum MethodCall<Txn> {
    Get(Scalar),
    Put(Scalar, State<Txn>),
    Post(Map<State<Txn>>),
    Delete(Scalar),
}

/// Host capability required to execute a bound Class method.
///
/// The transaction remains borrowed and host-owned. `tc-state` only performs
/// routing and binding; the kernel evaluates the [`OpDef`].
#[tc_ir::async_trait]
pub trait ClassExecutor: tc_collection::StorageContext + Sized + 'static {
    async fn execute_bound(
        &self,
        instance: ClassInstance<Self, Link>,
        definition: OpDef,
        call: MethodCall<Self>,
    ) -> tc_error::TCResult<State<Self>>;
}

struct ClassHandler {
    class: ClassDef<Link>,
}

#[tc_ir::async_trait]
impl<Txn> Handler<State<Txn>> for ClassHandler
where
    Txn: ClassExecutor,
{
    async fn get(&self, _txn: &Txn, parent: Scalar) -> tc_error::TCResult<State<Txn>> {
        let instance = ClassInstance::new(State::from(parent), self.class.clone(), Map::new());
        Ok(State::Object(Box::new(Object::Instance(instance))))
    }

    async fn post(&self, _txn: &Txn, members: Map<State<Txn>>) -> tc_error::TCResult<State<Txn>> {
        let instance = ClassInstance::new(State::default(), self.class.clone(), members);
        Ok(State::Object(Box::new(Object::Instance(instance))))
    }
}

impl<Txn> Route<State<Txn>> for Object<Txn>
where
    Txn: ClassExecutor,
{
    fn route(&self, path: &[PathSegment]) -> Option<Box<dyn Handler<State<Txn>> + '_>> {
        match self {
            Self::Class(class) if path.is_empty() => Some(Box::new(ClassHandler {
                class: class.clone(),
            })),
            Self::Class(_) => None,
            Self::Instance(instance) => route_instance(instance, path),
        }
    }
}

fn route_instance<'a, Txn>(
    instance: &'a ClassInstance<Txn, Link>,
    path: &[PathSegment],
) -> Option<Box<dyn Handler<State<Txn>> + 'a>>
where
    Txn: ClassExecutor,
{
    let Some((name, suffix)) = path.split_first() else {
        return instance.parent().route(path);
    };
    let name = name.as_str().parse::<tc_ir::Id>().ok()?;

    if let Some(member) = instance.members().get(&name) {
        member.route(suffix)
    } else if suffix.is_empty()
        && matches!(instance.class().prototype().get(&name), Some(Scalar::Op(_)))
    {
        let Some(Scalar::Op(definition)) = instance.class().prototype().get(&name) else {
            unreachable!("prototype member was checked above")
        };
        Some(Box::new(BoundMethodHandler {
            instance: instance.clone(),
            definition: definition.clone(),
        }) as Box<dyn Handler<State<Txn>>>)
    } else {
        // Prototype OpDefs are resolved to `ResolvedMember::BoundMethod` by the
        // graph executor. Native routing must not execute an OpDef or acquire a
        // transaction on its own, so unresolved paths continue to the concrete
        // parent state's native behavior.
        instance.parent().route(path)
    }
}

impl<Txn> Route<State<Txn>> for State<Txn>
where
    Txn: ClassExecutor,
{
    fn route(&self, path: &[PathSegment]) -> Option<Box<dyn Handler<State<Txn>> + '_>> {
        match self {
            Self::Collection(collection) => collection.route(path),
            Self::Object(object) => object.route(path),
            Self::None | Self::Scalar(_) | Self::Map(_) | Self::Tuple(_) => None,
        }
    }
}

struct BoundMethodHandler<Txn: ClassExecutor> {
    instance: ClassInstance<Txn, Link>,
    definition: OpDef,
}

#[tc_ir::async_trait]
impl<Txn: ClassExecutor> Handler<State<Txn>> for BoundMethodHandler<Txn> {
    async fn get(&self, txn: &Txn, key: Scalar) -> tc_error::TCResult<State<Txn>> {
        if !matches!(self.definition, OpDef::Get(_)) {
            return Err(tc_error::TCError::method_not_allowed(
                Method::Get,
                "Class method",
            ));
        }
        txn.execute_bound(
            self.instance.clone(),
            self.definition.clone(),
            MethodCall::Get(key),
        )
        .await
    }

    async fn put(&self, txn: &Txn, key: Scalar, value: State<Txn>) -> tc_error::TCResult<()> {
        if !matches!(self.definition, OpDef::Put(_)) {
            return Err(tc_error::TCError::method_not_allowed(
                Method::Put,
                "Class method",
            ));
        }
        txn.execute_bound(
            self.instance.clone(),
            self.definition.clone(),
            MethodCall::Put(key, value),
        )
        .await
        .map(|_| ())
    }

    async fn post(&self, txn: &Txn, params: Map<State<Txn>>) -> tc_error::TCResult<State<Txn>> {
        if !matches!(self.definition, OpDef::Post(_)) {
            return Err(tc_error::TCError::method_not_allowed(
                Method::Post,
                "Class method",
            ));
        }
        txn.execute_bound(
            self.instance.clone(),
            self.definition.clone(),
            MethodCall::Post(params),
        )
        .await
    }

    async fn delete(&self, txn: &Txn, key: Scalar) -> tc_error::TCResult<()> {
        if !matches!(self.definition, OpDef::Delete(_)) {
            return Err(tc_error::TCError::method_not_allowed(
                Method::Delete,
                "Class method",
            ));
        }
        txn.execute_bound(
            self.instance.clone(),
            self.definition.clone(),
            MethodCall::Delete(key),
        )
        .await
        .map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use tc_ir::Route;

    use super::*;
    use crate::runtime::tests::TestTxn;

    #[tc_ir::async_trait]
    impl ClassExecutor for TestTxn {
        async fn execute_bound(
            &self,
            _instance: ClassInstance<Self, Link>,
            definition: OpDef,
            _call: MethodCall<Self>,
        ) -> tc_error::TCResult<State<Self>> {
            Ok(State::Scalar(Scalar::Op(definition)))
        }
    }

    #[test]
    fn class_and_instance_are_native_state_routes() {
        let identity: Link = "/lib/example/Class".parse().expect("class identity");
        let mut prototype = Map::new();
        prototype.insert(
            "call".parse().expect("method name"),
            Scalar::Op(OpDef::Post(Vec::new())),
        );
        let class = ClassDef::new(
            identity,
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
