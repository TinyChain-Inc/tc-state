//! User-defined Class and instance semantics.
//!
//! The versioned wire representation is owned by `tcv2#68`. These runtime
//! types intentionally have no `destream` implementation until that contract is
//! accepted.

use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;

use tc_ir::{Id, Map, Scalar};

use super::{State, StateType};

/// A user-defined Class definition or concrete instance carried by [`State`].
#[derive(Clone, Debug)]
pub enum Object<Txn> {
    Class(ClassDef<pathlink::Link>),
    Instance(ClassInstance<Txn, pathlink::Link>),
}

impl<Txn> From<ClassDef<pathlink::Link>> for Object<Txn> {
    fn from(class: ClassDef<pathlink::Link>) -> Self {
        Self::Class(class)
    }
}

impl<Txn> From<ClassInstance<Txn, pathlink::Link>> for Object<Txn> {
    fn from(instance: ClassInstance<Txn, pathlink::Link>) -> Self {
        Self::Instance(instance)
    }
}

/// The default maximum number of user-defined classes visited during lookup.
pub const MAX_INHERITANCE_DEPTH: usize = 64;

/// A Class parent, distinguishing native extension from user-defined extension.
#[derive(Clone, Debug, PartialEq)]
pub enum ClassParent<I> {
    Native(StateType),
    Class(I),
}

/// An immutable user-defined Class definition.
///
/// `I` is supplied by the authoritative manifest/identity layer. Keeping it
/// generic prevents this crate from inventing the digest format owned by
/// `tcv2#68`.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassDef<I> {
    identity: I,
    parent: ClassParent<I>,
    prototype: Map<Scalar>,
}

impl<I> ClassDef<I> {
    pub fn new(identity: I, parent: ClassParent<I>, prototype: Map<Scalar>) -> Self {
        Self {
            identity,
            parent,
            prototype,
        }
    }

    pub fn identity(&self) -> &I {
        &self.identity
    }

    pub fn parent(&self) -> &ClassParent<I> {
        &self.parent
    }

    pub fn prototype(&self) -> &Map<Scalar> {
        &self.prototype
    }
}

impl<I: Hash> Hash for ClassDef<I> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // The identity layer binds this ID to the complete canonical manifest.
        // Hashing only the bound identity avoids defining a second manifest
        // serialization in tc-state.
        self.identity.hash(state);
    }
}

/// Resolve an immutable Class identity without performing I/O.
pub trait ClassResolver<I> {
    fn resolve(&self, identity: &I) -> Option<&ClassDef<I>>;
}

impl<I: Eq + Hash> ClassResolver<I> for std::collections::HashMap<I, ClassDef<I>> {
    fn resolve(&self, identity: &I) -> Option<&ClassDef<I>> {
        self.get(identity)
    }
}

impl<I: Ord> ClassResolver<I> for std::collections::BTreeMap<I, ClassDef<I>> {
    fn resolve(&self, identity: &I) -> Option<&ClassDef<I>> {
        self.get(identity)
    }
}

/// A concrete instance of a user-defined Class.
#[derive(Clone, Debug)]
pub struct ClassInstance<Txn, I> {
    parent: Box<State<Txn>>,
    class: ClassDef<I>,
    members: Map<State<Txn>>,
}

impl<Txn, I> ClassInstance<Txn, I> {
    pub fn new(parent: State<Txn>, class: ClassDef<I>, members: Map<State<Txn>>) -> Self {
        Self {
            parent: Box::new(parent),
            class,
            members,
        }
    }

    pub fn parent(&self) -> &State<Txn> {
        &self.parent
    }

    pub fn class(&self) -> &ClassDef<I> {
        &self.class
    }

    pub fn members(&self) -> &Map<State<Txn>> {
        &self.members
    }
}

/// The source of a resolved member.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemberSource {
    Instance,
    Prototype,
    Native,
}

/// A resolved member, with method definitions bound to their instance `self`.
#[derive(Debug)]
pub enum ResolvedMember<'a, Txn, I> {
    State {
        source: MemberSource,
        value: &'a State<Txn>,
    },
    Scalar {
        source: MemberSource,
        value: &'a Scalar,
    },
    BoundMethod {
        definition: &'a tc_ir::OpDef,
        instance: &'a ClassInstance<Txn, I>,
    },
}

impl<Txn, I> ResolvedMember<'_, Txn, I> {
    pub fn source(&self) -> MemberSource {
        match self {
            Self::State { source, .. } | Self::Scalar { source, .. } => *source,
            Self::BoundMethod { .. } => MemberSource::Prototype,
        }
    }
}

/// Typed Class construction and lookup failures.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ClassError {
    MalformedDefinition(String),
    InvalidParent(String),
    UnsupportedOverride { member: Id },
    MissingMember { member: Id },
    InheritanceCycle,
    InheritanceDepthExceeded { limit: usize },
}

impl fmt::Display for ClassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MalformedDefinition(reason) => write!(f, "malformed Class definition: {reason}"),
            Self::InvalidParent(parent) => write!(f, "invalid Class parent: {parent}"),
            Self::UnsupportedOverride { member } => {
                write!(f, "unsupported override of Class member {member}")
            }
            Self::MissingMember { member } => write!(f, "missing Class member {member}"),
            Self::InheritanceCycle => f.write_str("Class inheritance cycle"),
            Self::InheritanceDepthExceeded { limit } => {
                write!(f, "Class inheritance exceeds depth limit {limit}")
            }
        }
    }
}

impl std::error::Error for ClassError {}

impl<Txn, I> ClassInstance<Txn, I>
where
    I: Clone + Eq + Hash,
{
    /// Resolve `member` without evaluating references or projecting `parent`.
    ///
    /// Lookup order is instance member, concrete prototype, nearest inherited
    /// prototype, and finally native behavior. The native callback is invoked
    /// only after structural lookup is exhausted and receives no transaction.
    pub fn resolve_member<'a, R, N>(
        &'a self,
        member: &Id,
        classes: &'a R,
        native: N,
    ) -> Result<ResolvedMember<'a, Txn, I>, ClassError>
    where
        R: ClassResolver<I>,
        N: FnOnce(Option<&StateType>, &'a State<Txn>, &Id) -> Option<&'a State<Txn>>,
    {
        self.resolve_member_with_limit(member, classes, native, MAX_INHERITANCE_DEPTH)
    }

    pub fn resolve_member_with_limit<'a, R, N>(
        &'a self,
        member: &Id,
        classes: &'a R,
        native: N,
        limit: usize,
    ) -> Result<ResolvedMember<'a, Txn, I>, ClassError>
    where
        R: ClassResolver<I>,
        N: FnOnce(Option<&StateType>, &'a State<Txn>, &Id) -> Option<&'a State<Txn>>,
    {
        if let Some(value) = self.members.get(member) {
            return Ok(ResolvedMember::State {
                source: MemberSource::Instance,
                value,
            });
        }

        let mut class = &self.class;
        let mut visited = HashSet::new();
        let native_parent = 'inheritance: loop {
            let depth = visited.len();
            if depth == limit {
                return Err(ClassError::InheritanceDepthExceeded { limit });
            }

            if !visited.insert(class.identity.clone()) {
                return Err(ClassError::InheritanceCycle);
            }

            if let Some(value) = class.prototype.get(member) {
                return Ok(bind(self, value));
            }

            match &class.parent {
                ClassParent::Native(parent) => {
                    break 'inheritance Some(parent);
                }
                ClassParent::Class(parent) => {
                    class = classes
                        .resolve(parent)
                        .ok_or_else(|| ClassError::InvalidParent("unknown identity".into()))?;
                }
            }
        };

        native(native_parent, &self.parent, member)
            .map(|value| ResolvedMember::State {
                source: MemberSource::Native,
                value,
            })
            .ok_or_else(|| ClassError::MissingMember {
                member: member.clone(),
            })
    }
}

impl<I> ClassDef<I>
where
    I: Clone + Eq + Hash,
{
    /// Validate the inheritance chain and reject method/value kind changes.
    pub fn validate<R: ClassResolver<I>>(&self, classes: &R) -> Result<(), ClassError> {
        self.validate_with_limit(classes, MAX_INHERITANCE_DEPTH)
    }

    pub fn validate_with_limit<R: ClassResolver<I>>(
        &self,
        classes: &R,
        limit: usize,
    ) -> Result<(), ClassError> {
        let mut visited = HashSet::new();
        let mut class = self;
        let mut inherited = std::collections::HashMap::<Id, bool>::new();

        for depth in 0..=limit {
            if depth == limit {
                return Err(ClassError::InheritanceDepthExceeded { limit });
            }
            if !visited.insert(class.identity.clone()) {
                return Err(ClassError::InheritanceCycle);
            }

            for (member, value) in &class.prototype {
                let method = matches!(value, Scalar::Op(_));
                if let Some(child_method) = inherited.get(member) {
                    if *child_method != method {
                        return Err(ClassError::UnsupportedOverride {
                            member: member.clone(),
                        });
                    }
                } else {
                    inherited.insert(member.clone(), method);
                }
            }

            match &class.parent {
                ClassParent::Native(_) => return Ok(()),
                ClassParent::Class(parent) => {
                    class = classes
                        .resolve(parent)
                        .ok_or_else(|| ClassError::InvalidParent("unknown identity".into()))?;
                }
            }
        }

        unreachable!("bounded validation loop always returns")
    }
}

fn bind<'a, Txn, I>(
    instance: &'a ClassInstance<Txn, I>,
    value: &'a Scalar,
) -> ResolvedMember<'a, Txn, I> {
    match value {
        Scalar::Op(definition) => ResolvedMember::BoundMethod {
            definition,
            instance,
        },
        value => ResolvedMember::Scalar {
            source: MemberSource::Prototype,
            value,
        },
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{hash_map::DefaultHasher, HashMap};
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tc_value::Value;

    use super::*;

    type TestState = State<()>;

    fn id(value: &str) -> Id {
        value.parse().expect("Id")
    }

    fn scalar(value: u64) -> Scalar {
        Scalar::from(Value::from(value))
    }

    fn class(
        name: &str,
        parent: ClassParent<String>,
        members: &[(&str, Scalar)],
    ) -> ClassDef<String> {
        let prototype = members
            .iter()
            .cloned()
            .map(|(name, value)| (id(name), value))
            .collect();
        ClassDef::new(name.into(), parent, prototype)
    }

    fn instance(
        class: ClassDef<String>,
        members: &[(&str, TestState)],
    ) -> ClassInstance<(), String> {
        let members = members
            .iter()
            .cloned()
            .map(|(name, value)| (id(name), value))
            .collect();
        ClassInstance::new(TestState::None, class, members)
    }

    #[test]
    fn instance_then_prototype_then_parent_then_native() {
        let base = class(
            "base",
            ClassParent::Native(StateType::Tuple),
            &[("base", scalar(2)), ("overridden", scalar(3))],
        );
        let derived = class(
            "derived",
            ClassParent::Class("base".into()),
            &[("prototype", scalar(4)), ("overridden", scalar(5))],
        );
        let instance = instance(
            derived,
            &[
                ("instance", TestState::from(6_u64)),
                ("overridden", TestState::from(7_u64)),
            ],
        );
        let classes = HashMap::from([("base".into(), base)]);

        assert_eq!(
            instance
                .resolve_member(&id("instance"), &classes, |_, _, _| None)
                .expect("instance member")
                .source(),
            MemberSource::Instance
        );
        assert_eq!(
            instance
                .resolve_member(&id("prototype"), &classes, |_, _, _| None)
                .expect("prototype member")
                .source(),
            MemberSource::Prototype
        );
        assert_eq!(
            instance
                .resolve_member(&id("base"), &classes, |_, _, _| None)
                .expect("inherited member")
                .source(),
            MemberSource::Prototype
        );
        assert_eq!(
            instance
                .resolve_member(&id("overridden"), &classes, |_, _, _| None)
                .expect("override")
                .source(),
            MemberSource::Instance
        );

        let native_value = TestState::from(8_u64);
        let resolved = instance
            .resolve_member(&id("native"), &classes, |parent, _, _| {
                assert_eq!(parent, Some(&StateType::Tuple));
                Some(&native_value)
            })
            .expect("native member");
        assert_eq!(resolved.source(), MemberSource::Native);
    }

    #[test]
    fn prototype_method_is_bound_to_exact_instance() {
        let method = tc_ir::OpDef::Post(Vec::new());
        let class = class(
            "class",
            ClassParent::Native(StateType::Tuple),
            &[("call", Scalar::Op(method))],
        );
        let instance = instance(class, &[]);
        let classes = HashMap::new();
        let resolved = instance
            .resolve_member(&id("call"), &classes, |_, _, _| None)
            .expect("bound method");

        let ResolvedMember::BoundMethod {
            instance: bound, ..
        } = resolved
        else {
            panic!("expected bound method");
        };
        assert!(std::ptr::eq(bound, &instance));
    }

    #[test]
    fn invalid_parent_cycle_depth_and_missing_member_are_typed() {
        let derived = class("derived", ClassParent::Class("missing".into()), &[]);
        let invalid_instance = instance(derived, &[]);
        assert!(matches!(
            invalid_instance.resolve_member(&id("x"), &HashMap::new(), |_, _, _| None),
            Err(ClassError::InvalidParent(_))
        ));

        let a = class("a", ClassParent::Class("b".into()), &[]);
        let b = class("b", ClassParent::Class("a".into()), &[]);
        let cycle_instance = instance(a.clone(), &[]);
        let classes = HashMap::from([("a".into(), a), ("b".into(), b)]);
        assert!(matches!(
            cycle_instance.resolve_member(&id("x"), &classes, |_, _, _| None),
            Err(ClassError::InheritanceCycle)
        ));

        let base = class("base", ClassParent::Native(StateType::Tuple), &[]);
        let middle = class("middle", ClassParent::Class("base".into()), &[]);
        let top = class("top", ClassParent::Class("middle".into()), &[]);
        let depth_instance = instance(top, &[]);
        let classes = HashMap::from([("base".into(), base), ("middle".into(), middle)]);
        assert!(matches!(
            depth_instance.resolve_member_with_limit(&id("x"), &classes, |_, _, _| None, 2),
            Err(ClassError::InheritanceDepthExceeded { limit: 2 })
        ));

        let base = class("base", ClassParent::Native(StateType::Tuple), &[]);
        let instance = instance(base, &[]);
        assert!(matches!(
            instance.resolve_member(&id("x"), &HashMap::new(), |_, _, _| None),
            Err(ClassError::MissingMember { member }) if member == id("x")
        ));
    }

    #[test]
    fn identity_hash_is_stable_and_lookup_does_not_project_parent() {
        let first = class(
            "stable",
            ClassParent::Native(StateType::Tuple),
            &[("x", scalar(1))],
        );
        let second = first.clone();
        let mut left = DefaultHasher::new();
        let mut right = DefaultHasher::new();
        first.hash(&mut left);
        second.hash(&mut right);
        assert_eq!(left.finish(), right.finish());
        assert_eq!(first, second);

        let instance = instance(first, &[]);
        let native_calls = AtomicUsize::new(0);
        instance
            .resolve_member(&id("x"), &HashMap::new(), |_, _, _| {
                native_calls.fetch_add(1, Ordering::SeqCst);
                None
            })
            .expect("prototype member");
        assert_eq!(native_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn validation_accepts_same_kind_override_and_rejects_kind_change() {
        let base = class(
            "base",
            ClassParent::Native(StateType::Tuple),
            &[("x", scalar(1))],
        );
        let classes = HashMap::from([("base".into(), base)]);
        class(
            "valid",
            ClassParent::Class("base".into()),
            &[("x", scalar(2))],
        )
        .validate(&classes)
        .expect("same-kind override");

        let invalid = class(
            "invalid",
            ClassParent::Class("base".into()),
            &[("x", Scalar::Op(tc_ir::OpDef::Post(Vec::new())))],
        );
        assert!(matches!(
            invalid.validate(&classes),
            Err(ClassError::UnsupportedOverride { member }) if member == id("x")
        ));
    }
}
