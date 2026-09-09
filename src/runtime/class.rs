//! User-defined Class and instance semantics.
//!
use std::collections::HashSet;
use std::fmt;

use async_hash::{Digest as AsyncDigest, Hash as AsyncHash, Output};
use destream::{de, en, EncodeMap, IntoStream};
use pathlink::{Link, PathBuf};
use tc_ir::{Id, Map, Scalar};
use tc_value::class::NativeClass;

use super::{State, StateType};

/// A user-defined Class definition or concrete instance carried by [`State`].
#[derive(Clone, Debug)]
pub enum Object<Txn: tc_collection::StorageContext> {
    Class(ClassDef),
    Instance(ClassInstance<Txn>),
}

impl<Txn: tc_collection::StorageContext> From<ClassDef> for Object<Txn> {
    fn from(class: ClassDef) -> Self {
        Self::Class(class)
    }
}

impl<Txn: tc_collection::StorageContext> From<ClassInstance<Txn>> for Object<Txn> {
    fn from(instance: ClassInstance<Txn>) -> Self {
        Self::Instance(instance)
    }
}

/// The default maximum number of user-defined classes visited during lookup.
pub const MAX_INHERITANCE_DEPTH: usize = 64;

/// A Class parent, distinguishing native extension from user-defined extension.
#[derive(Clone, Debug, PartialEq)]
pub enum ClassParent {
    Native(StateType),
    Class(Link),
}

/// An immutable user-defined Class definition.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassDef {
    identity: Link,
    digest: [u8; 32],
    parent: ClassParent,
    prototype: Map<Scalar>,
}

/// The canonical immutable Class body hashed by [`ClassDef::digest`].
///
/// The digest field is deliberately absent, so its encoding has no circular
/// dependency. A concrete storage or wire boundary chooses the deterministic
/// codec and verifies the resulting bytes before accepting the definition.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassBody {
    parent: ClassParent,
    prototype: Map<Scalar>,
}

impl ClassBody {
    pub fn new(parent: ClassParent, prototype: Map<Scalar>) -> Self {
        Self { parent, prototype }
    }

    pub fn parent(&self) -> &ClassParent {
        &self.parent
    }

    pub fn prototype(&self) -> &Map<Scalar> {
        &self.prototype
    }

    /// Return the bare parent/prototype map accepted by the Class application root.
    pub fn definition(&self) -> Scalar {
        let parent = match &self.parent {
            ClassParent::Native(parent) => tc_value::Value::Link(
                parent
                    .path()
                    .to_string()
                    .parse()
                    .expect("native Class path"),
            ),
            ClassParent::Class(parent) => tc_value::Value::Link(parent.clone()),
        };
        Scalar::Map(Map::from_iter([
            ("parent".parse().expect("Class field"), Scalar::from(parent)),
            (
                "prototype".parse().expect("Class field"),
                Scalar::Map(self.prototype.clone()),
            ),
        ]))
    }

    /// Return the format-neutral digest of this immutable Class body.
    pub fn digest(&self) -> [u8; 32] {
        AsyncHash::<async_hash::Sha256>::hash(self).into()
    }
}

impl<D: AsyncDigest> AsyncHash<D> for &ClassBody {
    fn hash(self) -> Output<D> {
        let parent = match &self.parent {
            ClassParent::Native(parent) => parent.path().to_string(),
            ClassParent::Class(parent) => parent.to_string(),
        };
        AsyncHash::<D>::hash((parent, &self.prototype))
    }
}

impl TryFrom<Scalar> for ClassBody {
    type Error = ClassError;

    fn try_from(body: Scalar) -> Result<Self, Self::Error> {
        let Scalar::Map(mut body) = body else {
            return Err(ClassError::MalformedDefinition(
                "a Class body must be a map".to_string(),
            ));
        };

        let parent_key: Id = "parent".parse().expect("static Class field");
        let prototype_key: Id = "prototype".parse().expect("static Class field");
        let parent = body
            .remove(&parent_key)
            .ok_or_else(|| ClassError::MalformedDefinition("missing Class parent".to_string()))?;
        let prototype = body.remove(&prototype_key).ok_or_else(|| {
            ClassError::MalformedDefinition("missing Class prototype".to_string())
        })?;
        if !body.is_empty() {
            return Err(ClassError::MalformedDefinition(
                "unknown Class body field".to_string(),
            ));
        }

        let parent = match parent {
            Scalar::Value(tc_value::Value::Link(link)) => link.to_string(),
            Scalar::Value(tc_value::Value::String(parent)) => parent,
            _ => {
                return Err(ClassError::InvalidParent(
                    "a Class parent must be a canonical URI".to_string(),
                ));
            }
        };
        let parent = decode_parent(&parent).map_err(ClassError::InvalidParent)?;
        let Scalar::Map(prototype) = prototype else {
            return Err(ClassError::MalformedDefinition(
                "a Class prototype must be a map".to_string(),
            ));
        };

        Ok(Self::new(parent, prototype))
    }
}

impl ClassDef {
    pub fn new(
        identity: Link,
        digest: [u8; 32],
        parent: ClassParent,
        prototype: Map<Scalar>,
    ) -> Self {
        Self {
            identity,
            digest,
            parent,
            prototype,
        }
    }

    pub fn identity(&self) -> &Link {
        &self.identity
    }

    pub fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    pub fn parent(&self) -> &ClassParent {
        &self.parent
    }

    pub fn prototype(&self) -> &Map<Scalar> {
        &self.prototype
    }
}

impl ClassDef {
    pub fn from_body(identity: Link, body: ClassBody) -> Self {
        let digest = body.digest();
        Self {
            identity,
            digest,
            parent: body.parent,
            prototype: body.prototype,
        }
    }

    pub fn body(&self) -> ClassBody {
        ClassBody {
            parent: self.parent.clone(),
            prototype: self.prototype.clone(),
        }
    }

    pub fn validate_digest(&self) -> Result<(), ClassError> {
        (self.digest == self.body().digest())
            .then_some(())
            .ok_or(ClassError::DefinitionDigestMismatch)
    }

    fn extend_referenced_methods(
        &self,
        requirements: &mut std::collections::BTreeMap<
            Link,
            std::collections::BTreeSet<tc_ir::Method>,
        >,
    ) {
        for scalar in self.prototype.values() {
            scalar.visit_referenced_methods(&mut |link, method| {
                requirements.entry(link.clone()).or_default().insert(method);
            });
        }
    }

    pub fn effective_referenced_methods(
        &self,
        classes: &std::collections::BTreeMap<Link, Self>,
    ) -> Result<
        std::collections::BTreeMap<Link, std::collections::BTreeSet<tc_ir::Method>>,
        ClassError,
    > {
        analyze_classes(classes, [self.identity.clone()])?
            .remove(&self.identity)
            .ok_or_else(|| ClassError::InvalidParent(self.identity.to_string()))
    }
}

impl<D: AsyncDigest> AsyncHash<D> for &ClassDef {
    fn hash(self) -> Output<D> {
        let body = self.body();
        AsyncHash::<D>::hash((&self.identity, &body))
    }
}

/// Validate a Class batch and derive every effective application requirement in
/// one memoized inheritance traversal.
pub fn analyze_classes(
    classes: &std::collections::BTreeMap<Link, ClassDef>,
    roots: impl IntoIterator<Item = Link>,
) -> Result<
    std::collections::BTreeMap<
        Link,
        std::collections::BTreeMap<Link, std::collections::BTreeSet<tc_ir::Method>>,
    >,
    ClassError,
> {
    type Requirements = std::collections::BTreeMap<Link, std::collections::BTreeSet<tc_ir::Method>>;
    type Members = std::collections::BTreeMap<Id, bool>;

    fn analyze(
        identity: &Link,
        classes: &std::collections::BTreeMap<Link, ClassDef>,
        visiting: &mut std::collections::BTreeSet<Link>,
        memo: &mut std::collections::BTreeMap<Link, (Requirements, Members)>,
        depth: usize,
    ) -> Result<(Requirements, Members), ClassError> {
        if let Some(analysis) = memo.get(identity) {
            return Ok(analysis.clone());
        }
        if depth >= MAX_INHERITANCE_DEPTH {
            return Err(ClassError::InheritanceDepthExceeded {
                limit: MAX_INHERITANCE_DEPTH,
            });
        }
        if !visiting.insert(identity.clone()) {
            return Err(ClassError::InheritanceCycle);
        }
        let class = classes
            .get(identity)
            .ok_or_else(|| ClassError::InvalidParent(identity.to_string()))?;
        class.validate_digest()?;
        let (mut requirements, mut members) = match class.parent() {
            ClassParent::Class(parent) => analyze(parent, classes, visiting, memo, depth + 1)?,
            ClassParent::Native(_) => Default::default(),
        };
        for (member, value) in class.prototype() {
            let method = matches!(value, Scalar::Op(_));
            if members.get(member).is_some_and(|parent| *parent != method) {
                return Err(ClassError::UnsupportedOverride {
                    member: member.clone(),
                });
            }
            members.insert(member.clone(), method);
        }
        class.extend_referenced_methods(&mut requirements);
        visiting.remove(identity);
        memo.insert(identity.clone(), (requirements.clone(), members.clone()));
        Ok((requirements, members))
    }

    let mut memo = std::collections::BTreeMap::new();
    let mut output = std::collections::BTreeMap::new();
    for identity in roots {
        let (requirements, _) = analyze(
            &identity,
            classes,
            &mut std::collections::BTreeSet::new(),
            &mut memo,
            0,
        )?;
        output.insert(identity, requirements);
    }
    Ok(output)
}

impl de::FromStream for ClassDef {
    type Context = ();

    async fn from_stream<D: de::Decoder>(
        _context: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct ClassVisitor;

        impl de::Visitor for ClassVisitor {
            type Value = ClassDef;

            fn expecting() -> &'static str {
                "a canonical Class definition"
            }

            async fn visit_map<A: de::MapAccess>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut identity = None;
                let mut digest: Option<String> = None;
                let mut parent = None;
                let mut prototype = None;

                while let Some(key) = map.next_key::<String>(()).await? {
                    match key.as_str() {
                        "identity" => identity = Some(map.next_value(()).await?),
                        "digest" => digest = Some(map.next_value(()).await?),
                        "parent" => parent = Some(map.next_value::<String>(()).await?),
                        "prototype" => prototype = Some(map.next_value(()).await?),
                        _ => {
                            return Err(de::Error::custom(format!(
                                "unknown Class definition field {key}"
                            )));
                        }
                    }
                }

                let identity: String =
                    identity.ok_or_else(|| de::Error::custom("missing identity"))?;
                let identity = identity.parse::<Link>().map_err(de::Error::custom)?;

                let parent =
                    decode_parent(&parent.ok_or_else(|| de::Error::custom("missing parent"))?)
                        .map_err(de::Error::custom)?;

                let class = ClassDef {
                    identity,
                    digest: decode_digest(
                        &digest.ok_or_else(|| de::Error::custom("missing digest"))?,
                    )
                    .map_err(de::Error::custom)?,
                    parent,
                    prototype: prototype.ok_or_else(|| de::Error::custom("missing prototype"))?,
                };
                class.validate_digest().map_err(de::Error::custom)?;
                Ok(class)
            }
        }

        decoder.decode_map(ClassVisitor).await
    }
}

impl<'en> en::IntoStream<'en> for ClassDef {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let parent = encode_parent(&self.parent);
        let mut map = encoder.encode_map(Some(4))?;
        map.encode_entry("identity", self.identity.to_string())?;
        map.encode_entry("digest", hex::encode(self.digest))?;
        map.encode_entry("parent", parent)?;
        map.encode_entry("prototype", self.prototype)?;
        map.end()
    }
}

impl de::FromStream for ClassBody {
    type Context = ();

    async fn from_stream<D: de::Decoder>(
        _context: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct BodyVisitor;

        impl de::Visitor for BodyVisitor {
            type Value = ClassBody;

            fn expecting() -> &'static str {
                "a literal Class body"
            }

            async fn visit_map<A: de::MapAccess>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut parent = None;
                let mut prototype = None;
                while let Some(key) = map.next_key::<String>(()).await? {
                    match key.as_str() {
                        "parent" => parent = Some(map.next_value::<String>(()).await?),
                        "prototype" => prototype = Some(map.next_value(()).await?),
                        _ => {
                            return Err(de::Error::custom(format!(
                                "unknown Class body field {key}"
                            )));
                        }
                    }
                }
                let parent = decode_parent(
                    &parent.ok_or_else(|| de::Error::custom("missing Class parent"))?,
                )
                .map_err(de::Error::custom)?;
                Ok(ClassBody::new(
                    parent,
                    prototype.ok_or_else(|| de::Error::custom("missing Class prototype"))?,
                ))
            }
        }

        decoder.decode_map(BodyVisitor).await
    }
}

impl<'en> en::IntoStream<'en> for ClassBody {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let parent = encode_parent(&self.parent);
        let mut map = encoder.encode_map(Some(2))?;
        map.encode_entry("parent", parent)?;
        map.encode_entry("prototype", self.prototype)?;
        map.end()
    }
}

impl<'en> en::ToStream<'en> for ClassBody {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        self.clone().into_stream(encoder)
    }
}

impl<'en> en::ToStream<'en> for ClassDef {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        self.clone().into_stream(encoder)
    }
}

fn encode_parent(parent: &ClassParent) -> String {
    match parent {
        ClassParent::Native(parent) => parent.path().to_string(),
        ClassParent::Class(parent) => parent.to_string(),
    }
}

fn decode_parent(value: &str) -> Result<ClassParent, String> {
    let path = value
        .parse::<PathBuf>()
        .map_err(|err| format!("invalid native Class parent: {err}"))?;
    if let Some(parent) = StateType::from_path(path.as_ref()) {
        return Ok(ClassParent::Native(parent));
    }
    value
        .parse::<Link>()
        .map(ClassParent::Class)
        .map_err(|err| format!("invalid Class parent: {err}"))
}

fn decode_digest(value: &str) -> Result<[u8; 32], String> {
    let bytes = hex::decode(value).map_err(|err| format!("invalid Class digest: {err}"))?;
    bytes
        .try_into()
        .map_err(|_| "a Class digest must contain exactly 32 bytes".to_string())
}

impl From<ClassBody> for Scalar {
    fn from(body: ClassBody) -> Self {
        let parent = match body.parent {
            ClassParent::Native(parent) => parent
                .path()
                .to_string()
                .parse::<Link>()
                .expect("native Class paths are valid links"),
            ClassParent::Class(parent) => parent,
        };
        if body.prototype.is_empty() {
            Scalar::from(tc_value::Value::Link(parent))
        } else {
            Scalar::from(tc_ir::TCRef::Op(tc_ir::OpRef::Post((
                tc_ir::Subject::Link(parent),
                body.prototype,
            ))))
        }
    }
}

impl From<ClassDef> for Scalar {
    fn from(class: ClassDef) -> Self {
        class.body().into()
    }
}

/// A concrete instance of a user-defined Class.
#[derive(Clone, Debug)]
pub struct ClassInstance<Txn: tc_collection::StorageContext> {
    parent: Box<State<Txn>>,
    class: ClassDef,
    members: Map<State<Txn>>,
}

impl<Txn: tc_collection::StorageContext> ClassInstance<Txn> {
    pub fn new(parent: State<Txn>, class: ClassDef, members: Map<State<Txn>>) -> Self {
        Self {
            parent: Box::new(parent),
            class,
            members,
        }
    }

    pub fn parent(&self) -> &State<Txn> {
        &self.parent
    }

    pub fn class(&self) -> &ClassDef {
        &self.class
    }

    pub fn members(&self) -> &Map<State<Txn>> {
        &self.members
    }

    pub fn into_parts(self) -> (State<Txn>, ClassDef, Map<State<Txn>>) {
        (*self.parent, self.class, self.members)
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
pub enum ResolvedMember<'a, Txn: tc_collection::StorageContext> {
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
        instance: &'a ClassInstance<Txn>,
    },
}

impl<Txn: tc_collection::StorageContext> ResolvedMember<'_, Txn> {
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
    DefinitionDigestMismatch,
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
            Self::DefinitionDigestMismatch => f.write_str("Class definition digest mismatch"),
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

impl<Txn: tc_collection::StorageContext> ClassInstance<Txn> {
    /// Resolve `member` without evaluating references or projecting `parent`.
    ///
    /// Lookup order is instance member, concrete prototype, nearest inherited
    /// prototype, and finally native behavior. The native callback is invoked
    /// only after structural lookup is exhausted and receives no transaction.
    pub fn resolve_member<'a, N>(
        &'a self,
        member: &Id,
        classes: &'a std::collections::BTreeMap<Link, ClassDef>,
        native: N,
    ) -> Result<ResolvedMember<'a, Txn>, ClassError>
    where
        N: FnOnce(Option<&StateType>, &'a State<Txn>, &Id) -> Option<&'a State<Txn>>,
    {
        self.resolve_member_with_limit(member, classes, native, MAX_INHERITANCE_DEPTH)
    }

    pub fn resolve_member_with_limit<'a, N>(
        &'a self,
        member: &Id,
        classes: &'a std::collections::BTreeMap<Link, ClassDef>,
        native: N,
        limit: usize,
    ) -> Result<ResolvedMember<'a, Txn>, ClassError>
    where
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
                        .get(parent)
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

impl ClassDef {
    /// Validate the inheritance chain and reject method/value kind changes.
    pub fn validate(
        &self,
        classes: &std::collections::BTreeMap<Link, ClassDef>,
    ) -> Result<(), ClassError> {
        self.validate_with_limit(classes, MAX_INHERITANCE_DEPTH)
    }

    pub fn validate_with_limit(
        &self,
        classes: &std::collections::BTreeMap<Link, ClassDef>,
        limit: usize,
    ) -> Result<(), ClassError> {
        let mut visited = HashSet::new();
        let mut class = self;
        let mut inherited = std::collections::BTreeMap::<Id, bool>::new();

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
                        .get(parent)
                        .ok_or_else(|| ClassError::InvalidParent("unknown identity".into()))?;
                }
            }
        }

        unreachable!("bounded validation loop always returns")
    }
}

fn bind<'a, Txn: tc_collection::StorageContext>(
    instance: &'a ClassInstance<Txn>,
    value: &'a Scalar,
) -> ResolvedMember<'a, Txn> {
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
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use bytes::Bytes;
    use futures::{stream, TryStreamExt};
    use tc_value::Value;

    use super::*;

    type TestState = State<crate::runtime::tests::TestTxn>;

    fn id(value: &str) -> Id {
        value.parse().expect("Id")
    }

    fn scalar(value: u64) -> Scalar {
        Scalar::from(Value::from(value))
    }

    fn class_link(name: &str) -> Link {
        format!("/class/example-devco/{name}/1.0.0")
            .parse()
            .expect("Class identity")
    }

    fn class(name: &str, parent: ClassParent, members: &[(&str, Scalar)]) -> ClassDef {
        let prototype = members
            .iter()
            .cloned()
            .map(|(name, value)| (id(name), value))
            .collect();
        ClassDef::from_body(class_link(name), ClassBody::new(parent, prototype))
    }

    fn instance(
        class: ClassDef,
        members: &[(&str, TestState)],
    ) -> ClassInstance<crate::runtime::tests::TestTxn> {
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
            ClassParent::Class(class_link("base")),
            &[("prototype", scalar(4)), ("overridden", scalar(5))],
        );
        let instance = instance(
            derived,
            &[
                ("instance", TestState::from(6_u64)),
                ("overridden", TestState::from(7_u64)),
            ],
        );
        let classes = BTreeMap::from([(class_link("base"), base)]);

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
        let classes = BTreeMap::new();
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
        let derived = class("derived", ClassParent::Class(class_link("missing")), &[]);
        let invalid_instance = instance(derived, &[]);
        assert!(matches!(
            invalid_instance.resolve_member(&id("x"), &BTreeMap::new(), |_, _, _| None),
            Err(ClassError::InvalidParent(_))
        ));

        let a = class("a", ClassParent::Class(class_link("b")), &[]);
        let b = class("b", ClassParent::Class(class_link("a")), &[]);
        let cycle_instance = instance(a.clone(), &[]);
        let classes = BTreeMap::from([(class_link("a"), a), (class_link("b"), b)]);
        assert!(matches!(
            cycle_instance.resolve_member(&id("x"), &classes, |_, _, _| None),
            Err(ClassError::InheritanceCycle)
        ));

        let base = class("base", ClassParent::Native(StateType::Tuple), &[]);
        let middle = class("middle", ClassParent::Class(class_link("base")), &[]);
        let top = class("top", ClassParent::Class(class_link("middle")), &[]);
        let depth_instance = instance(top, &[]);
        let classes = BTreeMap::from([(class_link("base"), base), (class_link("middle"), middle)]);
        assert!(matches!(
            depth_instance.resolve_member_with_limit(&id("x"), &classes, |_, _, _| None, 2),
            Err(ClassError::InheritanceDepthExceeded { limit: 2 })
        ));

        let base = class("base", ClassParent::Native(StateType::Tuple), &[]);
        let instance = instance(base, &[]);
        assert!(matches!(
            instance.resolve_member(&id("x"), &BTreeMap::new(), |_, _, _| None),
            Err(ClassError::MissingMember { member }) if member == id("x")
        ));
    }

    #[test]
    fn lookup_does_not_project_parent() {
        let first = class(
            "stable",
            ClassParent::Native(StateType::Tuple),
            &[("x", scalar(1))],
        );
        let instance = instance(first, &[]);
        let native_calls = AtomicUsize::new(0);
        instance
            .resolve_member(&id("x"), &BTreeMap::new(), |_, _, _| {
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
        let classes = BTreeMap::from([(class_link("base"), base)]);
        class(
            "valid",
            ClassParent::Class(class_link("base")),
            &[("x", scalar(2))],
        )
        .validate(&classes)
        .expect("same-kind override");

        let invalid = class(
            "invalid",
            ClassParent::Class(class_link("base")),
            &[("x", Scalar::Op(tc_ir::OpDef::Post(Vec::new())))],
        );
        assert!(matches!(
            invalid.validate(&classes),
            Err(ClassError::UnsupportedOverride { member }) if member == id("x")
        ));
    }

    #[test]
    fn effective_requirements_include_inherited_prototypes() {
        let dependency: Link = "/lib/example-devco/math/1.0.0".parse().expect("dependency");
        let base_id: Link = "/class/example-devco/base/1.0.0"
            .parse()
            .expect("base identity");
        let derived_id: Link = "/class/example-devco/derived/1.0.0"
            .parse()
            .expect("derived identity");
        let reference = Scalar::from(tc_ir::TCRef::Op(tc_ir::OpRef::Get((
            tc_ir::Subject::Link(dependency.clone()),
            Scalar::default(),
        ))));
        let base = ClassDef::from_body(
            base_id.clone(),
            ClassBody::new(
                ClassParent::Native(StateType::Tuple),
                [(id("value"), reference)].into_iter().collect(),
            ),
        );
        let derived = ClassDef::from_body(
            derived_id.clone(),
            ClassBody::new(ClassParent::Class(base_id.clone()), Map::new()),
        );
        let classes = [(base_id, base), (derived_id, derived.clone())]
            .into_iter()
            .collect();
        let requirements = derived
            .effective_referenced_methods(&classes)
            .expect("effective requirements");
        assert!(requirements[&dependency].contains(&tc_ir::Method::Get));
    }

    #[tokio::test]
    async fn validated_class_codec_is_symmetric_and_digest_checked() {
        let identity: Link = "/class/example-devco/vector/1.0.0"
            .parse()
            .expect("identity");
        let parent = ClassParent::Native(StateType::Tuple);
        let prototype: Map<Scalar> = [(id("dimensions"), scalar(3))].into_iter().collect();
        let body = ClassBody::new(parent, prototype);
        let definition = ClassDef::from_body(identity, body);
        let encoded = destream_json::encode(definition.clone()).expect("encode");
        let decoded: ClassDef = destream_json::try_decode((), encoded)
            .await
            .expect("decode");
        assert_eq!(decoded, definition);
        assert!(decoded.validate_digest().is_ok());

        let invalid = stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(
            br#"{"manifest_version":2,"identity":"/class/example-devco/vector/1.0.0","digest":["sha256","4c24b66b4b5a80be4e96597c5b70d653dc223fbe2d505b9c865f10c8f14c6e91"],"parent":"/state/tuple","prototype":{}}"#,
        ))]);
        assert!(destream_json::try_decode::<_, _, ClassDef>((), invalid)
            .await
            .is_err());

        let mismatched = stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(
            br#"{"identity":"/class/example-devco/vector/1.0.0","digest":"0000000000000000000000000000000000000000000000000000000000000000","parent":"/state/scalar/tuple","prototype":{"dimensions":3}}"#,
        ))]);
        assert!(destream_json::try_decode::<_, _, ClassDef>((), mismatched)
            .await
            .is_err());
    }

    #[test]
    fn class_hash_composes_its_link_and_bare_body() {
        let body = ClassBody::new(
            ClassParent::Native(StateType::Tuple),
            [(id("dimensions"), scalar(3))].into_iter().collect(),
        );
        let first = ClassDef::from_body(
            "/class/example-devco/vector/1.0.0".parse().unwrap(),
            body.clone(),
        );
        let same = ClassDef::from_body(first.identity().clone(), body.clone());
        let other = ClassDef::from_body("/class/example-devco/vector/2.0.0".parse().unwrap(), body);
        assert_eq!(
            AsyncHash::<async_hash::Sha256>::hash(&first),
            AsyncHash::<async_hash::Sha256>::hash(&same)
        );
        assert_ne!(
            AsyncHash::<async_hash::Sha256>::hash(&first),
            AsyncHash::<async_hash::Sha256>::hash(&other)
        );
    }

    #[test]
    fn class_scalar_uses_the_v1_parent_or_post_form() {
        let parent: Link = "/class/example-devco/base/1.0.0".parse().unwrap();
        let empty: Scalar = ClassBody::new(ClassParent::Class(parent.clone()), Map::new()).into();
        assert_eq!(empty, Scalar::from(tc_value::Value::Link(parent.clone())));

        let prototype: Map<Scalar> = [(id("dimensions"), scalar(3))].into_iter().collect();
        let populated: Scalar =
            ClassBody::new(ClassParent::Class(parent.clone()), prototype.clone()).into();
        assert_eq!(
            populated,
            Scalar::from(tc_ir::TCRef::Op(tc_ir::OpRef::Post((
                tc_ir::Subject::Link(parent),
                prototype,
            ))))
        );
    }

    #[tokio::test]
    async fn bare_class_body_fixture_is_symmetric_and_rejects_metadata() {
        let valid = stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(include_bytes!(
            "../../fixtures/class_definition.json"
        )))]);
        let definition: ClassBody = destream_json::try_decode((), valid)
            .await
            .expect("Class fixture");
        let encoded = destream_json::encode(definition).expect("encode");
        let bytes = encoded
            .try_collect::<Vec<_>>()
            .await
            .expect("collect")
            .concat();
        assert_eq!(
            String::from_utf8(bytes).expect("utf8"),
            include_str!("../../fixtures/class_definition.json").trim_end()
        );

        let invalid = stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(include_bytes!(
            "../../fixtures/class_definition_invalid.json"
        )))]);
        assert!(destream_json::try_decode::<_, _, ClassBody>((), invalid)
            .await
            .is_err());
    }
}
