//! User-defined Class and instance semantics.
//!
use std::collections::HashSet;
use std::fmt;

use async_hash::{Digest as AsyncDigest, Hash as AsyncHash, Output};
use destream::de::FromStream;
use destream::{de, en, EncodeMap};
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
            Scalar::Value(value) => match value {
                tc_value::Value::Link(link) => link.to_string(),
                tc_value::Value::String(parent) => parent,
                _ => {
                    return Err(ClassError::InvalidParent(
                        "a Class parent must be a canonical URI".to_string(),
                    ))
                }
            },
            _ => {
                return Err(ClassError::InvalidParent(
                    "a Class parent must be a canonical URI".to_string(),
                ))
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
}

impl<D: AsyncDigest> AsyncHash<D> for &ClassDef {
    fn hash(self) -> Output<D> {
        let body = self.body();
        AsyncHash::<D>::hash((&self.identity, &body))
    }
}

/// Validate a Class batch in one memoized inheritance traversal.
pub fn validate_classes(
    classes: &std::collections::BTreeMap<Link, ClassDef>,
    roots: impl IntoIterator<Item = Link>,
) -> Result<(), ClassError> {
    type Members = std::collections::BTreeMap<Id, bool>;

    let mut memo = std::collections::BTreeMap::<Link, Members>::new();
    for identity in roots {
        let mut identity = &identity;
        let mut visiting = std::collections::BTreeSet::new();
        let mut frames = Vec::new();
        let mut members = loop {
            if let Some(members) = memo.get(identity) {
                break members.clone();
            }

            if frames.len() >= MAX_INHERITANCE_DEPTH {
                return Err(ClassError::InheritanceDepthExceeded {
                    limit: MAX_INHERITANCE_DEPTH,
                });
            }

            if !visiting.insert(identity) {
                return Err(ClassError::InheritanceCycle);
            }

            let class = classes
                .get(identity)
                .ok_or_else(|| ClassError::InvalidParent(identity.to_string()))?;

            class.validate_digest()?;
            frames.push((identity, class));
            match class.parent() {
                ClassParent::Class(parent) => identity = parent,
                ClassParent::Native(_) => break Members::new(),
            }
        };

        while let Some((identity, class)) = frames.pop() {
            for (member, value) in class.prototype() {
                let method = matches!(&value, Scalar::Op(_));
                if members.get(member).is_some_and(|parent| *parent != method) {
                    return Err(ClassError::UnsupportedOverride {
                        member: member.clone(),
                    });
                }

                members.insert(member.clone(), method);
            }

            memo.insert(identity.clone(), members.clone());
        }
    }

    Ok(())
}

impl de::FromStream for ClassDef {
    type Context = ();

    async fn from_stream<D: de::Decoder>((): (), decoder: &mut D) -> Result<Self, D::Error> {
        Self::decode_with_depth(decoder, tc_ir::MAX_SCALAR_DEPTH).await
    }
}

impl ClassDef {
    pub(crate) fn within_depth(&self, remaining: usize) -> bool {
        remaining
            .checked_sub(1)
            .is_some_and(|n| self.prototype.values().all(|value| value.within_depth(n)))
    }

    pub(crate) async fn decode_with_depth<D: de::Decoder>(
        decoder: &mut D,
        remaining: usize,
    ) -> Result<Self, D::Error> {
        let mut map = decoder.open_container(de::Kind::Map, None).await?;
        let mut identity = None;
        let mut digest = None;
        let mut parent = None;
        let mut prototype = None;
        while map.slot() != de::Slot::End {
            let key = String::from_stream((), decoder).await?;
            decoder.finish_child(&mut map).await?;
            match key.as_str() {
                "identity" => identity = Some(String::from_stream((), decoder).await?),
                "digest" => digest = Some(String::from_stream((), decoder).await?),
                "parent" => parent = Some(String::from_stream((), decoder).await?),
                "prototype" => {
                    // Keep Class decoder construction small in recursive State decoding.
                    let Scalar::Map(values) =
                        Box::pin(tc_ir::decode_scalar(decoder, remaining)).await?
                    else {
                        return Err(de::Error::custom("expected Class prototype map"));
                    };

                    prototype = Some(values);
                }
                _ => {
                    return Err(de::Error::custom(format!(
                        "unknown Class definition field {key}"
                    )))
                }
            }

            decoder.finish_child(&mut map).await?;
        }

        let identity = identity
            .ok_or_else(|| de::Error::custom("missing identity"))?
            .parse::<Link>()
            .map_err(de::Error::custom)?;

        let parent = decode_parent(&parent.ok_or_else(|| de::Error::custom("missing parent"))?)
            .map_err(de::Error::custom)?;

        let class = Self {
            identity,
            digest: decode_digest(&digest.ok_or_else(|| de::Error::custom("missing digest"))?)
                .map_err(de::Error::custom)?,
            parent,
            prototype: prototype.ok_or_else(|| de::Error::custom("missing prototype"))?,
        };

        class.validate_digest().map_err(de::Error::custom)?;
        Ok(class)
    }
}

impl<'en> en::IntoStream<'en> for ClassDef {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let parent = encode_parent(&self.parent);
        let mut map = encoder.encode_map(Some(4))?;
        map.encode_entry("identity", self.identity)?;
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
        let mut map = encoder.encode_map(Some(2))?;
        map.encode_entry("parent", encode_parent(&self.parent))?;
        map.encode_entry("prototype", &self.prototype)?;
        map.end()
    }
}

impl<'en> en::ToStream<'en> for ClassDef {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        let mut map = encoder.encode_map(Some(4))?;
        map.encode_entry("identity", &self.identity)?;
        map.encode_entry("digest", hex::encode(self.digest))?;
        map.encode_entry("parent", encode_parent(&self.parent))?;
        map.encode_entry("prototype", &self.prototype)?;
        map.end()
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
        declared_by: &'a Link,
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
                return Ok(bind(self, &class.identity, value));
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
                let method = matches!(&value, Scalar::Op(_));
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
    declared_by: &'a Link,
    value: &'a Scalar,
) -> ResolvedMember<'a, Txn> {
    match &value {
        Scalar::Op(definition) => ResolvedMember::BoundMethod {
            declared_by,
            definition,
            instance,
        },
        _ => ResolvedMember::Scalar {
            source: MemberSource::Prototype,
            value,
        },
    }
}

#[cfg(test)]
#[path = "../tests/unit/class.rs"]
mod tests;
