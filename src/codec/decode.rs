use std::collections::btree_map::Entry;

use destream::de::{self, FromStream};
use pathlink::PathBuf;
use tc_collection::StorageContext;
use tc_ir::{Id, Map};
use tc_value::Value;
use tc_value::class::NativeClass;

use crate::codec::class::{ObjectType, StateType};
use crate::{ClassDef, ClassInstance, Object, State};

fn descend<E: de::Error>(remaining: usize) -> Result<usize, E> {
    remaining.checked_sub(1).ok_or_else(|| {
        E::custom(format!(
            "State structural nesting exceeds {}",
            tc_ir::MAX_SCALAR_DEPTH
        ))
    })
}

impl<Txn: StorageContext> de::FromStream for State<Txn> {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(context: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        decode_state(context, decoder, tc_ir::MAX_SCALAR_DEPTH).await
    }
}

// Boxing bounds future sizes; the shared allowance bounds recursive polling.
// Child codecs receive the remaining allowance rather than starting a new limit.
/// Decode State using its caller's remaining structural allowance and delegated
/// collection allocation context.
pub async fn decode_state<Txn: StorageContext, D: de::Decoder>(
    context: Txn,
    decoder: &mut D,
    remaining: usize,
) -> Result<State<Txn>, D::Error> {
    let remaining = remaining.min(tc_ir::MAX_SCALAR_DEPTH);
    match decoder.peek_kind().await? {
        de::Kind::Leaf => {
            let value = Box::pin(Value::from_stream((), decoder)).await?;
            Ok(if matches!(value, Value::None) {
                State::None
            } else {
                State::from(value)
            })
        }
        de::Kind::Seq => tuple(context, decoder, remaining).await,
        de::Kind::Map => {
            let mut cursor = decoder.open_container(de::Kind::Map, None).await?;
            if cursor.slot() == de::Slot::End {
                descend::<D::Error>(remaining)?;
                return Ok(State::Map(Map::new()));
            }

            let key = String::from_stream((), decoder).await?;
            decoder.finish_child(&mut cursor).await?;
            if key.parse::<Id>().is_err() {
                Box::pin(tagged(context, decoder, cursor, key, remaining)).await
            } else {
                Box::pin(map(context, decoder, cursor, key, remaining))
                    .await
                    .map(State::Map)
            }
        }
    }
}

async fn tuple<Txn: StorageContext, D: de::Decoder>(
    context: Txn,
    decoder: &mut D,
    remaining: usize,
) -> Result<State<Txn>, D::Error> {
    let remaining = descend::<D::Error>(remaining)?;
    let mut cursor = decoder.open_container(de::Kind::Seq, None).await?;
    let mut values = Vec::new();

    while cursor.slot() != de::Slot::End {
        let child = context.subcontext(values.len().to_string());
        let value = Box::pin(decode_state(child, decoder, remaining)).await?;
        decoder.finish_child(&mut cursor).await?;
        values.try_reserve(1).map_err(de::Error::custom)?;
        values.push(value);
    }

    Ok(State::Tuple(values))
}

async fn map<Txn: StorageContext, D: de::Decoder>(
    context: Txn,
    decoder: &mut D,
    mut cursor: de::Container,
    mut key: String,
    remaining: usize,
) -> Result<Map<State<Txn>>, D::Error> {
    let remaining = descend::<D::Error>(remaining)?;
    let mut values = Map::new();

    loop {
        let id = key
            .parse::<Id>()
            .map_err(|err| tc_error::bad_request!("invalid state map key ID: {err}"))
            .map_err(de::Error::custom)?;

        let entry = match values.entry(id) {
            Entry::Vacant(entry) => entry,
            Entry::Occupied(_) => return Err(de::Error::custom("duplicate state map member")),
        };

        let child = context.subcontext(entry.key().to_string());
        let value = Box::pin(decode_state(child, decoder, remaining)).await?;
        decoder.finish_child(&mut cursor).await?;
        entry.insert(value);
        if cursor.slot() == de::Slot::End {
            return Ok(values);
        }

        key = String::from_stream((), decoder).await?;
        decoder.finish_child(&mut cursor).await?;
    }
}

async fn tagged<Txn: StorageContext, D: de::Decoder>(
    context: Txn,
    decoder: &mut D,
    mut cursor: de::Container,
    key: String,
    remaining: usize,
) -> Result<State<Txn>, D::Error> {
    let class = key
        .parse::<PathBuf>()
        .ok()
        .as_deref()
        .and_then(StateType::from_path);

    let Some(class) = class else {
        return Box::pin(tc_ir::decode_scalar_entry(decoder, cursor, key, remaining))
            .await
            .map(State::Scalar);
    };

    let state = match class {
        StateType::Tuple => {
            if decoder.peek_kind().await? != de::Kind::Seq {
                return Err(de::Error::custom("expected a TinyChain state tuple"));
            }

            Box::pin(tuple(context, decoder, remaining)).await?
        }
        StateType::Collection(class) => State::Collection(
            Box::pin(tc_collection::decode_collection(class, context, decoder)).await?,
        ),
        StateType::Scalar(_) => {
            return Box::pin(tc_ir::decode_scalar_entry(decoder, cursor, key, remaining))
                .await
                .map(State::Scalar);
        }
        StateType::Object(ObjectType::Class) => {
            let remaining = descend::<D::Error>(remaining)?;
            let class = Box::pin(ClassDef::decode_with_depth(decoder, remaining)).await?;
            State::from(Object::Class(class))
        }
        StateType::Object(ObjectType::Instance) => {
            Box::pin(instance(context, decoder, descend::<D::Error>(remaining)?)).await?
        }
    };

    Box::pin(finish_tag(decoder, &mut cursor)).await?;
    Ok(state)
}

// Keep the delimiter future out of every enclosing instance's polling frame.
// Both this helper and its child decoder are boxed at recursive boundaries.
async fn state_field<Txn: StorageContext, D: de::Decoder>(
    context: Txn,
    decoder: &mut D,
    sequence: &mut de::Container,
    remaining: usize,
) -> Result<State<Txn>, D::Error> {
    require_field::<D::Error>(sequence, 0)?;
    let value = Box::pin(decode_state(context, decoder, remaining)).await?;
    decoder.finish_child(sequence).await?;
    Ok(value)
}

fn require_field<E: de::Error>(sequence: &de::Container, index: usize) -> Result<(), E> {
    if sequence.slot() == de::Slot::End {
        Err(E::invalid_length(index, 3))
    } else {
        Ok(())
    }
}

async fn class_field<D: de::Decoder>(
    decoder: &mut D,
    sequence: &mut de::Container,
    remaining: usize,
) -> Result<ClassDef, D::Error> {
    require_field::<D::Error>(sequence, 1)?;
    let class = Box::pin(ClassDef::decode_with_depth(decoder, remaining)).await?;
    decoder.finish_child(sequence).await?;
    Ok(class)
}

async fn members_field<Txn: StorageContext, D: de::Decoder>(
    context: Txn,
    decoder: &mut D,
    sequence: &mut de::Container,
    remaining: usize,
) -> Result<Map<State<Txn>>, D::Error> {
    require_field::<D::Error>(sequence, 2)?;
    if decoder.peek_kind().await? != de::Kind::Map {
        return Err(de::Error::custom("expected a TinyChain state map"));
    }

    let mut cursor = decoder.open_container(de::Kind::Map, None).await?;
    let members = if cursor.slot() == de::Slot::End {
        descend::<D::Error>(remaining)?;
        Map::new()
    } else {
        let key = String::from_stream((), decoder).await?;
        decoder.finish_child(&mut cursor).await?;
        Box::pin(map(context, decoder, cursor, key, remaining)).await?
    };

    decoder.finish_child(sequence).await?;
    if sequence.slot() != de::Slot::End {
        Box::pin(de::IgnoredAny::from_stream((), decoder)).await?;
        decoder.finish_child(sequence).await?;
        return Err(de::Error::invalid_length(4, 3));
    }

    Ok(members)
}

async fn instance<Txn: StorageContext, D: de::Decoder>(
    context: Txn,
    decoder: &mut D,
    remaining: usize,
) -> Result<State<Txn>, D::Error> {
    let mut cursor = decoder.open_container(de::Kind::Seq, Some(3)).await?;
    let parent = Box::pin(state_field(
        context.subcontext("parent"),
        decoder,
        &mut cursor,
        remaining,
    ))
    .await?;

    let class = Box::pin(class_field(decoder, &mut cursor, remaining)).await?;
    let members = Box::pin(members_field(
        context.subcontext("members"),
        decoder,
        &mut cursor,
        remaining,
    ))
    .await?;

    Ok(State::from(Object::Instance(ClassInstance::new(
        parent, class, members,
    ))))
}

async fn finish_tag<D: de::Decoder>(
    decoder: &mut D,
    map: &mut de::Container,
) -> Result<(), D::Error> {
    decoder.finish_child(map).await?;
    if map.slot() != de::Slot::End {
        return Err(de::Error::custom("unexpected extra State type map entry"));
    }

    Ok(())
}
