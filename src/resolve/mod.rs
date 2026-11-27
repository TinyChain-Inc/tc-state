use std::collections::HashMap;
use std::future::Future;

use pathlink::Link;
use safecast::{CastInto, TryCastFrom};
use tc_error::{TCError, TCResult};
use tc_ir::{After, Cond, ForEach, Id, Map, OpDef, OpRef, Public, Scalar, Subject, TCRef, While};
use tc_value::Value;

use crate::State;
use super::OpExecutor;

enum Target<'a, Txn: OpExecutor> {
    Link(Link),
    Borrowed(&'a State<Txn>, pathlink::PathBuf),
}

impl<'a, Txn: OpExecutor> Target<'a, Txn> {
    fn new(
        subject: Subject,
        values: &'a HashMap<Id, State<Txn>>,
        self_link: Option<&'a State<Txn>>,
    ) -> TCResult<Self> {
        Ok(match subject {
            Subject::Link(link) => Self::Link(link),
            Subject::Ref(id, suffix) if id.as_str() == "self" => Self::Borrowed(
                self_link.ok_or_else(|| TCError::bad_request("OpDef has $self but no scope"))?,
                suffix,
            ),
            Subject::Ref(id, suffix) => Self::Borrowed(
                values
                    .get(id.as_str())
                    .ok_or_else(|| TCError::not_found(format!("unknown id ${}", id.as_str())))?,
                suffix,
            ),
        })
    }
}

enum Frame<'a, Txn: OpExecutor> {
    Tuple(std::vec::IntoIter<Scalar>, Vec<State<Txn>>),
    Map(
        std::collections::btree_map::IntoIter<Id, Scalar>,
        Id,
        Map<State<Txn>>,
    ),
    Get(Target<'a, Txn>),
    Delete(Target<'a, Txn>),
    PutKey(Target<'a, Txn>, Scalar, bool),
    PutValue(Target<'a, Txn>, State<Txn>),
    Post(Target<'a, Txn>),
    Cond(Scalar, Scalar),
    After(Scalar),
    WhileCond(Scalar, Scalar),
    WhileClosure(OpDef, Scalar),
    WhileState(OpDef, OpDef),
    WhileTest(OpDef, OpDef, State<Txn>),
    ForItems(Scalar, Id),
    ForOp(Vec<State<Txn>>, Id),
}

// The explicit Send contract breaks trait-inference cycles through OpExecutor.
#[expect(
    clippy::manual_async_fn,
    reason = "async fn hides the required Send contract"
)]
pub fn resolve_scalar<'a, Txn>(
    scalar: Scalar,
    values: &'a HashMap<Id, State<Txn>>,
    txn: &'a Txn,
    self_link: Option<&'a State<Txn>>,
) -> impl Future<Output = TCResult<State<Txn>>> + Send + 'a
where
    Txn: OpExecutor,
{
    async move {
        let mut frames = Vec::new();
        let mut current = scalar;
        loop {
            frames.try_reserve(2).map_err(|err| {
                TCError::internal(format!("cannot reserve resolution frames: {err}"))
            })?;

            let mut value = match current {
                Scalar::Value(value) => State::from(value),
                Scalar::Op(op) => State::Scalar(Scalar::Op(capture_op(op, values)?)),
                Scalar::Tuple(items) => {
                    let mut remaining = items.into_iter();
                    if let Some(next) = remaining.next() {
                        frames.push(Frame::Tuple(remaining, Vec::new()));
                        current = next;
                        continue;
                    }

                    State::Tuple(Vec::new())
                }
                Scalar::Map(items) => {
                    let mut remaining = items.into_iter();
                    if let Some((key, next)) = remaining.next() {
                        frames.push(Frame::Map(remaining, key, Map::new()));
                        current = next;
                        continue;
                    }

                    State::Map(Map::new())
                }
                Scalar::Ref(reference) => match *reference {
                    TCRef::Id(id) if id.as_str() == "self" => self_link
                        .cloned()
                        .ok_or_else(|| TCError::bad_request("OpDef has $self but no scope"))?,
                    TCRef::Id(id) => values.get(id.as_str()).cloned().ok_or_else(|| {
                        TCError::not_found(format!("unknown id ${}", id.as_str()))
                    })?,
                    TCRef::Op(op) => {
                        match op {
                            OpRef::Get((subject, key)) => {
                                frames.push(Frame::Get(Target::new(subject, values, self_link)?));
                                current = key;
                            }
                            OpRef::Delete((subject, key)) => {
                                frames
                                    .push(Frame::Delete(Target::new(subject, values, self_link)?));

                                current = key;
                            }
                            OpRef::Put((subject, key, value)) => {
                                let validate_key = matches!(&subject, Subject::Ref(id, _) if id.as_str() == "self");
                                frames.push(Frame::PutKey(
                                    Target::new(subject, values, self_link)?,
                                    value,
                                    validate_key,
                                ));

                                current = key;
                            }
                            OpRef::Post((subject, params)) => {
                                frames.push(Frame::Post(Target::new(subject, values, self_link)?));
                                current = Scalar::Map(params);
                            }
                        }

                        continue;
                    }
                    TCRef::Cond(cond) => {
                        let Cond {
                            cond,
                            then,
                            or_else,
                        } = *cond;

                        frames.push(Frame::Cond(then, or_else));
                        current = Scalar::from(cond);
                        continue;
                    }
                    TCRef::After(after) => {
                        let After { when, then } = *after;
                        frames.push(Frame::After(then));
                        current = when;
                        continue;
                    }
                    TCRef::While(while_ref) => {
                        let While {
                            cond,
                            closure,
                            state,
                        } = *while_ref;

                        frames.push(Frame::WhileCond(closure, state));
                        current = cond;
                        continue;
                    }
                    TCRef::ForEach(for_each) => {
                        let ForEach {
                            items,
                            op,
                            item_name,
                        } = *for_each;

                        frames.push(Frame::ForItems(op, item_name));
                        current = items;
                        continue;
                    }
                },
            };

            loop {
                let Some(frame) = frames.pop() else {
                    return Ok(value);
                };

                match frame {
                    Frame::Tuple(mut remaining, mut out) => {
                        out.push(value);
                        if let Some(next) = remaining.next() {
                            frames.push(Frame::Tuple(remaining, out));
                            current = next;
                            break;
                        }

                        value = State::Tuple(out);
                    }
                    Frame::Map(mut remaining, key, mut out) => {
                        out.insert(key, value);
                        if let Some((key, next)) = remaining.next() {
                            frames.push(Frame::Map(remaining, key, out));
                            current = next;
                            break;
                        }

                        value = State::Map(out);
                    }
                    Frame::Get(target) => {
                        let key = Scalar::try_cast_from(value, |_| {
                            TCError::bad_request("expected scalar GET key")
                        })?;

                        value = match target {
                            Target::Link(link) => get_link(txn, link, key).await?,
                            Target::Borrowed(state, suffix) => {
                                state.get(txn, suffix.as_ref(), key).await?
                            }
                        };
                    }
                    Frame::Delete(target) => {
                        let key = Scalar::try_cast_from(value, |_| {
                            TCError::bad_request("expected scalar DELETE key")
                        })?;

                        txn.check_write()?;
                        value = match target {
                            Target::Link(link) => delete_link(txn, link, key).await?,
                            Target::Borrowed(state, suffix) => {
                                state.delete(txn, suffix.as_ref(), key).await?;
                                State::default()
                            }
                        };
                    }
                    Frame::PutKey(target, next, validate_key) => {
                        // The self receiver historically validates its key before resolving the value.
                        if validate_key {
                            value = State::from(Scalar::try_cast_from(value, |_| {
                                TCError::bad_request("expected scalar PUT key")
                            })?);
                        }

                        frames.push(Frame::PutValue(target, value));
                        current = next;
                        break;
                    }
                    Frame::PutValue(target, key) => {
                        value = match target {
                            Target::Link(link) => put_link(txn, link, key, value).await?,
                            Target::Borrowed(state, suffix) => {
                                let key = Scalar::try_cast_from(key, |_| {
                                    TCError::bad_request("expected scalar PUT key")
                                })?;

                                txn.check_write()?;
                                state.put(txn, suffix.as_ref(), key, value).await?;
                                State::default()
                            }
                        };
                    }
                    Frame::Post(target) => {
                        let State::Map(params) = value else {
                            unreachable!("resolved POST parameter map")
                        };

                        value = match target {
                            Target::Link(link) => post_link(txn, link, params).await?,
                            Target::Borrowed(state, suffix) => {
                                state.post(txn, suffix.as_ref(), params).await?
                            }
                        };
                    }
                    Frame::After(next) => {
                        drop(value);
                        current = next;
                        break;
                    }
                    Frame::Cond(then, or_else) => {
                        let boolean = match condition(value)? {
                            Condition::Reference(reference) => {
                                frames.push(Frame::Cond(then, or_else));
                                current = Scalar::from(*reference);
                                break;
                            }
                            Condition::Value(value) => value,
                        };

                        let next = if boolean { then } else { or_else };
                        match next {
                            Scalar::Op(op) => {
                                let params = values_to_params_for_opdef(values, &op);
                                value = txn
                                    .execute_op(op, State::Map(params), self_link.cloned(), None)
                                    .await?;
                            }
                            kind => {
                                current = kind;
                                break;
                            }
                        }
                    }
                    Frame::WhileCond(closure, state) => {
                        let cond = state_to_opdef(value)?;
                        frames.push(Frame::WhileClosure(cond, state));
                        current = closure;
                        break;
                    }
                    Frame::WhileClosure(cond, state) => {
                        let closure = state_to_opdef(value)?;
                        frames.push(Frame::WhileState(cond, closure));
                        current = state;
                        break;
                    }
                    Frame::WhileState(cond, closure) => {
                        let result = txn
                            .execute_op(
                                cond.clone(),
                                State::Map(while_params(value.clone())?),
                                self_link.cloned(),
                                None,
                            )
                            .await?;

                        frames.push(Frame::WhileTest(cond, closure, value));
                        value = result;
                    }
                    Frame::WhileTest(cond, closure, state) => match condition(value)? {
                        Condition::Reference(reference) => {
                            frames.push(Frame::WhileTest(cond, closure, state));
                            current = Scalar::from(*reference);
                            break;
                        }
                        Condition::Value(false) => value = state,
                        Condition::Value(true) => {
                            value = txn
                                .execute_op(
                                    closure.clone(),
                                    State::Map(while_params(state)?),
                                    self_link.cloned(),
                                    None,
                                )
                                .await?;

                            frames.push(Frame::WhileState(cond, closure));
                        }
                    },
                    Frame::ForItems(op, name) => {
                        let items = tuple_state_to_items(value, "for_each")?;
                        frames.push(Frame::ForOp(items, name));
                        current = op;
                        break;
                    }
                    Frame::ForOp(items, name) => {
                        let op = state_to_opdef(value)?;
                        value = State::default();
                        for item in items {
                            let mut params = Map::new();
                            params.insert(name.clone(), item);
                            value = txn
                                .execute_op(
                                    op.clone(),
                                    State::Map(params),
                                    self_link.cloned(),
                                    None,
                                )
                                .await?;
                        }
                    }
                }
            }
        }
    }
}

enum Condition {
    Value(bool),
    Reference(Box<TCRef>),
}

fn condition<Txn: tc_collection::StorageContext>(state: State<Txn>) -> TCResult<Condition> {
    match state {
        State::Scalar(scalar) => match scalar {
            Scalar::Ref(reference) => Ok(Condition::Reference(reference)),
            Scalar::Value(value) => match &value {
                Value::Number(number) => Ok(Condition::Value((*number).cast_into())),
                Value::None => Err(TCError::bad_request("expected condition to be a boolean")),
                _ => Err(TCError::bad_request(format!(
                    "expected condition to be a scalar boolean; found {:?}",
                    Scalar::from(value)
                ))),
            },
            kind => Err(TCError::bad_request(format!(
                "expected condition to be a scalar boolean; found {:?}",
                kind
            ))),
        },
        State::None => Err(TCError::bad_request("expected condition to be a boolean")),
        _ => Err(TCError::bad_request("expected a scalar boolean condition")),
    }
}

fn capture_op<Txn: OpExecutor>(mut op: OpDef, values: &HashMap<Id, State<Txn>>) -> TCResult<OpDef> {
    op.validate()?;
    let mut captures = Vec::new();
    let mut required = std::collections::BTreeSet::new();
    op.requires(&mut required);
    for id in required {
        let Some(value) = values.get(&id).cloned() else {
            continue;
        };

        let scalar = Scalar::try_cast_from(value, |_| {
            TCError::bad_request(format!("OpDef capture ${id} must be a scalar value"))
        })?;

        captures.push((id, scalar));
    }

    if captures.is_empty() {
        return Ok(op);
    }

    match &mut op {
        OpDef::Get((_, form))
        | OpDef::Put((_, _, form))
        | OpDef::Post(form)
        | OpDef::Delete((_, form)) => form.splice(0..0, captures),
    };

    op.validate()?;
    Ok(op)
}

pub fn resolve_ref<Txn>(
    reference: TCRef,
    txn: &Txn,
    subject: Option<State<Txn>>,
) -> impl Future<Output = TCResult<State<Txn>>> + Send + 'static
where
    Txn: OpExecutor,
{
    // Clone eagerly so the returned future owns its transaction before polling.
    let txn = txn.clone();
    async move {
        resolve_scalar(
            Scalar::Ref(Box::new(reference)),
            &HashMap::new(),
            &txn,
            subject.as_ref(),
        )
        .await
    }
}

async fn put_link<Txn: OpExecutor>(
    txn: &Txn,
    link: Link,
    key: State<Txn>,
    value: State<Txn>,
) -> TCResult<State<Txn>> {
    let (key, value) = match crate::Collection::from_put(txn, &link, key, value).await? {
        tc_collection::route::LiteralPut::Created(state) => return Ok(state),
        tc_collection::route::LiteralPut::Unmatched { key, value } => (key, value),
    };

    // Constructing a fresh literal above has no effect on an existing receiver.
    txn.check_write()?;
    let key = Scalar::try_cast_from(key, |_| TCError::bad_request("expected scalar PUT key"))?;
    if is_state_link(&link) {
        let routes = super::Static::<Txn>::default();
        routes.put(txn, &link.path()[1..], key, value).await?;
    } else {
        txn.put(link, key, value).await?;
    }

    Ok(State::default())
}

fn is_state_link(link: &Link) -> bool {
    link.path()
        .first()
        .is_some_and(|segment| segment.as_str() == "state")
}

async fn get_link<Txn: OpExecutor>(txn: &Txn, link: Link, key: Scalar) -> TCResult<State<Txn>> {
    if is_state_link(&link) {
        let routes = super::Static::<Txn>::default();
        routes.get(txn, &link.path()[1..], key).await
    } else {
        txn.get(link, key).await
    }
}

async fn post_link<Txn: OpExecutor>(
    txn: &Txn,
    link: Link,
    params: Map<State<Txn>>,
) -> TCResult<State<Txn>> {
    if is_state_link(&link) {
        let routes = super::Static::<Txn>::default();
        routes.post(txn, &link.path()[1..], params).await
    } else {
        txn.post(link, params).await
    }
}

async fn delete_link<Txn: OpExecutor>(txn: &Txn, link: Link, key: Scalar) -> TCResult<State<Txn>> {
    if is_state_link(&link) {
        let routes = super::Static::<Txn>::default();
        routes.delete(txn, &link.path()[1..], key).await?;
    } else {
        txn.delete(link, key).await?;
    }

    Ok(State::default())
}

fn while_params<Txn: tc_collection::StorageContext>(
    state: State<Txn>,
) -> TCResult<Map<State<Txn>>> {
    let mut params = Map::new();
    let state_id: Id = "state"
        .parse()
        .map_err(|err| TCError::internal(format!("invalid while state id: {err}")))?;

    params.insert(state_id, state);
    Ok(params)
}

fn values_to_params_for_opdef<Txn: OpExecutor>(
    values: &HashMap<Id, State<Txn>>,
    opdef: &OpDef,
) -> Map<State<Txn>> {
    let mut params = Map::new();
    let mut required = std::collections::BTreeSet::new();
    opdef.requires(&mut required);
    for id in required {
        if let Some(value) = values.get(&id) {
            params.insert(id, value.clone());
        }
    }

    params
}

fn state_to_opdef<Txn: tc_collection::StorageContext>(state: State<Txn>) -> TCResult<OpDef> {
    match state {
        State::Scalar(scalar) => match scalar {
            Scalar::Op(op) => Ok(op),
            other => Err(TCError::bad_request(format!(
                "expected OpDef for While but found scalar {:?}",
                other
            ))),
        },
        _ => Err(TCError::bad_request("expected OpDef for While")),
    }
}

fn tuple_state_to_items<Txn: tc_collection::StorageContext>(
    state: State<Txn>,
    context: &str,
) -> TCResult<Vec<State<Txn>>> {
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

    Err(TCError::bad_request(format!(
        "expected tuple or map for {context}"
    )))
}

#[cfg(test)]
#[path = "../../tests/unit/resolve.rs"]
mod tests;
