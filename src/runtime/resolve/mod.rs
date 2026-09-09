use std::collections::HashMap;
use std::sync::Arc;

use crate::State;
use futures::future::BoxFuture;
use pathlink::Link;
use safecast::{CastInto, TryCastFrom};
use tc_error::{TCError, TCResult};
use tc_ir::{After, Cond, ForEach, Id, Map, OpDef, OpRef, Public, Scalar, Subject, TCRef, While};
use tc_value::Value;

use super::StateExecutor;

mod params;

use self::params::resolve_params;

pub fn resolve_scalar<'a, Txn>(
    scalar: Scalar,
    values: &'a Arc<HashMap<Id, State<Txn>>>,
    txn: &'a Txn,
    self_link: Option<&'a State<Txn>>,
) -> BoxFuture<'a, TCResult<State<Txn>>>
where
    Txn: StateExecutor,
{
    Box::pin(async move {
        match scalar {
            Scalar::Value(value) => Ok(State::from(value)),
            Scalar::Op(op) => Ok(State::Scalar(Scalar::Op(capture_op(op, values)?))),
            Scalar::Map(map) => {
                let mut out = Map::new();
                for (key, value) in map {
                    out.insert(key, resolve_scalar(value, values, txn, self_link).await?);
                }
                Ok(State::Map(out))
            }
            Scalar::Tuple(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(resolve_scalar(item, values, txn, self_link).await?);
                }
                Ok(State::Tuple(out))
            }
            Scalar::Ref(r) => match *r {
                TCRef::Id(id_ref) if id_ref.as_str() == "self" => self_link
                    .cloned()
                    .ok_or_else(|| TCError::bad_request("OpDef has $self but no scope")),
                TCRef::Id(id_ref) => values
                    .get(id_ref.as_str())
                    .cloned()
                    .ok_or_else(|| TCError::not_found(format!("unknown id ${}", id_ref.as_str()))),
                TCRef::Op(op) => resolve_opref(op, values, txn, self_link).await,
                TCRef::Cond(cond) => resolve_cond(*cond, values, txn, self_link).await,
                TCRef::After(after) => resolve_after(*after, values, txn, self_link).await,
                TCRef::While(while_ref) => resolve_while(*while_ref, values, txn, self_link).await,
                TCRef::ForEach(for_each) => {
                    resolve_for_each(*for_each, values, txn, self_link).await
                }
            },
        }
    })
}

fn capture_op<Txn: StateExecutor>(
    mut op: OpDef,
    values: &HashMap<Id, State<Txn>>,
) -> TCResult<OpDef> {
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
) -> BoxFuture<'static, TCResult<State<Txn>>>
where
    Txn: StateExecutor,
{
    let txn = txn.clone();
    Box::pin(async move {
        resolve_scalar(
            Scalar::Ref(Box::new(reference)),
            &Arc::new(HashMap::new()),
            &txn,
            subject.as_ref(),
        )
        .await
    })
}

async fn put_link<Txn: StateExecutor>(
    txn: &Txn,
    link: Link,
    key: State<Txn>,
    value: State<Txn>,
) -> TCResult<State<Txn>> {
    if let Some(state) = crate::Collection::from_put(&link, key.clone(), value.clone())? {
        return Ok(state);
    }

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

async fn get_link<Txn: StateExecutor>(txn: &Txn, link: Link, key: Scalar) -> TCResult<State<Txn>> {
    if is_state_link(&link) {
        let routes = super::Static::<Txn>::default();
        routes.get(txn, &link.path()[1..], key).await
    } else {
        txn.get(link, key).await
    }
}

async fn post_link<Txn: StateExecutor>(
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

async fn delete_link<Txn: StateExecutor>(
    txn: &Txn,
    link: Link,
    key: Scalar,
) -> TCResult<State<Txn>> {
    if is_state_link(&link) {
        let routes = super::Static::<Txn>::default();
        routes.delete(txn, &link.path()[1..], key).await?;
    } else {
        txn.delete(link, key).await?;
    }
    Ok(State::default())
}

fn resolve_opref<Txn: StateExecutor>(
    op: OpRef,
    values: &Arc<HashMap<Id, State<Txn>>>,
    txn: &Txn,
    self_link: Option<&State<Txn>>,
) -> BoxFuture<'static, TCResult<State<Txn>>> {
    let values = Arc::clone(values);
    let txn = txn.clone();
    let self_link = self_link.cloned();
    Box::pin(async move {
        match op {
            OpRef::Get((subject, key)) => match subject {
                Subject::Ref(id_ref, suffix) => {
                    if id_ref.as_str() == "self" {
                        let state = self_link
                            .as_ref()
                            .ok_or_else(|| TCError::bad_request("OpDef has $self but no scope"))?;
                        let key = Scalar::try_cast_from(
                            resolve_scalar(key, &values, &txn, self_link.as_ref()).await?,
                            |_| TCError::bad_request("expected scalar GET key"),
                        )?;
                        return state.get(&txn, suffix.as_ref(), key).await;
                    }
                    let state = values.get(id_ref.as_str()).cloned().ok_or_else(|| {
                        TCError::not_found(format!("unknown id ${}", id_ref.as_str()))
                    })?;
                    let key = Scalar::try_cast_from(
                        resolve_scalar(key, &values, &txn, self_link.as_ref()).await?,
                        |_| TCError::bad_request("expected scalar GET key"),
                    )?;
                    state.get(&txn, suffix.as_ref(), key).await
                }
                Subject::Link(_) => {
                    let link = resolve_subject(subject, values.as_ref(), self_link.as_ref())?;
                    let key = Scalar::try_cast_from(
                        resolve_scalar(key, &values, &txn, self_link.as_ref()).await?,
                        |_| TCError::bad_request("expected scalar GET key"),
                    )?;
                    get_link(&txn, link, key).await
                }
            },
            OpRef::Put((subject, key, value)) => match subject {
                Subject::Ref(id_ref, suffix) => {
                    if id_ref.as_str() == "self" {
                        let state = self_link
                            .as_ref()
                            .ok_or_else(|| TCError::bad_request("OpDef has $self but no scope"))?;
                        let key = Scalar::try_cast_from(
                            resolve_scalar(key, &values, &txn, self_link.as_ref()).await?,
                            |_| TCError::bad_request("expected scalar PUT key"),
                        )?;
                        let value =
                            resolve_scalar(value, &values, &txn, self_link.as_ref()).await?;
                        state.put(&txn, suffix.as_ref(), key, value).await?;
                        return Ok(State::default());
                    }
                    let state = values.get(id_ref.as_str()).cloned().ok_or_else(|| {
                        TCError::not_found(format!("unknown id ${}", id_ref.as_str()))
                    })?;
                    let local_key = resolve_scalar(key, &values, &txn, self_link.as_ref()).await?;
                    let local_value =
                        resolve_scalar(value, &values, &txn, self_link.as_ref()).await?;
                    let key = Scalar::try_cast_from(local_key, |_| {
                        TCError::bad_request("expected scalar PUT key")
                    })?;
                    state.put(&txn, suffix.as_ref(), key, local_value).await?;
                    Ok(State::default())
                }
                Subject::Link(_) => {
                    let link = resolve_subject(subject, values.as_ref(), self_link.as_ref())?;
                    let local_key = resolve_scalar(key, &values, &txn, self_link.as_ref()).await?;
                    let local_value =
                        resolve_scalar(value, &values, &txn, self_link.as_ref()).await?;
                    put_link(&txn, link, local_key, local_value).await
                }
            },
            OpRef::Post((subject, params)) => match subject {
                Subject::Link(_) => {
                    let link = resolve_subject(subject, values.as_ref(), self_link.as_ref())?;
                    let params = resolve_params(params, &values, &txn, self_link.as_ref()).await?;
                    post_link(&txn, link, params).await
                }
                Subject::Ref(id_ref, suffix) => {
                    let state = if id_ref.as_str() == "self" {
                        self_link
                            .clone()
                            .ok_or_else(|| TCError::bad_request("OpDef has $self but no scope"))?
                    } else {
                        values.get(id_ref.as_str()).cloned().ok_or_else(|| {
                            TCError::not_found(format!("unknown id ${}", id_ref.as_str()))
                        })?
                    };
                    let params = resolve_params(params, &values, &txn, self_link.as_ref()).await?;
                    state.post(&txn, suffix.as_ref(), params).await
                }
            },
            OpRef::Delete((subject, key)) => match subject {
                Subject::Link(_) => {
                    let link = resolve_subject(subject, values.as_ref(), self_link.as_ref())?;
                    let key = Scalar::try_cast_from(
                        resolve_scalar(key, &values, &txn, self_link.as_ref()).await?,
                        |_| TCError::bad_request("expected scalar DELETE key"),
                    )?;
                    delete_link(&txn, link, key).await
                }
                Subject::Ref(id_ref, suffix) => {
                    if id_ref.as_str() == "self" {
                        let state = self_link
                            .as_ref()
                            .ok_or_else(|| TCError::bad_request("OpDef has $self but no scope"))?;
                        let key = Scalar::try_cast_from(
                            resolve_scalar(key, &values, &txn, self_link.as_ref()).await?,
                            |_| TCError::bad_request("expected scalar DELETE key"),
                        )?;
                        state.delete(&txn, suffix.as_ref(), key).await?;
                        return Ok(State::default());
                    }
                    let state = values.get(id_ref.as_str()).cloned().ok_or_else(|| {
                        TCError::not_found(format!("unknown id ${}", id_ref.as_str()))
                    })?;
                    let key = Scalar::try_cast_from(
                        resolve_scalar(key, &values, &txn, self_link.as_ref()).await?,
                        |_| TCError::bad_request("expected scalar DELETE key"),
                    )?;
                    state.delete(&txn, suffix.as_ref(), key).await?;
                    Ok(State::default())
                }
            },
        }
    })
}

fn resolve_subject<Txn: tc_collection::StorageContext>(
    subject: Subject,
    values: &HashMap<Id, State<Txn>>,
    self_link: Option<&State<Txn>>,
) -> TCResult<Link> {
    match subject {
        Subject::Link(link) => Ok(link),
        Subject::Ref(id_ref, suffix) => {
            let base = if id_ref.as_str() == "self" {
                state_to_link(
                    self_link
                        .ok_or_else(|| TCError::bad_request("OpDef has $self but no scope"))?,
                    "self",
                )?
            } else {
                let state = values.get(id_ref.as_str()).ok_or_else(|| {
                    TCError::not_found(format!("unknown id ${}", id_ref.as_str()))
                })?;
                state_to_link(state, id_ref.as_str())?
            };

            let mut link = base;
            for segment in suffix.as_ref() {
                link = link.append(segment.clone());
            }
            Ok(link)
        }
    }
}

fn state_to_link<Txn: tc_collection::StorageContext>(
    state: &State<Txn>,
    id: &str,
) -> TCResult<Link> {
    match state {
        State::Scalar(Scalar::Value(Value::Link(link))) => Ok(link.clone()),
        State::Scalar(Scalar::Value(Value::String(link))) => link.parse().map_err(|err| {
            TCError::bad_request(format!(
                "expected id ${id} to resolve to a valid link, found invalid string: {err}"
            ))
        }),
        _ => Err(TCError::bad_request(format!(
            "expected id ${id} to resolve to a link"
        ))),
    }
}

async fn resolve_cond<Txn: StateExecutor>(
    cond_ref: Cond,
    values: &Arc<HashMap<Id, State<Txn>>>,
    txn: &Txn,
    self_link: Option<&State<Txn>>,
) -> TCResult<State<Txn>> {
    let Cond {
        cond,
        then,
        or_else,
    } = cond_ref;

    let cond_state = resolve_scalar(Scalar::from(cond), values, txn, self_link).await?;
    let cond_value = resolve_bool_state(cond_state, values, txn, self_link).await?;
    let branch = if cond_value { then } else { or_else };

    match branch {
        Scalar::Op(op_def) => {
            let params = values_to_params_for_opdef(values, &op_def);
            txn.execute_op(op_def, State::Map(params), self_link.cloned())
                .await
        }
        scalar => resolve_scalar(scalar, values, txn, self_link).await,
    }
}

async fn resolve_after<Txn: StateExecutor>(
    after: After,
    values: &Arc<HashMap<Id, State<Txn>>>,
    txn: &Txn,
    self_link: Option<&State<Txn>>,
) -> TCResult<State<Txn>> {
    let After { when, then } = after;
    resolve_scalar(when, values, txn, self_link).await?;
    resolve_scalar(then, values, txn, self_link).await
}

async fn resolve_while<Txn: StateExecutor>(
    while_ref: While,
    values: &Arc<HashMap<Id, State<Txn>>>,
    txn: &Txn,
    self_link: Option<&State<Txn>>,
) -> TCResult<State<Txn>> {
    let While {
        cond,
        closure,
        state,
    } = while_ref;

    let cond_def = resolve_scalar(cond, values, txn, self_link)
        .await
        .and_then(state_to_opdef)?;
    let closure_def = resolve_scalar(closure, values, txn, self_link)
        .await
        .and_then(state_to_opdef)?;
    let mut state = resolve_scalar(state, values, txn, self_link).await?;

    loop {
        let cond_state = txn
            .execute_op(
                cond_def.clone(),
                State::Map(while_params(state.clone())?),
                self_link.cloned(),
            )
            .await?;

        let should_continue = resolve_bool_state(cond_state, values, txn, self_link).await?;

        if !should_continue {
            return Ok(state);
        }

        state = txn
            .execute_op(
                closure_def.clone(),
                State::Map(while_params(state)?),
                self_link.cloned(),
            )
            .await?;
    }
}

async fn resolve_for_each<Txn: StateExecutor>(
    for_each: ForEach,
    values: &Arc<HashMap<Id, State<Txn>>>,
    txn: &Txn,
    self_link: Option<&State<Txn>>,
) -> TCResult<State<Txn>> {
    let ForEach {
        items,
        op,
        item_name,
    } = for_each;

    let items = resolve_scalar(items, values, txn, self_link).await?;
    let items = tuple_state_to_items(items, "for_each")?;
    let op_def = resolve_scalar(op, values, txn, self_link)
        .await
        .and_then(state_to_opdef)?;

    let mut last_state: Option<State<Txn>> = None;
    for item in items {
        let mut params = Map::new();
        params.insert(item_name.clone(), item);
        last_state = Some(
            txn.execute_op(op_def.clone(), State::Map(params), self_link.cloned())
                .await?,
        );
    }

    Ok(last_state.unwrap_or_default())
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

fn values_to_params_for_opdef<Txn: StateExecutor>(
    values: &Arc<HashMap<Id, State<Txn>>>,
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
        State::Scalar(Scalar::Op(op)) => Ok(op),
        State::Scalar(other) => Err(TCError::bad_request(format!(
            "expected OpDef for While but found scalar {other:?}"
        ))),
        _ => Err(TCError::bad_request("expected OpDef for While")),
    }
}

fn tuple_state_to_items<Txn: tc_collection::StorageContext>(
    state: State<Txn>,
    context: &str,
) -> TCResult<Vec<State<Txn>>> {
    match state {
        State::Tuple(items) => Ok(items),
        State::Scalar(Scalar::Tuple(items)) => Ok(items.into_iter().map(State::Scalar).collect()),
        State::Map(map) => Ok(map
            .into_iter()
            .map(|(id, _value)| State::Scalar(Scalar::Value(Value::String(id.to_string()))))
            .collect()),
        State::Scalar(Scalar::Map(map)) => Ok(map
            .into_iter()
            .map(|(id, _value)| State::Scalar(Scalar::Value(Value::String(id.to_string()))))
            .collect()),
        _ => Err(TCError::bad_request(format!(
            "expected tuple or map for {context}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tc_ir::{Claim, NetworkTime, Transaction, TxnId};

    use super::*;
    use crate::runtime::tests::TestTxn;

    #[derive(Clone)]
    struct MockTxn {
        storage: TestTxn,
        outbound: Arc<Mutex<Vec<Link>>>,
        subjects: Arc<Mutex<Vec<Option<State<Self>>>>>,
    }

    impl MockTxn {
        fn new() -> Self {
            Self {
                storage: TestTxn::new(),
                outbound: Arc::default(),
                subjects: Arc::default(),
            }
        }
    }

    impl Transaction for MockTxn {
        fn id(&self) -> TxnId {
            self.storage.id()
        }

        fn timestamp(&self) -> NetworkTime {
            self.storage.timestamp()
        }

        fn claim(&self) -> &Claim {
            self.storage.claim()
        }
    }

    impl tc_collection::StorageContext for MockTxn {
        type File = tc_collection::PersistentFile;

        fn context(
            &self,
        ) -> impl std::future::Future<
            Output = TCResult<freqfs::DirLock<tc_collection::PersistentFile>>,
        > + Send {
            self.storage.context()
        }

        fn subcontext(&self, name: impl Into<String>) -> Self {
            let mut txn = self.clone();
            txn.storage = txn.storage.subcontext(name);
            txn
        }

        fn subcontext_unique(&self) -> Self {
            let mut txn = self.clone();
            txn.storage = txn.storage.subcontext_unique();
            txn
        }

        fn materialized_tensor_bytes(&self) -> usize {
            self.storage.materialized_tensor_bytes()
        }
    }

    impl StateExecutor for MockTxn {
        async fn resolve_class(&self, _identity: &Link) -> TCResult<crate::ClassDef> {
            Err(TCError::not_found("test Class"))
        }

        async fn get(&self, target: Link, _key: Scalar) -> TCResult<State<Self>> {
            self.outbound.lock().expect("outbound calls").push(target);
            Ok(State::from(Value::from(true)))
        }

        async fn put(&self, target: Link, _key: Scalar, _value: State<Self>) -> TCResult<()> {
            self.outbound.lock().expect("outbound calls").push(target);
            Ok(())
        }

        async fn post(&self, target: Link, _params: Map<State<Self>>) -> TCResult<State<Self>> {
            self.outbound.lock().expect("outbound calls").push(target);
            Ok(State::from(Value::from(true)))
        }

        async fn delete(&self, target: Link, _key: Scalar) -> TCResult<()> {
            self.outbound.lock().expect("outbound calls").push(target);
            Ok(())
        }

        async fn execute_op(
            &self,
            _definition: OpDef,
            _args: State<Self>,
            subject: Option<State<Self>>,
        ) -> TCResult<State<Self>> {
            self.subjects.lock().expect("subjects").push(subject);
            Ok(State::None)
        }
    }

    #[tokio::test]
    async fn preserves_outbound_transaction_and_concrete_subject() {
        let txn = MockTxn::new();
        let target: Link = "/lib/example-devco/math/1.0.0".parse().expect("target");
        let get = TCRef::Op(OpRef::Get((
            Subject::Link(target.clone()),
            Scalar::default(),
        )));
        resolve_ref(get, &txn, None).await.expect("outbound GET");
        assert_eq!(&*txn.outbound.lock().expect("outbound calls"), &[target]);

        let subject = State::from(Value::from("instance"));
        let cond = TCRef::Cond(Box::new(Cond::new(
            TCRef::Op(OpRef::Get((
                Subject::Link("/lib/example-devco/flag/1.0.0".parse().expect("flag")),
                Scalar::default(),
            ))),
            Scalar::Op(OpDef::Post(Vec::new())),
            Scalar::default(),
        )));
        resolve_ref(cond, &txn, Some(subject.clone()))
            .await
            .expect("nested OpDef");
        let subjects = txn.subjects.lock().expect("subjects");
        assert_eq!(subjects.len(), 1);
        assert!(matches!(subjects[0], Some(State::Scalar(_))));
    }
}

async fn resolve_bool_state<Txn: StateExecutor>(
    mut state: State<Txn>,
    values: &Arc<HashMap<Id, State<Txn>>>,
    txn: &Txn,
    self_link: Option<&State<Txn>>,
) -> TCResult<bool> {
    loop {
        match state {
            State::Scalar(Scalar::Ref(r)) => {
                state = resolve_scalar(Scalar::Ref(r), values, txn, self_link).await?;
            }
            State::Scalar(Scalar::Value(Value::Number(number))) => {
                return Ok(number.cast_into());
            }
            State::Scalar(Scalar::Value(Value::None)) | State::None => {
                return Err(TCError::bad_request(
                    "expected condition to be a boolean".to_string(),
                ));
            }
            State::Scalar(other) => {
                return Err(TCError::bad_request(format!(
                    "expected condition to be a scalar boolean; found {other:?}"
                )));
            }
            _ => return Err(TCError::bad_request("expected a scalar boolean condition")),
        }
    }
}
