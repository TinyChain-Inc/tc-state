use super::*;

pub(super) fn resolve_params<Txn: StateExecutor>(
    params: Map<Scalar>,
    values: &Arc<HashMap<Id, State<Txn>>>,
    txn: &Txn,
    self_link: Option<&State<Txn>>,
) -> BoxFuture<'static, TCResult<Map<State<Txn>>>> {
    let values = Arc::clone(values);
    let txn = txn.clone();
    let self_link = self_link.cloned();
    Box::pin(async move {
        let mut resolved = Map::new();
        for (key, value) in params {
            let value = resolve_scalar(value, &values, &txn, self_link.as_ref()).await?;
            resolved.insert(key, value);
        }
        Ok(resolved)
    })
}
