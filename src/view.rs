use tc_error::TCResult;
use tc_ir::{IntoView, Map, Scalar};

use crate::State;

/// A transaction-consistent terminal representation of [`State`].
pub enum StateView {
    None,
    Scalar(Scalar),
    Map(Map<StateView>),
    Tuple(Vec<StateView>),
    Collection(tc_collection::CollectionView),
}

fn state_view<Txn>(
    state: State<Txn>,
    txn: Txn,
) -> futures::future::BoxFuture<'static, TCResult<StateView>>
where
    Txn: tc_collection::StorageContext + 'static,
{
    Box::pin(async move {
        match state {
            State::None => Ok(StateView::None),
            State::Scalar(scalar) => Ok(StateView::Scalar(scalar)),
            State::Map(map) => {
                let mut view = Map::new();
                for (id, state) in map {
                    view.insert(id, state_view(state, txn.clone()).await?);
                }
                Ok(StateView::Map(view))
            }
            State::Tuple(tuple) => {
                let mut view = Vec::with_capacity(tuple.len());
                for state in tuple {
                    view.push(state_view(state, txn.clone()).await?);
                }
                Ok(StateView::Tuple(view))
            }
            State::Collection(collection) => {
                collection.into_view(txn).await.map(StateView::Collection)
            }
            State::Object(_) => Err(tc_error::TCError::new(
                tc_error::ErrorKind::NotImplemented,
                "Class/instance views require the canonical tcv2#68 wire contract",
            )),
        }
    })
}

impl<Txn> IntoView for State<Txn>
where
    Txn: tc_collection::StorageContext + 'static,
{
    type Txn = Txn;
    type View = StateView;

    async fn into_view(self, txn: Txn) -> TCResult<Self::View> {
        state_view(self, txn).await
    }
}
