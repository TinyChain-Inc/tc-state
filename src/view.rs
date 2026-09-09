use tc_error::TCResult;
use tc_ir::{IntoView, Map, Scalar};

use crate::{ClassDef, ClassInstance, Object, State};

/// A transaction-consistent terminal representation of a user-defined object.
pub enum ObjectView {
    Class(ClassDef),
    Instance {
        parent: Box<StateView>,
        class: ClassDef,
        members: Map<StateView>,
    },
}

/// A transaction-consistent terminal representation of [`State`].
pub enum StateView {
    None,
    Scalar(Scalar),
    Map(Map<StateView>),
    Tuple(Vec<StateView>),
    Collection(tc_collection::CollectionView),
    Object(ObjectView),
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
            State::Object(object) => match *object {
                Object::Class(class) => Ok(StateView::Object(ObjectView::Class(class))),
                Object::Instance(instance) => instance_view(instance, txn).await,
            },
        }
    })
}

async fn instance_view<Txn>(instance: ClassInstance<Txn>, txn: Txn) -> TCResult<StateView>
where
    Txn: tc_collection::StorageContext + 'static,
{
    let (parent, class, members) = instance.into_parts();
    let parent = Box::new(state_view(parent, txn.clone()).await?);
    let mut member_view = Map::new();
    for (id, member) in members {
        member_view.insert(id, state_view(member, txn.clone()).await?);
    }

    Ok(StateView::Object(ObjectView::Instance {
        parent,
        class,
        members: member_view,
    }))
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
