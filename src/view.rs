use std::collections::btree_map;
use std::vec;

use tc_error::{TCError, TCResult};
use tc_ir::{Id, IntoView, Map, Scalar};

use crate::{ClassDef, Object, State};

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
/// Collection streams retain their own native resources until consumption or drop.
pub enum StateView {
    None,
    Scalar(Scalar),
    Map(Map<StateView>),
    Tuple(Vec<StateView>),
    Collection(tc_collection::CollectionView),
    Object(Box<ObjectView>),
}

enum Frame<Txn: tc_collection::StorageContext> {
    Tuple {
        remaining: vec::IntoIter<State<Txn>>,
        values: Vec<StateView>,
    },
    Map {
        remaining: btree_map::IntoIter<Id, State<Txn>>,
        key: Id,
        values: Map<StateView>,
    },
    InstanceParent {
        class: Box<ClassDef>,
        members: Map<State<Txn>>,
    },
    InstanceMembers {
        parent: Box<StateView>,
        class: Box<ClassDef>,
        remaining: btree_map::IntoIter<Id, State<Txn>>,
        key: Id,
        values: Map<StateView>,
    },
}

async fn state_view<Txn>(state: State<Txn>, txn: Txn) -> TCResult<StateView>
where
    Txn: tc_collection::StorageContext + 'static,
{
    let mut frames = Vec::new();
    let mut current = state;
    loop {
        frames
            .try_reserve(1)
            .map_err(|error| TCError::internal(format!("cannot reserve view frames: {error}")))?;

        let mut view = match current {
            State::None => StateView::None,
            State::Scalar(scalar) => StateView::Scalar(scalar),
            State::Collection(collection) => {
                StateView::Collection(collection.into_view(txn.clone()).await?)
            }
            State::Chain(chain) => StateView::Collection(chain.into_view(txn.clone()).await?),
            State::Tuple(items) => {
                let mut remaining = items.into_iter();
                if let Some(next) = remaining.next() {
                    frames.push(Frame::Tuple {
                        remaining,
                        values: Vec::new(),
                    });

                    current = next;
                    continue;
                }

                StateView::Tuple(Vec::new())
            }
            State::Map(items) => {
                let mut remaining = items.into_iter();
                if let Some((key, next)) = remaining.next() {
                    frames.push(Frame::Map {
                        remaining,
                        key,
                        values: Map::new(),
                    });

                    current = next;
                    continue;
                }

                StateView::Map(Map::new())
            }
            State::Object(object) => match *object {
                Object::Class(class) => StateView::Object(Box::new(ObjectView::Class(class))),
                Object::Instance(instance) => {
                    let (parent, class, members) = instance.into_parts();
                    frames.push(Frame::InstanceParent {
                        class: Box::new(class),
                        members,
                    });

                    current = parent;
                    continue;
                }
            },
        };

        loop {
            let Some(frame) = frames.pop() else {
                return Ok(view);
            };

            view = match frame {
                Frame::Tuple {
                    mut remaining,
                    mut values,
                } => {
                    values.try_reserve(1).map_err(|error| {
                        TCError::internal(format!("cannot reserve view tuple: {error}"))
                    })?;

                    values.push(view);
                    if let Some(next) = remaining.next() {
                        frames.push(Frame::Tuple { remaining, values });
                        current = next;
                        break;
                    }

                    StateView::Tuple(values)
                }
                Frame::Map {
                    mut remaining,
                    key,
                    mut values,
                } => {
                    values.insert(key, view);
                    if let Some((key, next)) = remaining.next() {
                        frames.push(Frame::Map {
                            remaining,
                            key,
                            values,
                        });

                        current = next;
                        break;
                    }

                    StateView::Map(values)
                }
                Frame::InstanceParent { class, members } => {
                    let mut remaining = members.into_iter();
                    if let Some((key, next)) = remaining.next() {
                        frames.push(Frame::InstanceMembers {
                            parent: Box::new(view),
                            class,
                            remaining,
                            key,
                            values: Map::new(),
                        });

                        current = next;
                        break;
                    }

                    StateView::Object(Box::new(ObjectView::Instance {
                        parent: Box::new(view),
                        class: *class,
                        members: Map::new(),
                    }))
                }
                Frame::InstanceMembers {
                    parent,
                    class,
                    mut remaining,
                    key,
                    mut values,
                } => {
                    values.insert(key, view);
                    if let Some((key, next)) = remaining.next() {
                        frames.push(Frame::InstanceMembers {
                            parent,
                            class,
                            remaining,
                            key,
                            values,
                        });

                        current = next;
                        break;
                    }

                    StateView::Object(Box::new(ObjectView::Instance {
                        parent,
                        class: *class,
                        members: values,
                    }))
                }
            };
        }
    }
}

impl<Txn> IntoView for State<Txn>
where
    Txn: tc_collection::StorageContext + 'static,
{
    type Txn = Txn;
    type View = StateView;
    async fn into_view(self, txn: Txn) -> TCResult<Self::View> {
        state_view(self, txn.clone())
            .await
            .map_err(|error| txn.storage_error(error))
    }
}
