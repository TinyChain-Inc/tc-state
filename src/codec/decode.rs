use destream::de;
use number_general::Number;
use tc_collection::btree::DecodedBTreePayload;
use tc_ir::NativeClass;
use tc_ir::{Map, Scalar};
use tc_value::Value;

use super::class::{CollectionType, StateType};
use super::helpers::{decode_op_def_entry, decode_value_entry, drain_remaining_entries};
use super::parse::{parse_state_map_id, parse_state_path};
use crate::runtime::{BTreeCollection, Collection, State, StateContext, Tensor};

struct StateSeq(Vec<State>);

impl de::FromStream for StateSeq {
    type Context = StateContext;

    async fn from_stream<D: de::Decoder>(
        context: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct StateSeqVisitor {
            context: StateContext,
        }

        impl de::Visitor for StateSeqVisitor {
            type Value = Vec<State>;

            fn expecting() -> &'static str {
                "a TinyChain state tuple"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut items: Vec<State> = if let Some(size) = seq.size_hint() {
                    Vec::with_capacity(size)
                } else {
                    Vec::new()
                };

                while let Some(value) = seq.next_element::<State>(self.context.clone()).await? {
                    items.push(value);
                }

                Ok(items)
            }
        }

        decoder
            .decode_seq(StateSeqVisitor { context })
            .await
            .map(StateSeq)
    }
}

impl de::FromStream for State {
    type Context = StateContext;

    async fn from_stream<D: de::Decoder>(
        context: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct StateVisitor {
            context: StateContext,
        }

        impl de::Visitor for StateVisitor {
            type Value = State;

            fn expecting() -> &'static str {
                "a TinyChain state placeholder"
            }

            fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(State::None)
            }

            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(State::None)
            }

            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(State::from(Number::from(value)))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(State::from(Number::from(value)))
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(State::from(Number::from(value)))
            }

            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
                Ok(State::from(Number::from(value)))
            }

            fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(State::from(Value::from(value)))
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut items: Vec<State> = if let Some(size) = seq.size_hint() {
                    Vec::with_capacity(size)
                } else {
                    Vec::new()
                };

                while let Some(value) = seq.next_element::<State>(self.context.clone()).await? {
                    items.push(value);
                }

                Ok(State::Tuple(items))
            }

            async fn visit_map<A: de::MapAccess>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let Some(key) = map.next_key::<String>(()).await? else {
                    return Ok(State::Map(Map::new()));
                };

                if !key.starts_with('/') {
                    let mut out = Map::<State>::new();
                    let value = map.next_value::<State>(self.context.clone()).await?;
                    let id = parse_state_map_id(&key).map_err(de::Error::custom)?;
                    out.insert(id, value);

                    while let Some(key) = map.next_key::<String>(()).await? {
                        let value = map.next_value::<State>(self.context.clone()).await?;
                        let id = parse_state_map_id(&key).map_err(de::Error::custom)?;
                        out.insert(id, value);
                    }

                    return Ok(State::Map(out));
                }

                let path = parse_state_path(&key).map_err(de::Error::custom)?;

                if let Some(op_def) = decode_op_def_entry(&path, &mut map).await? {
                    drain_remaining_entries(&mut map).await?;
                    return Ok(State::Scalar(Scalar::Op(op_def)));
                }

                let state_type = StateType::from_path(&path).ok_or_else(|| {
                    de::Error::invalid_value(path.to_string(), "a known TinyChain state type path")
                })?;

                match state_type {
                    StateType::Tuple => {
                        let StateSeq(tuple) =
                            map.next_value::<StateSeq>(self.context.clone()).await?;
                        drain_remaining_entries(&mut map).await?;
                        Ok(State::Tuple(tuple))
                    }
                    StateType::Collection(CollectionType::BTree(_)) => {
                        let decode_context = self
                            .context
                            .state_decode_context()
                            .map_err(de::Error::custom)?;
                        let payload = map
                            .next_value::<DecodedBTreePayload>(decode_context)
                            .await?;
                        drain_remaining_entries(&mut map).await?;

                        Ok(State::Collection(Collection::from(
                            BTreeCollection::with_schema(payload.schema, payload.btree),
                        )))
                    }
                    StateType::Collection(CollectionType::Tensor(_)) => {
                        let tensor = map.next_value::<Tensor>(()).await?;
                        drain_remaining_entries(&mut map).await?;
                        Ok(State::Collection(Collection::Tensor(tensor)))
                    }
                    StateType::Collection(CollectionType::Table(_)) => Err(de::Error::custom(
                        "Table decode requires a host collection root and is not available in StateContext",
                    )),
                    StateType::Scalar(value_type) => {
                        let value = decode_value_entry(value_type, &mut map).await?;
                        drain_remaining_entries(&mut map).await?;
                        Ok(State::Scalar(Scalar::from(value)))
                    }
                }
            }
        }

        decoder.decode_any(StateVisitor { context }).await
    }
}
