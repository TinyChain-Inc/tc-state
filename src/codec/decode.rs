use destream::de;
use number_general::Number;
use tc_ir::NativeClass;
use tc_ir::{Map, Scalar};
use tc_value::Value;

use super::class::StateType;
use super::helpers::{decode_op_def_entry, decode_value_entry, drain_remaining_entries};
use super::parse::{parse_state_map_id, parse_state_path};
use crate::runtime::State;

struct StateSeq<Txn>(Vec<State<Txn>>);

struct StateSeqVisitor<Txn> {
    context: Txn,
}

impl<Txn: tc_collection::StorageContext> de::Visitor for StateSeqVisitor<Txn> {
    type Value = Vec<State<Txn>>;

    fn expecting() -> &'static str {
        "a TinyChain state tuple"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut items = Vec::with_capacity(seq.size_hint().unwrap_or_default());
        let mut index = 0usize;
        while let Some(value) = seq
            .next_element::<State<Txn>>(self.context.subcontext(index.to_string()))
            .await?
        {
            items.push(value);
            index += 1;
        }

        Ok(items)
    }
}

impl<Txn: tc_collection::StorageContext> de::FromStream for StateSeq<Txn> {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(
        context: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        decoder
            .decode_seq(StateSeqVisitor { context })
            .await
            .map(Self)
    }
}

struct StateVisitor<Txn> {
    context: Txn,
}

impl<Txn: tc_collection::StorageContext> de::Visitor for StateVisitor<Txn> {
    type Value = State<Txn>;

    fn expecting() -> &'static str {
        "a TinyChain state value"
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

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut items = Vec::with_capacity(seq.size_hint().unwrap_or_default());
        let mut index = 0usize;
        while let Some(value) = seq
            .next_element::<State<Txn>>(self.context.subcontext(index.to_string()))
            .await?
        {
            items.push(value);
            index += 1;
        }

        Ok(State::Tuple(items))
    }

    async fn visit_map<A: de::MapAccess>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let Some(key) = map.next_key::<String>(()).await? else {
            return Ok(State::Map(Map::new()));
        };

        if !key.starts_with('/') {
            let mut out = Map::<State<Txn>>::new();
            let id = parse_state_map_id(&key).map_err(de::Error::custom)?;
            let value = map
                .next_value::<State<Txn>>(self.context.subcontext(key.clone()))
                .await?;
            out.insert(id, value);

            while let Some(key) = map.next_key::<String>(()).await? {
                let id = parse_state_map_id(&key).map_err(de::Error::custom)?;
                let value = map
                    .next_value::<State<Txn>>(self.context.subcontext(key.clone()))
                    .await?;
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
                let StateSeq(tuple) = map.next_value::<StateSeq<Txn>>(self.context).await?;
                drain_remaining_entries(&mut map).await?;
                Ok(State::Tuple(tuple))
            }
            StateType::Collection(collection_type) => {
                let collection =
                    tc_collection::decode_collection(collection_type, self.context, &mut map)
                        .await?;
                drain_remaining_entries(&mut map).await?;
                Ok(State::Collection(collection))
            }
            StateType::Scalar(value_type) => {
                let value = decode_value_entry(value_type, &mut map).await?;
                drain_remaining_entries(&mut map).await?;
                Ok(State::Scalar(Scalar::from(value)))
            }
        }
    }
}

impl<Txn: tc_collection::StorageContext> de::FromStream for State<Txn> {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(
        context: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        decoder.decode_any(StateVisitor { context }).await
    }
}
