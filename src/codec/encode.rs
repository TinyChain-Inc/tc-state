use destream::en::{self, EncodeMap, IntoStream, MapStream};
use futures::stream;
use tc_value::class::NativeClass;

use crate::{ObjectType, ObjectView, StateType, StateView};

impl<'en> IntoStream<'en> for StateView {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::None => encoder.encode_unit(),
            Self::Scalar(scalar) => scalar.into_stream(encoder),
            Self::Collection(collection) => collection.into_stream(encoder),
            Self::Tuple(items) => encoder.encode_seq_stream(stream::iter(items)),
            Self::Map(items) => encoder.encode_map_stream(stream::iter(items)),
            Self::Object(object) => (*object).into_stream(encoder),
        }
    }
}

impl<'en> IntoStream<'en> for ObjectView {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let mut map = encoder.encode_map(Some(1))?;
        match self {
            Self::Class(class) => {
                map.encode_entry(
                    StateType::Object(ObjectType::Class).path().to_string(),
                    class,
                )?;
            }
            Self::Instance {
                parent,
                class,
                members,
            } => {
                map.encode_entry(
                    StateType::Object(ObjectType::Instance).path().to_string(),
                    (*parent, class, MapStream::from(stream::iter(members))),
                )?;
            }
        }

        map.end()
    }
}
