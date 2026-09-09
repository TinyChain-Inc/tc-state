use destream::en::{self, EncodeMap};

use tc_value::class::NativeClass;

use crate::{ObjectType, ObjectView, StateType, StateView};

impl<'en> en::IntoStream<'en> for StateView {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::None => encoder.encode_unit(),
            Self::Scalar(scalar) => scalar.into_stream(encoder),
            Self::Map(map) => map.into_stream(encoder),
            Self::Tuple(tuple) => tuple.into_stream(encoder),
            Self::Collection(collection) => collection.into_stream(encoder),
            Self::Object(ObjectView::Class(class)) => {
                let mut map = encoder.encode_map(Some(1))?;
                map.encode_entry(
                    StateType::Object(ObjectType::Class).path().to_string(),
                    class,
                )?;
                map.end()
            }
            Self::Object(ObjectView::Instance {
                parent,
                class,
                members,
            }) => {
                let mut map = encoder.encode_map(Some(1))?;
                map.encode_entry(
                    StateType::Object(ObjectType::Instance).path().to_string(),
                    (parent, class, members),
                )?;
                map.end()
            }
        }
    }
}
