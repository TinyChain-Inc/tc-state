use destream::en;

use crate::StateView;

impl<'en> en::IntoStream<'en> for StateView {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::None => encoder.encode_unit(),
            Self::Scalar(scalar) => scalar.into_stream(encoder),
            Self::Map(map) => map.into_stream(encoder),
            Self::Tuple(tuple) => tuple.into_stream(encoder),
            Self::Collection(collection) => collection.into_stream(encoder),
        }
    }
}
