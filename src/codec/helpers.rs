use destream::de;
use pathlink::PathBuf;
use tc_ir::OpDef;
use tc_value::{Value, ValueType};

use super::parse::parse_link_value;

pub(super) async fn decode_value_entry<A: de::MapAccess>(
    value_type: ValueType,
    map: &mut A,
) -> Result<Value, A::Error> {
    match value_type {
        ValueType::Bytes => map
            .next_value::<bytes::Bytes>(())
            .await
            .map(|bytes| Value::Bytes(bytes.to_vec().into())),
        ValueType::Link => {
            let link_raw = map.next_value::<String>(()).await?;
            let link = parse_link_value(&link_raw).map_err(de::Error::custom)?;
            Ok(Value::Link(link))
        }
        ValueType::Number => map
            .next_value::<number_general::Number>(())
            .await
            .map(Value::Number),
        ValueType::None => {
            let _ = map.next_value::<de::IgnoredAny>(()).await?;
            Ok(Value::None)
        }
        ValueType::String => map.next_value::<String>(()).await.map(Value::String),
        ValueType::Tuple => map.next_value::<Vec<Value>>(()).await.map(Value::Tuple),
    }
}

pub(super) async fn decode_op_def_entry<A: de::MapAccess>(
    path: &PathBuf,
    map: &mut A,
) -> Result<Option<OpDef>, A::Error> {
    let op = if path.as_ref() == &tc_ir::OPDEF_GET[..] {
        OpDef::Get(map.next_value::<tc_ir::GetOp>(()).await?)
    } else if path.as_ref() == &tc_ir::OPDEF_PUT[..] {
        OpDef::Put(map.next_value::<tc_ir::PutOp>(()).await?)
    } else if path.as_ref() == &tc_ir::OPDEF_POST[..] {
        OpDef::Post(map.next_value::<tc_ir::PostOp>(()).await?)
    } else if path.as_ref() == &tc_ir::OPDEF_DELETE[..] {
        OpDef::Delete(map.next_value::<tc_ir::DeleteOp>(()).await?)
    } else {
        return Ok(None);
    };

    op.validate()
        .map_err(|err| de::Error::custom(err.to_string()))?;
    Ok(Some(op))
}

pub(super) async fn drain_remaining_entries<A: de::MapAccess>(map: &mut A) -> Result<(), A::Error> {
    while map.next_key::<de::IgnoredAny>(()).await?.is_some() {
        let _ = map.next_value::<de::IgnoredAny>(()).await?;
    }

    Ok(())
}
