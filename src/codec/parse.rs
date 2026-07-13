use std::fmt::Display;
use std::str::FromStr;

use pathlink::{Link, PathBuf};
use tc_error::TCError;
use tc_ir::Id;

pub(super) fn decode_err(context: &'static str, err: impl Display) -> TCError {
    tc_error::bad_request!("{context}: {err}")
}

pub(super) fn parse_state_map_id(key: &str) -> Result<Id, TCError> {
    key.parse::<Id>()
        .map_err(|err| decode_err("invalid state map key ID", err))
}

pub(super) fn parse_state_path(key: &str) -> Result<PathBuf, TCError> {
    key.parse::<PathBuf>()
        .map_err(|err| decode_err("invalid state type path", err))
}

pub(super) fn parse_link_value(link_raw: &str) -> Result<Link, TCError> {
    Link::from_str(link_raw).map_err(|err| decode_err("invalid Link scalar value", err))
}
