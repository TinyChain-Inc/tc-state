#![forbid(unsafe_code)]

mod codec;
mod runtime;
mod view;

pub use runtime::*;
pub use view::StateView;

#[cfg(test)]
mod architecture_tests {
    #[test]
    fn state_transaction_type_is_explicit() {
        let state = include_str!("runtime/mod.rs");
        assert!(!state.contains("State<Txn ="));
    }

    #[test]
    fn views_and_codecs_have_separate_ownership() {
        let view = include_str!("view.rs");
        assert!(!view.contains("destream"));
        assert!(!view.contains("IntoStream"));

        let encode = include_str!("codec/encode.rs");
        for forbidden in ["Handler", "Public", "Route<", ".route("] {
            assert!(
                !encode.contains(forbidden),
                "state encoding must not depend on {forbidden}"
            );
        }
    }
}
