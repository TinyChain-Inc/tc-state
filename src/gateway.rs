//! Transaction-bound access to host resources.

use std::future::Future;

use pathlink::Link;
use tc_ir::{Map, Scalar};

use super::{ClassDef, State};

/// Trusted host capability for resource access and Class lookup.
///
/// Calls retain the transaction's authority and execution scope. Resource owners
/// enforce their permissions; implementing this trait does not grant access.
pub trait Gateway: tc_collection::StorageContext + Sized + 'static {
    /// Reject effects forbidden by the inherited execution scope before mutation.
    /// This is an execution-scope check, not complete resource authorization.
    /// Pure POST calculations and fresh literal construction remain available.
    fn check_write(&self) -> tc_error::TCResult<()>;

    fn resolve_class(
        &self,
        identity: &Link,
    ) -> impl Future<Output = tc_error::TCResult<ClassDef>> + Send;

    fn get(
        &self,
        target: Link,
        key: Scalar,
    ) -> impl Future<Output = tc_error::TCResult<State<Self>>> + Send;

    fn put(
        &self,
        target: Link,
        key: Scalar,
        value: State<Self>,
    ) -> impl Future<Output = tc_error::TCResult<()>> + Send;

    fn post(
        &self,
        target: Link,
        params: Map<State<Self>>,
    ) -> impl Future<Output = tc_error::TCResult<State<Self>>> + Send;

    fn delete(
        &self,
        target: Link,
        key: Scalar,
    ) -> impl Future<Output = tc_error::TCResult<()>> + Send;
}
