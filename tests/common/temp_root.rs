//! Private fixture ownership; callers retain the root until values and streams drop.

use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug)]
pub struct TempRoot(PathBuf);

impl TempRoot {
    pub fn new() -> Arc<Self> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();

        for attempt in 0u64.. {
            let path = std::env::temp_dir().join(format!(
                "tc-state-test-txn-{}-{stamp}-{attempt}",
                std::process::id()
            ));

            match std::fs::create_dir(&path) {
                Ok(()) => return Arc::new(Self(path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create transaction root: {error}"),
            }
        }

        unreachable!("directory-name space exhausted")
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[tokio::test]
async fn fixture_directories_follow_final_ownership_and_unwinding() {
    let (root, txn) = super::TestTxn::new();
    let path = root.path().to_path_buf();
    drop(txn);
    assert!(path.is_dir(), "the enclosing fixture still owns the root");
    drop(root);
    assert!(!path.exists());

    let (root, txn) = super::TestTxn::new();
    let path = root.path().to_path_buf();
    let clone = txn.clone();
    drop(root);
    drop(txn);
    assert!(path.is_dir(), "the cloned transaction still owns the root");
    drop(clone);
    assert!(!path.exists());

    let (root, txn) = super::TestTxn::new();
    let path = root.path().to_path_buf();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _root = root;
        let _txn = txn;
        panic!("fixture unwind");
    }));

    assert!(outcome.is_err());
    assert!(!path.exists());

    let (first, first_txn) = super::TestTxn::new();
    let (second, second_txn) = super::TestTxn::new();
    assert_ne!(first.path(), second.path());
    let first_path = first.path().to_path_buf();
    let second_path = second.path().to_path_buf();
    drop(first_txn);
    drop(first);
    assert!(!first_path.exists());
    assert!(second_path.is_dir());
    drop(second_txn);
    drop(second);
    assert!(!second_path.exists());
}
