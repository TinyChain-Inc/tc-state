# tc-state roadmap

This file contains only unimplemented State work. Current ownership is described
in the crate README and workspace architecture.

## Planned work

- Extend canonical native State routing only as new shared runtime behavior is
  implemented; adapters and the server must continue delegating to it.
- Support executable Service member values through the existing `State`,
  `Route<State>`, and `StateExecutor` contracts without a Service-specific
  resolver.
- Preserve one Tensor State variant while `tc-collection` replaces its private
  in-memory backend with persistent storage.
- Integrate future Chain values through the ordinary recursive State and
  collection contracts; do not add replay, filesystem, or host policy here.
