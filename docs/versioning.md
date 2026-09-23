# Versioning and release policy

BoostDropout uses Semantic Versioning.

- A patch release fixes a defect without changing a documented public API or an
  established experimental protocol.
- A minor release adds backward-compatible functionality.
- A major release may change a public API, artifact schema, or default protocol
  in a way that requires downstream changes.

Every release must update `pyproject.toml`, `CHANGELOG.md`, and the release
tag. A release candidate must pass CI and the committed smoke configuration
from a clean virtual environment. Research results must always cite the Git
commit and configuration used, rather than relying on a package version alone.
