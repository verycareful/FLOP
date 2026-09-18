# Contributing

## Licence

FLOP is under the Mozilla Public License 2.0, and contributions are accepted
under the same licence: inbound equals outbound. Every source file carries
the MPL-2.0 notice from Exhibit A of the licence; a new file gets the same
header, with the copyright line as it is in the existing files.

## Provenance

Every algorithm in FLOP is written from its published description and cites
the paper by section in the file header. No code is taken from any other
implementation, whatever its licence; other implementations may be read for
understanding, and nothing more. A contribution that cannot say where its
algorithm came from, or that carries another project's code, is not merged.

## Style

- The tree is formatted with the `.clang-format` at the root. Run it before
  committing.
- Comments explain the code as it stands: how it works, why it is there,
  what breaks without it. A comment about what changed, when, or why it
  changed belongs in the commit message and the changelog, not in the code.
- No em dashes and no double hyphens anywhere that ships: code, comments,
  documentation, commit messages. A colon, a comma, parentheses or two
  sentences instead.
- No numeric constant without a source: `<numbers>` for mathematical
  constants, `std::numeric_limits` for type properties, the paper for an
  algorithm's parameters.
- Every finiteness check goes through `flop/detail/fp.hpp`. No `isfinite`,
  no comparison with a NaN, no infinity used as a sentinel: the suite runs
  under `-ffast-math` and a guard that only works under the strict model is
  not a guard.
- Commit messages follow Conventional Commits: `type(scope): summary`, with
  the scope naming the area (`cobyla`, `facade`, `build`, `docs`, `tests`).

## Tests

Every feature release is followed by a test-only release that pins it. Test
files are named `test_v<version>_<area>.cpp` with the version of the feature
they pin, and the suite is built twice, under the strict and the fast-math
floating-point models; both binaries must pass on every CI leg.

## Issues

A bug report states what was observed, what was expected, and a reproducer
using only the public API. A feature request states the problem before the
proposed change. Severity labels: `low`, `medium`, `high`, `critical`.
