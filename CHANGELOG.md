# Changelog

All notable changes to FLOP are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) with two additions:
release dates carry the time and zone at which the version was cut, and
every release entry ends with a `### Results` section giving the full-suite
totals of the run that gated it. Versions are `MajorA.MajorB.Minor.Patch`;
patch `.1` of every minor is a test-only release.

## [0.1.0.1] - 2026-09-18 23:16 IST

The test release for COBYLA. No library behaviour changes; the suite pins
what 0.1.0.0 does, and where it found the library wrong it says so with a
test that stays red until the fix.

### Added

- The suite, 102 tests across 10 suites, built twice from one set of
  sources (strict and `-ffast-math`) and run on four trees (GCC 16, GCC 14,
  clang 22, and GCC 16 with the library itself under `-ffast-math`):
  - every `std::invalid_argument` an entry point can throw, with the
    guarantee that a rejected call evaluated nothing, and the mid-run
    `std::runtime_error` for a non-finite constraint value;
  - unconstrained problems with closed-form optima (sphere in 2, 16 and 96
    dimensions, Rosenbrock in 2 and 10, Beale, Powell's singular function,
    a one-dimensional cosine);
  - Powell's 1994 test problems 1 to 9 from the paper's starting points,
    held to their optima, plus Hock-Schittkowski 24 and 35;
  - box bounds: every evaluated point inside the box, the initial simplex
    included, and an inactive box giving the unbounded trajectory to the
    bit;
  - a replay of nineteen recorded NLopt 2.7.1 COBYLA runs on QAOA and
    MA-QAOA landscapes (`tests/data/nlopt-corpus/`, 2 to 80 parameters):
    the initial simplex replays bit-exactly on every run, and how far past
    it the two implementations agree is reported, not asserted (they part
    at the first trust-region step, which each computes its own way);
  - the batch channel's call shape, each `Status` on a problem built to
    produce it, the evaluation trace, the facade against the template
    entry point (to the bit when they share a floating-point model), and
    the finiteness check on every NaN payload and both infinities;
  - determinism to the bit, run against run.

### Changed

- Test file names carry the release that ships them
  (`test_v0101_<area>.cpp`).
- `docs/algorithms/cobyla.md` states the batch order as it is (the
  displaced points, then `x0`), names the limits below, and says what the
  suite holds the paper's rules to. `docs/Architecture.md` states why the
  finiteness check reads memory by reference: under `-ffinite-math-only`
  clang declares every `double` a function returns or takes by value free
  of NaN and infinity, so a NaN returned by value is undefined before any
  check can see it. GCC keeps the bits. One test that hands a NaN back
  through the objective's return value is skipped under clang with
  `-ffast-math` for that reason.
- clang-tidy conformance for the checks clang-tidy 18 runs (nodiscard on
  const accessors, optional access through a named reference, `Status` on
  `std::uint8_t`), and `actions/checkout` v5 in both workflows.

### Known defects, pinned red

Four tests fail on purpose and will pass in 0.1.0.2:

- `V0101Constrained.HS24TwoActiveConstraintsAndABox`,
  `V0101Constrained.HS35ConvexQuadraticOnAnActiveConstraint`,
  `V0101Bounds.ABoxWithConstraintsKeepsBothPromises`: with inequality
  constraints and a box together, the constrained path leaves the box at
  some trial points and can end at the unconstrained minimum with the
  constraint violated. Constraints alone and a box alone are held at every
  evaluation.
- `V0101Batch.TheFirstCallHoldsX0AndTheCoordinatePokes`: the first batch
  sends `x0` last; the contract is `x0` first, so that evaluation index 0
  is `x0` on both paths.

### Results

102 tests across 10 suites, 98 passed in both binaries, strict and
-ffast-math, on every tree (27 s per binary; Linux x86-64; GCC 16.2.1,
GCC 14, clang 22.1.8, and GCC 16.2.1 with the library under -ffast-math).
The four failures are the pinned defects above. The clang -ffast-math
binary additionally skips one test, named above, and passes 97.

## [0.1.0.0] - 2026-09-18 19:32 IST

The first release: COBYLA, written from Powell's 1994 paper, and the
library it lives in.

### Added

- `flop::cobyla::minimize` for an objective with or without inequality
  constraints, box bounds per coordinate, and `minimize_batch` for
  objectives that evaluate several points in one call (the initial simplex
  is one call of `n + 1` points).
- `flop::Minimizer`, a facade that selects an algorithm by name and is
  compiled once into the library, so a caller's floating-point flags never
  reach the solver.
- `Result` with `Status` (`XtolReached`, `FtolReached`, `StopValueReached`,
  `MaxEvaluationsReached`, `RoundoffLimited`), the final trust radius and
  the largest constraint violation at the returned point; every invalid
  input throws `std::invalid_argument` before the first evaluation.
- The trust-region subproblem in closed form when unconstrained and by
  bisection over Hildreth projections when constrained; box bounds are a
  projection inside the unconstrained step and linear rows of the
  constrained one.
- Fast-math discipline throughout: finiteness is read from the exponent
  bits, no NaN or infinity is ever used as a sentinel, and the tests and
  benchmarks build twice, strict and `-ffast-math`.
- CMake package with install and export, a Google Benchmark target, CI on
  GCC 13 and 14 and clang under both floating-point models, and lint on
  clang-format and clang-tidy with warnings as errors.
- Documentation: architecture, one page per API header, and the algorithm
  page listing every place the implementation departs from the paper.

### Results

No test suite yet: the suite is the 0.1.0.1 release. The release was gated
on Release builds under GCC 16.2.1 and clang 22.1.8, strict and
`-ffast-math`, and on a ten-problem smoke run under both models (Rosenbrock
in 2 and 10 dimensions, spheres in 3, 16 and 96, a linear objective on the
unit disc, a quadratic from an infeasible start, one-dimensional cosine,
the batch channel and the facade). Against NLopt 2.7.1 on the same points,
radii and budgets: the 96-dimensional sphere converges to 4e-13 in 7482
evaluations where NLopt reaches 2e-9 at its 10000 cap, the 16-dimensional
one in 887 evaluations, the box problem in 9 against 57, the unit-disc
problem in 62 against 118; Rosenbrock in 2 dimensions at a 5000 cap trails
NLopt (1.5e-3 against 4.5e-4). Linux, x86-64.
