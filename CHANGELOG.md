# Changelog

All notable changes to FLOP are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) with two additions:
release dates carry the time and zone at which the version was cut, and
every release entry ends with a `### Results` section giving the full-suite
totals of the run that gated it. Versions are `MajorA.MajorB.Minor.Patch`;
patch `.1` of every minor is a test-only release.

## [0.1.0.2] - 2026-09-19 13:48 IST

The patch the test release asked for, and then the release that makes
COBYLA the one I want to ship: the four pinned defects are fixed, the
constrained trust-region subproblem is solved exactly by an active-set
method, every rule of the iteration has been read against the paper's text
and one corrected, and a trust radius that can grow lifts the method off
the valley floor where the paper's radius crawls. Measured against Powell's
own tables and against NLopt 2.7.1 over twenty starting points per problem,
FLOP now reaches the final radius in fewer evaluations on each of the
paper's ten problems and reaches a smaller gap at a fixed budget on
Rosenbrock in two and ten dimensions and on Powell's singular function.

### Added

- `flop::cobyla::Options::trust_region_growth`, on by default. The paper's
  radius `rho` is never increased, and along a valley that is a crawl: once
  the simplex has flattened onto the floor its linear model is the exact
  gradient along it, every step is a perfect step of length `rho`, nothing
  fails, and `rho` stays (Rosenbrock from (-1.2, 1) spent 1446 evaluations
  at one radius). The step radius `Delta` now doubles after a step whose
  merit fall is at least half of the prediction and at least half of the
  previous step's, never past `initial_step`, and halves on a failed step
  down to `rho`, which keeps its own schedule as `Delta`'s floor and is cut
  only when a step fails with `Delta` already at it. The two radii are the
  arrangement of Powell's NEWUOA and BOBYQA. Off, the method is the paper's
  exactly, and the suite holds both.
- `include/flop/detail/active_set.hpp`: exact projection onto a polyhedron
  by the dual active-set method of Goldfarb and Idnani (1983) with the
  identity Hessian, a QR of the active rows updated by Givens rotations,
  and a certificate for an empty polyhedron.
- Tests, 66 new: the subproblem against closed forms to 1e-13 (an active
  halfspace and the ball, a chord of minimisers and the least-norm one, an
  infeasible origin, contradictory rows, a zero row, a duplicated row, a
  corner inside the ball, box rows with a constraint, one hyperplane in
  five variables, a warm workspace, the same bits twice); Powell's Tables 1
  and 2, problems (A) to (J) at his settings (`x0 = (1, ..., 1)`, `rhobeg
  = 1/2`, `rhoend = 1e-3` and `1e-4`) held to a factor of two of his
  evaluation counts, violations and distances and the square of that on
  the objective error, with a floor of single precision on a printed zero
  and of `rhoend` on a distance; the growing radius against the paper's on
  a valley, on a sphere and on a constrained problem; the reported
  violation's bits; and the corpus replay to a tolerance of 1e-9, which
  now reports how far each run agrees with NLopt beyond the bit-exact
  simplex (one to five evaluations, up to NLopt's first regenerated vertex,
  which NLopt perturbs).
- A constrained row in the benchmark: the sphere with one plane through
  its centre, active at the optimum, so the active-set subproblem runs on
  every iteration.

### Changed

- The constrained trust-region subproblem. The paper defines the step and
  gives one sentence on its computation; FLOP's solver for that definition
  is now exact. Both stages are a linear objective over the polyhedron of
  linearised constraints and box rows inside the ball, each followed along
  one scalar parameter through the exact projection: stage one finds the
  least violation level at which the relaxed polyhedron reaches the ball
  (on a fixed active set the projection of the origin is affine in the
  level, so the crossing is a quadratic in closed form, and an empty
  polyhedron hands back the level at which its certificate disappears);
  stage two minimises the objective through the ball's multiplier, where
  the projection is `u - v / nu` with `u` and `v` orthogonal and the `nu ->
  0` limit is the least-norm minimiser, the paper's own tie-break. The
  constrained suite went from 17 seconds to 3 milliseconds, the whole
  suite from 27 seconds to a fifth of one, and the constrained path costs
  0.7 microseconds per evaluation at 16 variables and 15 at 80, against
  2.2 and 248 for NLopt on the same host.
- The batch initial simplex sends `x0` first, then the `n` displaced
  points, so evaluation index 0 is `x0` on both paths, as the
  documentation had stated.
- The paper's problem (J), Hock-Schittkowski 108, joins the shared test
  problems.

### Fixed

- Constraints together with a box: the step no longer leaves the box and
  no longer stops at the unconstrained minimum. The box rows are part of
  the polyhedron and never relaxed by the violation level, and the step is
  clamped into the box afterwards, so the box holds to the bit. The three
  tests pinned red in 0.1.0.1 pass.
- The batch order test pinned red in 0.1.0.1 passes.
- The penalty rule of section 2: `mu` becomes twice the least penalty at
  which the step is predicted to lower the merit whenever it is below one
  and a half times that value, as the paper states, not only when the
  prediction was already nonpositive. On Fletcher's problem 9.1.15 at
  `rhoend = 1e-4` the run had ended on a base violating its constraint by
  1e-4; it now ends feasible.
- `max_constraint_violation` never carries a negative zero. A constraint
  that is exactly zero contributes `-0.0` to the maximum, and under
  `-ffast-math` the compiler is free to hand that sign back; the sign is
  now cleared through the integer representation, which no floating-point
  flag can undo.

### Results

168 tests across 14 suites, all passed in both binaries, strict and
-ffast-math, on every tree (0.2 s per binary; Linux x86-64; GCC 16.2.1, GCC
14, clang 22.1.8, and GCC 16.2.1 with the library under -ffast-math). The
clang -ffast-math binary skips the one test that hands a NaN back through
the objective's return value, as in 0.1.0.1.

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
