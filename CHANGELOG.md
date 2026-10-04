# Changelog

All notable changes to FLOP are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) with two additions:
release dates carry the time and zone at which the version was cut, and
every release entry ends with a `### Results` section giving the full-suite
totals of the run that gated it. Versions are `MajorA.MajorB.Minor.Patch`;
patch `.1` of every minor is a test-only release.

## [0.1.1.1] - 2026-10-04 13:27 IST

The test release for Nelder-Mead. No library behaviour changes. The suite
holds the method to the papers it is written from, rule by rule and
property by property, and where it found the library wrong it says so with
a test that stays red until the fix. It found more than I expected: one of
the defects is a run that cannot end without an evaluation cap.

### Added

- The Nelder-Mead suite, 115 tests across 12 suites, built twice from one
  set of sources (strict and `-ffast-math`) and run on four trees:
  - every acceptance test of Lagarias, Reeds, Wright and Wright (section
    2.1) on both sides of its inequality and at its boundary, the trial
    points of both coefficient sets in one to eight dimensions, the
    tie-breaking rules, and the paper's own example on page 118, all in
    exact arithmetic and to the bit, with every expected point computed
    from the paper's formulas;
  - what the paper proves, checked on the method's own runs: the
    structure (2.9) of a step and the change index (2.8), the bound on how
    long the worst value can stand, Lemmas 3.1, 3.3, 3.5 and 4.3, and the
    one-dimensional properties of section 4; and Gao and Han's sufficient
    descent (2.2);
  - a slow transcription of section 2 in the test tree, which recomputes
    the centroid, re-sorts the simplex and computes the radius exactly
    every iteration, and which must ask for the points the method
    evaluated and stop where it stopped, to the bit in exact arithmetic
    and to rounding over runs of up to 96 dimensions; for every radius a
    run passes through, an x tolerance at that radius must stop the run
    exactly there;
  - Gao and Han's coefficients against their formula (4.1) for every
    dimension up to 4096, with the conditions (2.1); their Table 1, whose
    claim holds (the standard coefficients stall on their problem (4.3)
    from 20 dimensions, the adaptive ones do not), and which the method
    reproduces evaluation for evaluation in 7 of its 12 rows; and their
    Figures 1 and 2 (the adaptive share of reflection steps stays below
    0.45);
  - every stopping rule, the evaluation cap at every kind of step, the
    precision floor at three scales, box bounds (the initial simplex in a
    box, projection, the face test and its restarts), the batch channel's
    call shape, every `std::invalid_argument` on both entry points and the
    facade, `NELDER_MEAD` through `Minimizer`, a compile-time check that
    `flop/minimizer.hpp` does not pull in COBYLA, determinism, extreme
    objective values, and convergence on the sphere up to 128 dimensions
    and on the More, Garbow and Hillstrom problems of Gao and Han's Table
    2 with closed-form minimisers.

### Changed

- `docs/algorithms/nelder-mead.md` states why the standard coefficients
  stall in Gao and Han's terms (a growing share of reflection steps, and
  the descent their Theorem 2.1 guarantees shrinking as (n - 1)/(2 n^2))
  rather than as a distortion of the simplex. "Verification against the
  paper" now says what each rule is checked against, by page and equation,
  including the reading of the nonshrink ordering rule: printed as a
  maximum on page 116, defined as the smallest index by the text above it
  and by the example on page 118.
- Both Nelder-Mead pages state that a restart after the face test is one
  batch call of `n` points, and that when a point inside a batch call
  meets `stop_value`, every point of the call has been evaluated, counts,
  and can be the result, where the scalar channel stops at the point that
  met it.

### Known defects, pinned red

Thirteen tests fail on purpose in the strict binaries and twelve in the
`-ffast-math` ones, and will pass in 0.1.1.2. They pin six defects, and a
seventh is listed without a pin:

- The precision floor, `epsilon` times the largest of `initial_step` and the
  best vertex's coordinates, can sit below the smallest radius the
  arithmetic reaches. A simplex then stalls two units in the last place
  wide, every shrink rounds back onto the same points, and a run with only
  a cap shrinks until the cap; one with only a tolerance below the floor
  never ends. It happens near 1 as well as near 1e6.
  `V0111Status.ThePrecisionFloorIsReachableAtEveryScale`, and
  `V0111Status.WithOnlyACapTheRunEndsAtThePrecisionFloor` in the strict
  binaries.
- On a box, a convex problem whose minimum is off a face can stop
  `XtolReached` one unit in the last place from the bound, at a point that
  is not the minimiser: a reflection projected onto the bound starts an
  outside contraction that is projected too, and after the face test's
  restart the simplex collapses next to the face, where the test no longer
  applies. A minimum on a corner of the box can be missed the same way.
  `V0111Bounds.AConvexProblemWithItsMinimumOffTheFaceReachesIt`,
  `V0111Bounds.AMinimumOnACornerInFiveDimensionsIsReached` (strict and clang
  `-ffast-math`), `V0111Bounds.AnOptimumOutsideTheBoxLandsOnItsFace` (GCC
  `-ffast-math`).
- The face test does not run before a `RoundoffLimited` stop, so a run with
  only a cap ends flat on a face it could leave.
  `V0111Bounds.ARunWithOnlyACapStillGetsTheFaceTest`.
- An initial vertex moved to a bound is computed as `x0 + (bound - x0)`,
  which can round past the bound, so the method evaluates outside the box.
  The helper is shared, and COBYLA does the same.
  `V0111Bounds.AnInitialVertexMovedToABoundLandsOnIt`.
- Nothing checks a trial point for overflow, so a simplex near the top of
  the range hands the objective an infinite coordinate, in Nelder-Mead and
  in COBYLA's initial simplex.
  `V0111Fp.NoNonFiniteCoordinateIsEverHandedToTheObjective`.
- `Result::final_radius` after a cap or `stop_value` part way through a
  shrink or a restart, or through the first simplex inside a box narrower
  than `initial_step`, describes vertices that were placed and never
  evaluated. It should be the radius of the last simplex whose every
  vertex was evaluated, and `initial_step` while the first one is
  incomplete. `V0111Status.FinalRadiusOnACapInsideAShrinkIsTheLastEvaluatedSimplex`,
  `V0111Status.FinalRadiusWhileTheFirstSimplexIsIncompleteIsTheInitialStep`,
  `V0111Bounds.FinalRadiusOnACapInsideARestartIsTheLastEvaluatedSimplex`, and
  through them `V0111Status.EveryCapGivesAPrefixOfTheUncappedRun`,
  `V0111Reference.LongRunsAgreeToRoundingAndStopTogether` and
  `V0111Bookkeeping.TheStopIsDecidedExactlyAtEveryRadiusTheRunPasses`, whose
  points, stops and results otherwise agree with the transcription.
- Not pinned, because it cannot be seen from outside: with objective values
  at `-DBL_MAX` and `+DBL_MAX` in one simplex, the spread the f tolerances
  test overflows to infinity, which `-ffinite-math-only` makes undefined.
  The answer it gives is the right one, since the true spread exceeds every
  tolerance too.

### Results

283 tests across 26 suites (2.7 to 3.2 s per binary; Linux x86-64; GCC
16.2.1, GCC 14.3.1, clang 23.1.1, and GCC 16.2.1 with the library under
-ffast-math). Every test of the earlier releases passed in both binaries on
every tree. The strict binaries passed 270, the GCC -ffast-math binaries
271, and the clang -ffast-math binary 269, skipping the two tests that hand
a NaN back through the objective's return value, as in every release since
0.1.0.1. The failures are the pinned defects above.

## [0.1.1.0] - 2026-10-04 01:19 IST

The second algorithm: Nelder-Mead, written from the statement of
Lagarias, Reeds, Wright and Wright, with Gao and Han's coefficients for
higher dimensions, behind the same interface and facade as COBYLA. Its
test suite is the 0.1.1.1 release; the rules still to be read against the
paper's text are listed on the algorithm page.

### Added

- `flop::nelder_mead::minimize` and `minimize_batch`: the simplex method of
  Lagarias, Reeds, Wright and Wright (SIAM J. Optim. 9(1), 1998, section
  2), with reflection, expansion, outside and inside contraction and
  shrink, the paper's acceptance tests and its two tie-breaking rules.
  Gao and Han's dimension-dependent coefficients (Comput. Optim. Appl.
  51(1), 2012) are the default; `Options::adaptive_coefficients = false`
  selects the standard 1, 2, 1/2, 1/2, which are also used at n = 1.
- A step that does not shrink costs O(n) beyond its evaluations: vertices
  stay in fixed slots and only a rank index moves, the centroid comes from
  a running sum rebuilt every n + 1 replacements, the new vertex is ranked
  by binary search, and the stopping radius is bracketed from
  per-coordinate extents, with the O(n^2) radius computed only near a
  stop. On a sphere at a budget of 20 n the optimizer's own cost is 82 ns
  per evaluation at 16 parameters and 0.70 us at 128, unbounded, and 104
  ns and 0.87 us inside a box.
- Box bounds project every trial point onto the box. A simplex flattened
  onto a box face is tested off the face before a tolerance stop is
  reported, and restarts from a better point when there is one, so a run
  cannot stop on the wrong face and report `XtolReached`.
- The batch channel carries the initial simplex, `x0` first, and every
  shrink.
- `flop::Minimizer::create("NELDER_MEAD")` and
  `Minimizer::set_adaptive_coefficients`.
- `bench_nelder_mead`, at the parameter counts of `bench_cobyla`, bounded
  and unbounded and with both coefficient sets, and its CI smoke run.
- Documentation: `docs/algorithms/nelder-mead.md`, naming every deviation
  from the paper and every rule still to be verified against its text, and
  `docs/api/nelder_mead.md`.

### Changed

- **Breaking:** `Result::final_trust_radius` is renamed `final_radius`. It
  holds the scale the method reached when it stopped, COBYLA's trust
  radius or Nelder-Mead's simplex radius, and the x tolerances act on it.
- A facade setter for an option the selected algorithm does not have
  throws `std::invalid_argument`, and the constrained overloads throw for
  `NELDER_MEAD` before any evaluation, so nothing a caller passes is
  silently ignored.
- `flop/minimizer.hpp` does not include `flop/cobyla.hpp`. Code that
  reached COBYLA through the facade header includes `flop/cobyla.hpp`
  itself, or `flop/flop.hpp`.
- The scalar and batch evaluators and the box helpers (the initial-simplex
  offset and the projection onto the box) are shared by both algorithms
  under `flop/detail`. COBYLA's suite passes with only the rename applied.

### Results

168 tests across 14 suites, all passed in both binaries, strict and
-ffast-math, on every tree (0.2 to 0.4 s per binary; Linux x86-64; GCC
16.2.1, GCC 14.3.1, clang 23.1.1, and GCC 16.2.1 with the library under
-ffast-math). The clang -ffast-math binary skips the one test that hands a
NaN back through the objective's return value, as in every release since
0.1.0.1.

## [0.1.0.4] - 2026-09-19 17:16 IST

One test, the third time. It was red on GCC 13 in the 0.1.0.2 run and red
on clang 21 in the 0.1.0.3 run, both times in the `-ffast-math` binary
only, and the 0.1.0.3 rewrite did not touch the cause because I had the
cause wrong. This release has it right, with the evidence, and the test
now pins what it means to pin.

### Fixed

- `V0101Constrained.TheViolationReportedIsTheViolationAtTheReturnedPoint`
  rebuilt the violation at the returned point as a maximum taken from zero
  and asked for bit equality with the value the library reports. At that
  point Rosen-Suzuki's constraint values sit at the rounding floor, a few
  units of 1e-15, and one is often exactly `+0.0`: clang 22 under
  `-ffast-math` returns a point with `c[0]` at bits zero on my machine. When
  no constraint is violated the true violation is zero and the two sides
  can differ only in the zero's sign. The library reports `+0.0` by bits;
  under `-fno-signed-zeros` a compiler may compile the test's maximum as a
  max instruction that returns either zero, and GCC 13 and clang 21 do,
  while GCC 14, GCC 15, GCC 16 and clang 22 keep the branch. The 0.1.0.3
  entry attributed the failure to a second evaluation of the constraints
  rounding differently; that rewrite pinned a copy of the library's own
  values and failed the same way, which is what ruled the account out.
  The test keeps the record of every evaluation and the requirement that
  the returned point be one of them, and now admits only strictly violated
  constraints to the maximum, so every value compared is strictly positive
  and a feasible point yields the literal `0.0`.

### Results

168 tests across 14 suites, all passed in both binaries, strict and
-ffast-math, on every tree (0.2 s per binary; Linux x86-64; GCC 16.2.1, GCC
14.3.1, clang 22.1.8, and GCC 16.2.1 with the library under -ffast-math).
The clang -ffast-math binary skips the one test that hands a NaN back
through the objective's return value, as in the three releases before.

## [0.1.0.3] - 2026-09-19 16:59 IST

The 0.1.0.2 CI run was red on one leg out of six: GCC 13, the `-ffast-math`
test binary, one test. The library was right and the test was not, and the
leg that caught it was one I never build here. This release fixes the test
so that it pins what it means to pin, and makes CI build exactly the four
trees I build on my own machine, so that a green run here is the same
statement as a green run there.

### Changed

- CI builds the four trees the library is developed under and nothing
  else: the default GCC and the default clang of the image, `g++-14`, and
  the default GCC with the library itself under `-ffast-math`. Every leg
  still builds and runs both test binaries. The image is `ubuntu-26.04`,
  which ships every compiler named, so no leg installs anything. The lint
  workflow runs `clang-format` and `clang-tidy` 22 on the same image, the
  version the tree is formatted and tidied with. GCC 13 and clang 18 are
  no longer built anywhere, and the stated requirement follows what is
  verified: GCC 14 or later, clang 20 or later.

### Fixed

- `V0101Constrained.TheViolationReportedIsTheViolationAtTheReturnedPoint`
  evaluated Rosen-Suzuki's constraint polynomials a second time at the
  returned point and asked for bit equality with the value the library
  reports. Under `-ffast-math` a compiler may round the two call sites
  differently, and GCC 13 does, so the test failed on that leg while the
  library reported exactly the violation of the point it returned. The
  test now records every evaluation through `Options::on_evaluation`,
  requires the returned point to be bit-identical to one it recorded, and
  rebuilds the violation from that record's constraint values with negation
  and a comparison only, which no floating-point model can reorder. The
  pin is stronger for it: it also fails if the returned point was never
  evaluated.

### Results

168 tests across 14 suites, all passed in both binaries, strict and
-ffast-math, on every tree (0.2 s per binary; Linux x86-64; GCC 16.2.1, GCC
14.3.1, clang 22.1.8, and GCC 16.2.1 with the library under -ffast-math).
The clang -ffast-math binary skips the one test that hands a NaN back
through the objective's return value, as in 0.1.0.1 and 0.1.0.2.

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
