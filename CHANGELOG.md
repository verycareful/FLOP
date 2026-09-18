# Changelog

All notable changes to FLOP are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) with two additions:
release dates carry the time and zone at which the version was cut, and
every release entry ends with a `### Results` section giving the full-suite
totals of the run that gated it. Versions are `MajorA.MajorB.Minor.Patch`;
patch `.1` of every minor is a test-only release.

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
