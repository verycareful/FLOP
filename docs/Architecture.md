# Architecture

FLOP is a header-mostly C++23 library of derivative-free and, in time,
gradient-based optimizers, built for objectives that are expensive to
evaluate. Its design has one idea: the algorithm is a class template on the
objective, so an evaluation inlines into the optimizer loop, and one
type-erased facade serves the caller who chooses an algorithm by name at run
time.

## The two surfaces

The template entry points (`flop::cobyla::minimize` and its siblings) are the
fast path. They are constrained by concepts in `flop/concepts.hpp`:

- `ScalarObjective`: `f(x) -> double` for `std::span<const double> x`.
- `BatchObjective`: `f(xs, out)` evaluates every point of `xs` and writes one
  value per point. FLOP uses it where an algorithm has independent points to
  evaluate; COBYLA's initial simplex is one such place. FLOP never spawns a
  thread; the caller decides how a batch is evaluated.
- `ConstraintFunction`: `c(x, out)` writes the constraint values, feasible
  when every value is at least zero.

An objective that lacks a capability an algorithm needs fails to compile.

The facade, `flop::Minimizer`, holds the objective as a `std::function`,
selects the algorithm by name through `Minimizer::create`, and is compiled
once in the library. `create` is the only place a name is compared with a
string, and it throws on a name it does not know. A C shim would wrap the
facade, never the templates.

## Shared vocabulary

Every algorithm takes `flop::Options` (stopping rules, optional box bounds,
the initial step, an evaluation trace) and returns `flop::Result` (the best
point, its value, the evaluation count, a `Status`, the final trust radius,
the largest constraint violation). Algorithm-specific settings extend
`flop::Options` by inheritance, as `flop::cobyla::Options` does.

`Status` names why the run stopped. Reaching the evaluation cap is a status
like any other; `flop::converged(status)` is the question a caller means and
it answers false at the cap. There is no success flag.

## Floating point

Every guard in FLOP survives `-ffast-math`. Finiteness is read from the
exponent bits of the object representation, by reference, never through a
by-value double (`flop/detail/fp.hpp`): under `-ffinite-math-only` clang
declares every `double` a function returns or takes by value free of NaN and
infinity, so only a value read from memory can be tested at all. No infinity or NaN is used as a
sentinel anywhere: a missing bound is an empty `std::optional`, a running
best is a value plus a validity flag. The test suite is built twice in one
build, under the strict model and under `-ffast-math`, and both variants must
pass. The templated core runs under the caller's own model, which is why it
may not depend on the model at all.

## Adding an algorithm

1. A header `flop/<name>.hpp` with an `Options` struct extending
   `flop::Options` and template entry points over the concepts, validating
   input through `flop/detail/validate.hpp`.
2. The implementation under `flop/detail/`, citing its source paper by
   section, with no random state unless the method is stochastic by
   definition, and no arithmetic that depends on the floating-point model.
3. One `Impl` in `src/minimizer.cpp`, one entry in the name list, one line in
   `Minimizer::create`.
4. An algorithm page under `docs/algorithms/`, an API page for the header,
   tests in the next test release, and a benchmark row.
