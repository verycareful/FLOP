# cobyla

Powell's COBYLA: derivative-free minimisation with inequality constraints and
box bounds. The algorithm is described in `docs/algorithms/cobyla.md`.

```cpp
#include "flop/cobyla.hpp"
// namespace flop::cobyla
```

## Options

```cpp
struct Options : flop::Options {
    double final_trust_radius = 0.0;   // rhoend; 0 derives it from the x tolerances
    bool trust_region_growth = true;   // let the step radius grow above rho; false is the paper's method
};
```

The shared fields are documented in `docs/api/options.md`.

## Functions

```cpp
template <ScalarObjective F>
Result minimize(F&& f, std::span<const double> x0, const Options& opts);

template <ScalarObjective F, ConstraintFunction C>
Result minimize(F&& f, C&& c, std::size_t n_constraints,
                std::span<const double> x0, const Options& opts);

template <BatchObjective F>
Result minimize_batch(F&& f, std::span<const double> x0, const Options& opts);

template <BatchObjective F, ConstraintFunction C>
Result minimize_batch(F&& f, C&& c, std::size_t n_constraints,
                      std::span<const double> x0, const Options& opts);
```

`f` is called with a point and returns its objective value. `c` is called
with a point and a span of `n_constraints` doubles and writes the constraint
values; a point is feasible when every value is at least zero. The batch
forms use `f(xs, out)` for the initial simplex, `n + 1` independent points
with `x0` first, and one point at a time afterwards.

Every entry point returns the best point seen with the reason the run ended
(`docs/api/result.md`).

## Exceptions

- `std::invalid_argument` at entry: empty or non-finite `x0`, no stopping
  criterion, a negative or non-finite tolerance, a non-positive
  `initial_step`, bounds of the wrong length, a lower bound above its upper
  bound, or `x0` outside the bounds.
- `std::runtime_error` during the run: the objective or a constraint
  returned a non-finite value.
- Anything the objective throws propagates unchanged.

## Example

```cpp
#include <span>
#include "flop/cobyla.hpp"

double rosenbrock(std::span<const double> x) {
    const double a = 1.0 - x[0], b = x[1] - x[0] * x[0];
    return a * a + 100.0 * b * b;
}

int main() {
    flop::cobyla::Options opts;
    opts.stopping.xtol_rel = 1e-8;
    opts.stopping.max_evaluations = 2000;
    opts.initial_step = 0.5;
    const double x0[2] = {-1.2, 1.0};
    const flop::Result r = flop::cobyla::minimize(rosenbrock, x0, opts);
    return flop::converged(r.status) ? 0 : 1;
}
```
