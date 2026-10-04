# nelder_mead

The Nelder-Mead simplex method: derivative-free minimisation, optionally
inside box bounds. The algorithm is described in
`docs/algorithms/nelder-mead.md`.

```cpp
#include "flop/nelder_mead.hpp"
// namespace flop::nelder_mead
```

## Options

```cpp
struct Options : flop::Options {
    bool adaptive_coefficients = true;   // Gao and Han's; false is the standard 1, 2, 1/2, 1/2
};
```

The shared fields are documented in `docs/api/options.md`. For this method
the x tolerances act on the simplex radius, the largest coordinate distance
from a vertex to the best one, and the f tolerances on the spread of `f`
over the simplex.

## Functions

```cpp
template <ScalarObjective F>
Result minimize(F&& f, std::span<const double> x0, const Options& opts);

template <BatchObjective F>
Result minimize_batch(F&& f, std::span<const double> x0, const Options& opts);
```

`f` is called with a point and returns its objective value. The batch form
uses `f(xs, out)` for the initial simplex, `n + 1` independent points with
`x0` first; for every shrink, the vertices it moved; for every rung of the
poll ladder on a box, up to `2n` points; and for every restart's simplex,
`n` points. Every other evaluation is one point at a time. When a point inside
such a call meets `stop_value`, every point of the call has been evaluated
and counts. The method takes no nonlinear constraints, and there is no
overload for them.

Every entry point returns the best point evaluated with the reason the run
ended (`docs/api/result.md`). `Result::final_radius` is the radius of the
last simplex whose every vertex was evaluated, or `initial_step` while the
first one is incomplete, and `max_constraint_violation` is zero.

## Exceptions

- `std::invalid_argument` at entry: empty or non-finite `x0`, no stopping
  criterion, a negative or non-finite tolerance, a non-positive
  `initial_step`, bounds of the wrong length, a lower bound above its upper
  bound, `x0` outside the bounds, an `initial_step` for which
  `x0 +- initial_step` overflows, or `x0` or an initial vertex with a
  coordinate beyond `DBL_MAX / (n + 5)`, the range the method evaluates in
  (`docs/algorithms/nelder-mead.md`).
- `std::runtime_error` during the run: the objective returned a non-finite
  value, or the method would evaluate a point beyond that range (in
  practice, an objective unbounded below).
- Anything the objective throws propagates unchanged.

## Example

```cpp
#include <numbers>
#include <span>
#include "flop/nelder_mead.hpp"

double rosenbrock(std::span<const double> x) {
    const double a = 1.0 - x[0], b = x[1] - x[0] * x[0];
    return a * a + 100.0 * b * b;
}

int main() {
    flop::nelder_mead::Options opts;
    opts.stopping.xtol_rel = 1e-8;
    opts.stopping.max_evaluations = 2000;
    opts.initial_step = 0.5;
    const double lo[2] = {-2.0 * std::numbers::pi, -2.0 * std::numbers::pi};
    const double hi[2] = {2.0 * std::numbers::pi, 2.0 * std::numbers::pi};
    opts.bounds = flop::Bounds::box(lo, hi);
    const double x0[2] = {-1.2, 1.0};
    const flop::Result r = flop::nelder_mead::minimize(rosenbrock, x0, opts);
    return flop::converged(r.status) ? 0 : 1;
}
```
