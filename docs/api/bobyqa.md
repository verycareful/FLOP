# bobyqa

Powell's BOBYQA: derivative-free minimisation by a quadratic model,
optionally inside box bounds. The algorithm is described in
`docs/algorithms/bobyqa.md`.

```cpp
#include "flop/bobyqa.hpp"
// namespace flop::bobyqa
```

## Options

```cpp
struct Options : flop::Options {
    double final_trust_radius = 0.0;        // rhoend; 0 derives it from the x tolerances
    std::size_t interpolation_points = 0;   // m; 0 is 2n + 1
    bool shrink_initial_step = false;       // fit the first radius to a narrow box
    bool newuoa_weight = true;              // NEWUOA's weight in the choice of the point replaced
};
```

The shared fields are documented in `docs/api/options.md`. For this method
`initial_step` is the first trust-region radius, the x tolerances set the
final one, and the f tolerances act on the fall in `f` across a step that
lowers the best value. `interpolation_points` is `m`, the number of points
the quadratic model interpolates: `0` selects `2n + 1`, and any other value
must lie in `[n + 2, (n + 1)(n + 2) / 2]`. Every coordinate bounded on both
sides must have room for two first steps, `upper - lower >= 2 initial_step`,
unless `shrink_initial_step` is set, which lowers the first radius to
`min(initial_step, min_i (upper_i - lower_i) / 2)`.

## Functions

```cpp
template <ScalarObjective F>
Result minimize(F&& f, std::span<const double> x0, const Options& opts);

template <BatchObjective F>
Result minimize_batch(F&& f, std::span<const double> x0, const Options& opts);
```

`f` is called with a point and returns its objective value. The batch form
uses `f(xs, out)` for the starting points: the first `min(m, 2n + 1)` as
one call with `x0` first, then the other `m - 2n - 1`, which depend on the
first values, as a second. It also takes the new points of a RESCUE. Every
other evaluation is one point at a time. When a point inside such a call
meets `stop_value`, every point of the call has been evaluated and counts.
The method takes no nonlinear constraints, and there is no overload for
them.

Every point evaluated is inside the box. A coordinate the method steps onto
a bound takes the bound's own value, and `x0` is moved onto a bound, or to
`initial_step` inside it, when it is closer to the bound than that.

Every entry point returns the best point evaluated with the reason the run
ended (`docs/api/result.md`). `Result::final_radius` is the lower bound
`rho` on the trust-region radius at the end, and `max_constraint_violation`
is zero.

## Exceptions

- `std::invalid_argument` at entry: empty or non-finite `x0`, no stopping
  criterion, a negative or non-finite tolerance, a non-positive
  `initial_step`, bounds of the wrong length, a lower bound not below its
  upper bound, `x0` outside the bounds, an `initial_step` for which
  `x0 +- initial_step` overflows, `interpolation_points` outside its range,
  a negative or non-finite `final_trust_radius`, or a box narrower than
  `2 initial_step` without `shrink_initial_step`.
- `std::runtime_error` during the run: the objective returned a non-finite
  value, or the method would evaluate a point beyond the largest double (in
  practice, an objective unbounded below).
- Anything the objective throws propagates unchanged.

## Example

```cpp
#include <span>
#include "flop/bobyqa.hpp"

double rosenbrock(std::span<const double> x) {
    const double a = 1.0 - x[0], b = x[1] - x[0] * x[0];
    return a * a + 100.0 * b * b;
}

int main() {
    flop::bobyqa::Options opts;
    opts.stopping.xtol_abs = 1e-8;
    opts.stopping.max_evaluations = 2000;
    opts.initial_step = 0.5;
    const double lo[2] = {-2.0, -2.0};
    const double hi[2] = {2.0, 2.0};
    opts.bounds = flop::Bounds::box(lo, hi);
    const double x0[2] = {-1.2, 1.0};
    const flop::Result r = flop::bobyqa::minimize(rosenbrock, x0, opts);
    return flop::converged(r.status) ? 0 : 1;
}
```
