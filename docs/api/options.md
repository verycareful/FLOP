# Options, Stopping and Bounds

The settings every algorithm shares.

```cpp
#include "flop/options.hpp"   // brings flop/stopping.hpp and flop/bounds.hpp
// namespace flop
```

## Stopping

```cpp
struct Stopping {
    std::size_t max_evaluations = 0;   // 0 = no cap
    double xtol_rel = 0.0;             // rho stops at xtol_rel * initial_step
    double xtol_abs = 0.0;             // absolute floor on rho
    double ftol_rel = 0.0;             // relative to |f|
    double ftol_abs = 0.0;
    std::optional<double> stop_value;  // stop as soon as f <= stop_value
};
```

At least one criterion must be set; validation throws otherwise, because a
method with no stopping rule would run forever. `max_evaluations` is checked
before every evaluation and counts batch points singly. The x tolerances act
on the trust-region radius, the f tolerances on the fall in `f` across a
step that moves the base, and `stop_value` after every evaluation.

## Bounds

```cpp
struct Bounds {
    std::vector<std::optional<double>> lower;
    std::vector<std::optional<double>> upper;
    static Bounds box(std::span<const double> lo, std::span<const double> hi);
    static Bounds none(std::size_t n);
    std::size_t size() const noexcept;
};
```

One optional bound per coordinate in each direction. A missing bound is an
empty optional, never an infinity. `box` bounds every coordinate on both
sides; `none(n)` bounds nothing and can be edited entry by entry. Validation
requires both vectors to have the length of `x0`, every present bound to be
finite, `lower <= upper` where both are present, and `x0` inside the box.

## Options

```cpp
struct Options {
    Stopping stopping;
    std::optional<Bounds> bounds;
    double initial_step = 1.0;
    std::function<void(const Evaluation&)> on_evaluation;
};
```

`initial_step` is the first step along each coordinate (Powell's `rhobeg`).
The default fits a problem scaled around unity; there is no rule deriving it
from `x0`.

`on_evaluation` is called once per objective evaluation, in order, with an
`Evaluation` (`docs/api/result.md`) whose spans are valid only for the
duration of the call. Empty means off.

## Example

```cpp
flop::Options opts;
opts.stopping.max_evaluations = 200;
opts.stopping.xtol_rel = 1e-6;
opts.initial_step = 0.3;
const double lo[2] = {-6.3, -6.3}, hi[2] = {6.3, 6.3};
opts.bounds = flop::Bounds::box(lo, hi);
std::vector<double> history;
opts.on_evaluation = [&](const flop::Evaluation& e) { history.push_back(e.f); };
```
