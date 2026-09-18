# Result, Status and Evaluation

What a run returns, and what the evaluation trace sees.

```cpp
#include "flop/result.hpp"   // brings flop/status.hpp
// namespace flop
```

## Status

```cpp
enum class Status {
    XtolReached,
    FtolReached,
    StopValueReached,
    MaxEvaluationsReached,
    RoundoffLimited
};
constexpr bool converged(Status s) noexcept;
constexpr std::string_view to_string(Status s) noexcept;
```

`converged` is true for the first three. Reaching the evaluation cap is an
outcome, reported as itself, and `converged` answers false for it. There is
no success flag.

## Result

```cpp
struct Result {
    std::vector<double> x;
    double f;
    std::size_t evaluations;
    Status status;
    double final_trust_radius;
    double max_constraint_violation;
};
```

`x` is the best point seen, feasible when one was found, and `f` its value.
`evaluations` counts objective calls, batch points singly.
`max_constraint_violation` is `max(0, -min_i c_i(x))`, zero without
constraints.

## Evaluation

```cpp
struct Evaluation {
    std::size_t index;
    std::span<const double> x;
    double f;
    std::span<const double> constraints;
};
```

Handed to `Options::on_evaluation`. `index` starts at zero. The spans point
into the optimizer's working storage and are valid only during the call.

## Example

```cpp
const flop::Result r = flop::cobyla::minimize(f, x0, opts);
if (!flop::converged(r.status)) {
    std::cerr << "stopped: " << flop::to_string(r.status)
              << " after " << r.evaluations << " evaluations\n";
}
```
