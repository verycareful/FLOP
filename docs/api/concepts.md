# Concepts

The shapes an objective, a batch objective and a constraint function must
have. Every algorithm's template entry points are constrained by them.

```cpp
#include "flop/concepts.hpp"
// namespace flop
```

## ScalarObjective

```cpp
template <class F>
concept ScalarObjective = /* f(std::span<const double>) -> double */;
```

A callable taking the point as a span and returning something convertible to
`double`. Lambdas, function pointers and function objects all qualify.

## BatchObjective

```cpp
template <class F>
concept BatchObjective =
    /* f(std::span<const std::span<const double>> xs, std::span<double> out) */;
```

Evaluates every point of `xs` and writes `out[i]` for each, in order;
`xs.size() == out.size()` always. FLOP never spawns a thread: how the batch
is evaluated is the caller's decision. An algorithm uses this channel where
it has independent points to evaluate and the scalar channel elsewhere.

## ConstraintFunction

```cpp
template <class C>
concept ConstraintFunction = /* c(std::span<const double> x, std::span<double> out) */;
```

Writes `out[i] = c_i(x)` for every constraint; feasible when every value is
at least zero. `out.size()` is the count the caller declared at the entry
point.

## Example

```cpp
auto f = [](std::span<const double> x) { return x[0] * x[0] + x[1]; };
auto c = [](std::span<const double> x, std::span<double> out) {
    out[0] = x[1];               // x[1] >= 0
    out[1] = 1.0 - x[0] * x[0];  // |x[0]| <= 1
};
static_assert(flop::ScalarObjective<decltype(f)>);
static_assert(flop::ConstraintFunction<decltype(c)>);
```
