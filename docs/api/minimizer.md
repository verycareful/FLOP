# Minimizer

The type-erased facade: one algorithm chosen by name at run time, the
objective held as a `std::function`. Compiled once in the library.

```cpp
#include "flop/minimizer.hpp"
// namespace flop
```

## Class

```cpp
class Minimizer {
public:
    using Objective = std::function<double(std::span<const double>)>;
    using BatchObjective =
        std::function<void(std::span<const std::span<const double>>, std::span<double>)>;
    using Constraints = std::function<void(std::span<const double>, std::span<double>)>;

    static Minimizer create(std::string_view name);
    static std::span<const std::string_view> names() noexcept;
    std::string_view name() const noexcept;

    Minimizer& set_final_trust_radius(double rhoend);

    Result minimize(const Objective& f, std::span<const double> x0, const Options& opts) const;
    Result minimize(const Objective& f, const Constraints& c, std::size_t n_constraints,
                    std::span<const double> x0, const Options& opts) const;
    Result minimize(const BatchObjective& f, std::span<const double> x0, const Options& opts) const;
    Result minimize(const BatchObjective& f, const Constraints& c, std::size_t n_constraints,
                    std::span<const double> x0, const Options& opts) const;
};
```

`create` accepts `"COBYLA"`. It is case-sensitive and throws
`std::invalid_argument` on any other name, naming the known ones. `names()`
lists them. `set_final_trust_radius` sets the option COBYLA reads beyond the
shared set and returns the minimizer for chaining.

A `Minimizer` is movable and not copyable.

## Exceptions

`create` throws `std::invalid_argument` on an unknown name. The `minimize`
overloads throw what the underlying algorithm throws (`docs/api/cobyla.md`).

## Example

```cpp
#include "flop/minimizer.hpp"

int main() {
    auto m = flop::Minimizer::create("COBYLA");
    flop::Options opts;
    opts.stopping.max_evaluations = 500;
    opts.stopping.xtol_rel = 1e-6;
    const double x0[3] = {1.0, 1.0, 1.0};
    const flop::Result r = m.minimize(
        [](std::span<const double> x) { return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]; },
        x0, opts);
    return r.f < 1e-8 ? 0 : 1;
}
```
