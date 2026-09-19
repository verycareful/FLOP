# FLOP
<!-- Language and build -->
[![C++23](https://img.shields.io/badge/C%2B%2B-23-00599C?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/23)
[![CMake](https://img.shields.io/badge/CMake-3.25%2B-064F8C?logo=cmake&logoColor=white)](https://cmake.org/)
<!-- Test and benchmark -->
[![GoogleTest](https://img.shields.io/badge/GoogleTest-1.14.0-4285F4?logo=google&logoColor=white)](https://github.com/google/googletest)
[![Google Benchmark](https://img.shields.io/badge/Google%20Benchmark-1.8.3-4285F4)](https://github.com/google/benchmark) <!-- no shields.io logo for Google Benchmark; swap one in if it appears -->
<!-- CI -->
[![CI](https://github.com/verycareful/FLOP/actions/workflows/ci.yml/badge.svg)](https://github.com/verycareful/FLOP/actions/workflows/ci.yml)
<!-- Release -->
[![Version](https://img.shields.io/badge/version-0.1.0.2-blue)](https://github.com/verycareful/FLOP/blob/main/CHANGELOG.md)
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL--2.0-brightgreen)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Early%20Development-orange)](https://github.com/verycareful/FLOP)

Fast Library of Optimization Procedures: derivative-free optimizers for
objectives that are expensive to evaluate, written in C++23 from the
published algorithms, with no dependency at run time.

The first algorithm is Powell's COBYLA, with inequality constraints, box
bounds, a batch channel for the independent points of its initial simplex,
and a trace of every evaluation. Nelder-Mead, BOBYQA, SPSA and gradient and
population methods follow on the same interface.

## Design in one paragraph

An algorithm is a class template on the objective, so an evaluation inlines
into the optimizer loop; concepts (`ScalarObjective`, `BatchObjective`,
`ConstraintFunction`) say what an objective must offer, and an objective that
lacks a capability fails to compile. One type-erased facade,
`flop::Minimizer::create("COBYLA")`, serves the caller who chooses at run
time. A run returns the best point with a `Status` that names why it
stopped; reaching the evaluation cap is a status, not a success, and there
is no flag that says otherwise. Every guard survives `-ffast-math`: the
suite is built under the strict and the fast-math models in one build and
both must pass.

## Quick start

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

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/tests/flop_tests_strict
./build/tests/flop_tests_fast
```

Options: `FLOP_BUILD_TESTS` (on when FLOP is the top-level project),
`FLOP_BUILD_BENCHMARKS` (off), `FLOP_FAST_MATH` (compile the library's own
sources with `-ffast-math`; off), `FLOP_MARCH_NATIVE` (off).

To consume FLOP from another CMake project:

```cmake
include(FetchContent)
FetchContent_Declare(flop
    GIT_REPOSITORY https://github.com/verycareful/FLOP.git
    GIT_TAG        <a tag>)
FetchContent_MakeAvailable(flop)
target_link_libraries(your_target PRIVATE flop::flop)
```

## Documentation

- `docs/Architecture.md`: the design, and how an algorithm is added.
- `docs/api/`: one page per public header.
- `docs/algorithms/cobyla.md`: the method as implemented, with every
  deviation from the paper named.
- `CONTRIBUTING.md`: how to contribute, and the provenance rule.

## Requirements

A C++23 compiler (GCC 13 or later, Clang 18 or later), CMake 3.25 or later.
GoogleTest and Google Benchmark are fetched at configure time when their
targets are enabled.

## License

Copyright (c) 2026 Sricharan Suresh (github.com/verycareful).

FLOP is licensed under the Mozilla Public License 2.0. You may use it in any
program under any licence, including a proprietary one, and distribute the
result under terms of your choice. What the licence requires: changes to
FLOP's own files are published under the MPL-2.0, and anyone who receives a
binary containing FLOP is told that FLOP's source is available and where
(section 3.2 of the licence). The full text is in `LICENSE`.
