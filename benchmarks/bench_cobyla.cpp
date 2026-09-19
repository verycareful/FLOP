// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// bench_cobyla - the optimizer's own cost per evaluation
// =============================================================================
//
// The objective is a sphere, nanoseconds to evaluate, so wall time divided by
// evaluations is the optimizer's per-iteration cost: the simplex algebra and
// the trust-region step. Reported at the parameter counts the lindblad
// measurements used, unbounded and inside a box of plus or minus 2 pi, for a
// fixed evaluation budget. The numbers to compare against, NLopt 2.7.1's
// COBYLA on the same host on 2026-09-14: about 3 microseconds per evaluation
// at 16 parameters, 0.03 to 0.4 ms at 96 unbounded, 0.45 to 0.73 ms at 80
// with bounds. The constrained row adds one linear constraint that is active
// at the optimum, so the active-set subproblem runs on every iteration;
// NLopt on that row, same host, 2026-09-19: 2.2 microseconds per evaluation
// at 16 parameters and 0.25 ms at 80.

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <vector>

#include "flop/cobyla.hpp"

namespace {

double sphere(std::span<const double> x) {
    double s = 0.0;
    for (double v : x) s += (v - 0.1) * (v - 0.1);
    return s;
}

// One linear constraint through the sphere's centre, active at the
// optimum, so every iteration takes the constrained step: the row measures
// the active-set subproblem, which the unconstrained rows never enter.
void plane(std::span<const double> x, std::span<double> out) {
    double s = 0.0;
    for (double v : x) s += v;
    out[0] = 0.1 * static_cast<double>(x.size()) - s;
}

enum class Kind : std::uint8_t { Unbounded, Bounded, Constrained };

void run(benchmark::State& state, Kind kind) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const std::size_t budget = 20 * n;
    std::vector<double> x0(n, 0.5);
    flop::cobyla::Options opts;
    opts.stopping.max_evaluations = budget;
    opts.stopping.xtol_rel = 1e-12;
    opts.initial_step = 0.3;
    if (kind == Kind::Bounded) {
        std::vector<double> lo(n, -2.0 * std::numbers::pi), hi(n, 2.0 * std::numbers::pi);
        opts.bounds = flop::Bounds::box(lo, hi);
    }
    std::size_t evaluations = 0;
    for (auto _ : state) {
        flop::Result r = kind == Kind::Constrained
                             ? flop::cobyla::minimize(sphere, plane, 1, x0, opts)
                             : flop::cobyla::minimize(sphere, x0, opts);
        benchmark::DoNotOptimize(r);
        evaluations += r.evaluations;
    }
    // items_per_second is evaluations per second; ns_per_eval is its
    // reciprocal, the number to compare with NLopt's.
    state.SetItemsProcessed(static_cast<int64_t>(evaluations));
    state.counters["evals_per_run"] =
        static_cast<double>(evaluations) / static_cast<double>(state.iterations());
    state.counters["ns_per_eval"] = benchmark::Counter(
        static_cast<double>(evaluations), benchmark::Counter::kIsRate | benchmark::Counter::kInvert,
        benchmark::Counter::kIs1000);
}

void BM_Cobyla_Unbounded(benchmark::State& state) {
    run(state, Kind::Unbounded);
}
void BM_Cobyla_Bounded(benchmark::State& state) {
    run(state, Kind::Bounded);
}
void BM_Cobyla_Constrained(benchmark::State& state) {
    run(state, Kind::Constrained);
}

}  // namespace

// Google Benchmark registers through static objects whose constructors can
// throw, which is the library's design and not this file's to change.
// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
BENCHMARK(BM_Cobyla_Unbounded)
    ->Arg(2)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(80)
    ->Arg(96)
    ->Arg(128);
// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
BENCHMARK(BM_Cobyla_Bounded)->Arg(2)->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Arg(80)->Arg(96)->Arg(128);
// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
BENCHMARK(BM_Cobyla_Constrained)->Arg(2)->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Arg(80);

BENCHMARK_MAIN();
