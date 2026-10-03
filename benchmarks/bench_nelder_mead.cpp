// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// bench_nelder_mead - the optimizer's own cost per evaluation
// =============================================================================
//
// The objective is a sphere, nanoseconds to evaluate, so wall time divided by
// evaluations is the optimizer's own cost per evaluation: the centroid, the
// trial points, the ranking and the radius. Reported at the parameter counts
// of bench_cobyla, unbounded and inside a box of plus or minus 2 pi, for a
// fixed evaluation budget of 20 n, with the dimension-adaptive coefficients
// (the default) and with the standard ones. A nonshrink iteration is O(n) by
// construction, so ns_per_eval should grow linearly in n, not quadratically.

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <vector>

#include "flop/nelder_mead.hpp"

namespace {

double sphere(std::span<const double> x) {
    double s = 0.0;
    for (double v : x) s += (v - 0.1) * (v - 0.1);
    return s;
}

void run(benchmark::State& state, bool bounded, bool adaptive) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const std::size_t budget = 20 * n;
    std::vector<double> x0(n, 0.5);
    flop::nelder_mead::Options opts;
    opts.stopping.max_evaluations = budget;
    opts.stopping.xtol_rel = 1e-12;
    opts.initial_step = 0.3;
    opts.adaptive_coefficients = adaptive;
    if (bounded) {
        std::vector<double> lo(n, -2.0 * std::numbers::pi), hi(n, 2.0 * std::numbers::pi);
        opts.bounds = flop::Bounds::box(lo, hi);
    }
    std::size_t evaluations = 0;
    for (auto _ : state) {
        flop::Result r = flop::nelder_mead::minimize(sphere, x0, opts);
        benchmark::DoNotOptimize(r);
        evaluations += r.evaluations;
    }
    // items_per_second is evaluations per second; ns_per_eval is its
    // reciprocal.
    state.SetItemsProcessed(static_cast<int64_t>(evaluations));
    state.counters["evals_per_run"] =
        static_cast<double>(evaluations) / static_cast<double>(state.iterations());
    state.counters["ns_per_eval"] = benchmark::Counter(
        static_cast<double>(evaluations), benchmark::Counter::kIsRate | benchmark::Counter::kInvert,
        benchmark::Counter::kIs1000);
}

void BM_NelderMead_Unbounded(benchmark::State& state) {
    run(state, false, true);
}
void BM_NelderMead_Bounded(benchmark::State& state) {
    run(state, true, true);
}
void BM_NelderMead_Standard(benchmark::State& state) {
    run(state, false, false);
}

}  // namespace

// Google Benchmark registers through static objects whose constructors can
// throw, which is the library's design and not this file's to change.
// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
BENCHMARK(BM_NelderMead_Unbounded)
    ->Arg(2)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(80)
    ->Arg(96)
    ->Arg(128);
// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
BENCHMARK(BM_NelderMead_Bounded)
    ->Arg(2)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(80)
    ->Arg(96)
    ->Arg(128);
// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
BENCHMARK(BM_NelderMead_Standard)->Arg(2)->Arg(16)->Arg(64)->Arg(128);

BENCHMARK_MAIN();
