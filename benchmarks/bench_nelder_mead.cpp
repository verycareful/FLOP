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
//
// The active box row is different in kind: it runs to a stop rather than to
// the budget, because the poll ladder runs only before a stop. The lower
// bound 0.2 on every even coordinate puts the sphere's minimiser on a face,
// so the box acts on the run: the face test fixes those coordinates on the
// bound, and every stop verdict is preceded by a ladder. evals_per_run
// carries both costs (up to 2n points per rung, about
// log2(initial_step / xtol_abs) rungs per ladder) on top of the method's own.
// capped is 1 when a run ended at the safety cap instead of a stop, which
// would make the row measure something else.
//
// The numbers to compare against, NLopt 2.7.1's NELDERMEAD on the same rows,
// same host (x86-64, GCC 16.2.1, NLopt built at -O3 -march=native), on
// 2026-11-04, in ns per evaluation: unbounded 29 at 2 parameters, 207 at
// 16, 830 at 64, 2727 at 128; inside the box that never binds 30, 169, 827,
// 2718. On the active box row NLopt needs 13285 evaluations at 8 parameters
// and reaches the cap at 16 and 32, where this row stops in 710, 1784 and
// 4568.

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

// Runs to a stop on the face lo_i = 0.2 of the even coordinates. The cap is
// a safety net, far above what a stop takes.
void run_active_box(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    std::vector<double> x0(n, 0.5);
    flop::nelder_mead::Options opts;
    opts.stopping.max_evaluations = 2000 * n;
    opts.stopping.xtol_abs = 1e-8;
    opts.initial_step = 0.3;
    std::vector<double> lo(n, -2.0 * std::numbers::pi), hi(n, 2.0 * std::numbers::pi);
    for (std::size_t i = 0; i < n; i += 2) lo[i] = 0.2;
    opts.bounds = flop::Bounds::box(lo, hi);
    std::size_t evaluations = 0;
    bool capped = false;
    for (auto _ : state) {
        flop::Result r = flop::nelder_mead::minimize(sphere, x0, opts);
        benchmark::DoNotOptimize(r);
        evaluations += r.evaluations;
        capped = capped || r.status == flop::Status::MaxEvaluationsReached;
    }
    state.SetItemsProcessed(static_cast<int64_t>(evaluations));
    state.counters["evals_per_run"] =
        static_cast<double>(evaluations) / static_cast<double>(state.iterations());
    state.counters["ns_per_eval"] = benchmark::Counter(
        static_cast<double>(evaluations), benchmark::Counter::kIsRate | benchmark::Counter::kInvert,
        benchmark::Counter::kIs1000);
    state.counters["capped"] = capped ? 1.0 : 0.0;
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
void BM_NelderMead_ActiveBox(benchmark::State& state) {
    run_active_box(state);
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
// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
BENCHMARK(BM_NelderMead_ActiveBox)->Arg(2)->Arg(8)->Arg(16)->Arg(32);

BENCHMARK_MAIN();
