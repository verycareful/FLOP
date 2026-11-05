// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// bench_bobyqa - the optimizer's own cost per evaluation
// =============================================================================
//
// The objective is a shifted sphere with unequal curvatures, nanoseconds to
// evaluate, so wall time divided by evaluations is the optimizer's own cost
// per evaluation: the trust-region step, the update of the model and of H,
// and the occasional shift of origin. With m = 2n + 1 an iteration is O(m^2)
// = O(n^2) by construction, so ns_per_eval should grow quadratically in n.
// The rows run to a stop at xtol_abs = 1e-8 from x0 = 0.5, initial_step
// 0.3, unbounded and inside a box of plus or minus 2 pi, so evals_per_run
// is the method's own count; capped is 1 when a run ended at the safety cap
// instead, which would make the row measure something else.
//
// The active box row puts the minimiser on a face: the lower bound 0.2 on
// every even coordinate binds, so the steps run along the bounds and the
// projections in TRSBOX and ALTMOV are exercised.
//
// The numbers to compare against, NLopt 2.7.1's BOBYQA (the Ubuntu 24.04
// package) on the same rows with xtol_abs 1e-8, measured beside this file's
// rows on 2026-11-05, x86-64, GCC 13.3, medians of five repetitions, in ns
// per evaluation with evaluations per run in parentheses. Unbounded: 376
// (26) at 2 parameters, 1825 (45) at 8, 5795 (61) at 16, 18286 (94) at 32,
// 44531 (161) at 64, 118467 (289) at 128. Inside the box that never binds:
// 306, 1846, 7218, 17829, 47491, 126886, with the same counts. Active box:
// 352 (24), 2051 (45), 7141 (84), 29271 (135), 81430 (259) at 2 to 64.

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <vector>

#include "flop/bobyqa.hpp"

namespace {

double sphere(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double d = x[i] - 0.1;
        s += (1.0 + static_cast<double>(i % 4)) * d * d;
    }
    return s;
}

enum class Box : std::uint8_t { None, Wide, Active };

void run(benchmark::State& state, Box box) {
    const auto n = static_cast<std::size_t>(state.range(0));
    std::vector<double> x0(n, 0.5);
    flop::bobyqa::Options opts;
    opts.stopping.max_evaluations = 200 * n;
    opts.stopping.xtol_abs = 1e-8;
    opts.initial_step = 0.3;
    if (box != Box::None) {
        std::vector<double> lo(n, -2.0 * std::numbers::pi), hi(n, 2.0 * std::numbers::pi);
        if (box == Box::Active)
            for (std::size_t i = 0; i < n; i += 2) lo[i] = 0.2;
        opts.bounds = flop::Bounds::box(lo, hi);
    }
    std::size_t evaluations = 0;
    bool capped = false;
    for (auto _ : state) {
        flop::Result r = flop::bobyqa::minimize(sphere, x0, opts);
        benchmark::DoNotOptimize(r);
        evaluations += r.evaluations;
        capped = capped || r.status == flop::Status::MaxEvaluationsReached;
    }
    // items_per_second is evaluations per second; ns_per_eval is its
    // reciprocal.
    state.SetItemsProcessed(static_cast<int64_t>(evaluations));
    state.counters["evals_per_run"] =
        static_cast<double>(evaluations) / static_cast<double>(state.iterations());
    state.counters["ns_per_eval"] = benchmark::Counter(
        static_cast<double>(evaluations), benchmark::Counter::kIsRate | benchmark::Counter::kInvert,
        benchmark::Counter::kIs1000);
    state.counters["capped"] = capped ? 1.0 : 0.0;
}

void BM_Bobyqa_Unbounded(benchmark::State& state) {
    run(state, Box::None);
}
void BM_Bobyqa_Bounded(benchmark::State& state) {
    run(state, Box::Wide);
}
void BM_Bobyqa_ActiveBox(benchmark::State& state) {
    run(state, Box::Active);
}

}  // namespace

// Google Benchmark registers through static objects whose constructors can
// throw, which is the library's design and not this file's to change.
// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
BENCHMARK(BM_Bobyqa_Unbounded)->Arg(2)->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Arg(128);
// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
BENCHMARK(BM_Bobyqa_Bounded)->Arg(2)->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Arg(128);
// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
BENCHMARK(BM_Bobyqa_ActiveBox)->Arg(2)->Arg(8)->Arg(16)->Arg(32)->Arg(64);

BENCHMARK_MAIN();
