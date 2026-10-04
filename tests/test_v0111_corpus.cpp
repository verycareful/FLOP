// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.1.1: does Nelder-Mead minimise.
//
// The method has no convergence theorem beyond one dimension (Lagarias et
// al.), so these are measurements against the literature, not proofs. The
// More, Garbow and Hillstrom problems are the ones of Gao and Han's Table 2
// with a closed-form minimiser, run from More et al.'s starting points with
// Gao and Han's initial simplex ((3.1): x0 + 0.05 x0_i e_i, for which one
// initial_step of 0.05 max(1, |x0|) stands in) and a tolerance six orders
// below theirs ((4.2): TolX = 1e-4). Each must reach at least the final f
// Table 2 reports for the same coefficient set, and come within the paper's
// TolX of the minimiser. The sphere runs to 128 dimensions; the standard set
// only where the documentation says it works, below about ten. And the two
// documented examples do what their text says.
//
// Distances are held to ten tolerances. Nothing bounds the distance from
// the best vertex to the minimiser when the simplex is small; ten is the bar
// for stagnation, which leaves a stalled run orders of magnitude further
// away (Gao and Han's Table 1: the standard method on x'x at n = 40 stops at
// f = 2.45e-4 with TolX = 1e-4, a hundred tolerances out).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <string>
#include <vector>

#include "flop/flop.hpp"
#include "v0111_problems.hpp"

namespace {

using Vec = std::vector<double>;

constexpr double kStagnation = 10.0;

class V0111CorpusMgh : public ::testing::TestWithParam<v0111::Mgh> {};

}  // namespace

TEST_P(V0111CorpusMgh, ReachesTheMinimiserAtLeastAsWellAsGaoAndHanTable2) {
    const v0111::Mgh& p = GetParam();
    double scale = 1.0;
    for (const double v : p.x0) scale = std::max(scale, std::fabs(v));
    const double tol_x = 1e-4;  // Gao and Han (4.2)
    for (const bool adaptive : {true, false}) {
        flop::nelder_mead::Options o = v0111::options(0.05 * scale, 200000, adaptive);
        o.stopping.xtol_abs = 1e-10;
        const flop::Result r = flop::nelder_mead::minimize(p.f, p.x0, o);
        const char* set = adaptive ? "ANMS" : "SNMS";
        EXPECT_EQ(r.status, flop::Status::XtolReached) << set;
        EXPECT_LE(r.f, adaptive ? p.table2_anms : p.table2_snms) << set;
        Vec x = r.x;
        if (p.permutation_invariant) std::ranges::sort(x);
        EXPECT_LE(v0111::max_abs_diff(x, p.x_opt), tol_x) << set;
    }
}

INSTANTIATE_TEST_SUITE_P(Table2, V0111CorpusMgh, ::testing::ValuesIn(v0111::mgh_problems()),
                         [](const ::testing::TestParamInfo<v0111::Mgh>& tpi) {
                             std::string s = tpi.param.name;
                             for (char& ch : s)
                                 if (ch == ' ') ch = '_';
                             return s;
                         });

TEST(V0111Corpus, TheSphereConvergesInEveryDimension) {
    for (const std::size_t n : {std::size_t{2}, std::size_t{8}, std::size_t{16}, std::size_t{32},
                                std::size_t{64}, std::size_t{128}}) {
        for (const bool adaptive : {true, false}) {
            if (!adaptive && n > 8) continue;
            flop::nelder_mead::Options o = v0111::options(1.0, 2000000, adaptive);
            o.stopping.xtol_abs = 1e-8;
            const flop::Result r = flop::nelder_mead::minimize(v0111::sphere, Vec(n, 0.0), o);
            EXPECT_EQ(r.status, flop::Status::XtolReached) << n;
            Vec c(n);
            for (std::size_t i = 0; i < n; ++i) c[i] = v0111::sphere_centre(i);
            EXPECT_LE(v0111::max_abs_diff(r.x, c), kStagnation * o.stopping.xtol_abs)
                << "n " << n << (adaptive ? " adaptive" : " standard");
        }
    }
}

TEST(V0111Corpus, TheHeaderExampleDoesWhatItsCommentSays) {
    // flop/nelder_mead.hpp: Rosenbrock from (-1.2, 1), xtol_rel 1e-8, cap
    // 2000, initial_step 0.5; "r.x near (1, 1), flop::converged(r.status)
    // true".
    auto rosenbrock = [](std::span<const double> x) {
        const double a = 1.0 - x[0], b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    };
    flop::nelder_mead::Options opts;
    opts.stopping.xtol_rel = 1e-8;
    opts.stopping.max_evaluations = 2000;
    opts.initial_step = 0.5;
    const double x0[2] = {-1.2, 1.0};
    const flop::Result r = flop::nelder_mead::minimize(rosenbrock, x0, opts);
    EXPECT_TRUE(flop::converged(r.status));
    const double tol = opts.stopping.xtol_rel * opts.initial_step;
    EXPECT_LE(v0111::max_abs_diff(r.x, Vec{1.0, 1.0}), kStagnation * tol);
}

TEST(V0111Corpus, TheApiPageExampleReturnsZero) {
    // docs/api/nelder_mead.md: the same problem inside [-2 pi, 2 pi]^2;
    // main returns 0 when the run converged.
    auto rosenbrock = [](std::span<const double> x) {
        const double a = 1.0 - x[0], b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    };
    flop::nelder_mead::Options opts;
    opts.stopping.xtol_rel = 1e-8;
    opts.stopping.max_evaluations = 2000;
    opts.initial_step = 0.5;
    const double lo[2] = {-2.0 * std::numbers::pi, -2.0 * std::numbers::pi};
    const double hi[2] = {2.0 * std::numbers::pi, 2.0 * std::numbers::pi};
    opts.bounds = flop::Bounds::box(lo, hi);
    const double x0[2] = {-1.2, 1.0};
    const flop::Result r = flop::nelder_mead::minimize(rosenbrock, x0, opts);
    EXPECT_EQ(flop::converged(r.status) ? 0 : 1, 0);
}
