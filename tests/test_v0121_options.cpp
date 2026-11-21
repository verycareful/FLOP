// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: BOBYQA's own options and the evaluation callback.
//
// interpolation_points is m, from n + 2 to (n + 1)(n + 2) / 2 (Powell,
// DAMTP 2009/NA06, section 1), and 0 selects the report's 2n + 1. The first
// m evaluations are the starting points of section 2, so a cap of m ends a
// run after exactly them. shrink_initial_step lowers the first radius to
// min(initial_step, min_i (b_i - a_i) / 2) instead of refusing a narrow box
// (2.1). on_evaluation is called once per evaluation, in order, with the
// point and the value the objective returned.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <set>
#include <span>
#include <utility>
#include <vector>

#include "flop/flop.hpp"
#include "v0121_problems.hpp"

namespace {

using v0121::Vec;

struct Outcome {
    flop::Result r;
    v0121::Trace trace;
};

Outcome run(const Vec& x0, flop::bobyqa::Options o) {
    Outcome out;
    o.on_evaluation = v0121::recorder(out.trace);
    out.r = flop::bobyqa::minimize(v0111::sphere, x0, o);
    return out;
}

Outcome run_batch(const Vec& x0, flop::bobyqa::Options o) {
    Outcome out;
    o.on_evaluation = v0121::recorder(out.trace);
    auto fb = [](std::span<const std::span<const double>> xs, std::span<double> vals) {
        for (std::size_t i = 0; i < xs.size(); ++i) vals[i] = v0111::sphere(xs[i]);
    };
    out.r = flop::bobyqa::minimize_batch(fb, x0, o);
    return out;
}

bool same_trace(const v0121::Trace& a, const v0121::Trace& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].index != b[i].index || !v0121::same_bits(a[i].x, b[i].x) ||
            !v0121::same_bits(a[i].f, b[i].f))
            return false;
    return true;
}

}  // namespace

TEST(V0121Options, ACapOfMEndsTheRunAfterExactlyTheStartingPoints) {
    for (std::size_t n = 1; n <= 4; ++n) {
        const Vec x0(n, 0.5);
        for (std::size_t m = n + 2; m <= (n + 1) * (n + 2) / 2; ++m) {
            flop::bobyqa::Options o = v0121::options(0.25, m);
            o.interpolation_points = m;
            const Outcome a = run(x0, o);
            EXPECT_EQ(a.r.evaluations, m) << "n " << n << " m " << m;
            EXPECT_EQ(a.r.status, flop::Status::MaxEvaluationsReached);
            EXPECT_EQ(a.trace.size(), m);
            // One fewer ends it one point earlier.
            o.stopping.max_evaluations = m - 1;
            const Outcome b = run(x0, o);
            EXPECT_EQ(b.r.evaluations, m - 1) << "n " << n << " m " << m;
            EXPECT_EQ(b.r.status, flop::Status::MaxEvaluationsReached);
        }
    }
}

TEST(V0121Options, ZeroSelectsTwoNPlusOne) {
    for (std::size_t n = 1; n <= 4; ++n) {
        const Vec x0(n, 0.5);
        flop::bobyqa::Options zero = v0121::options_xtol(0.25, 1e-6, 500);
        flop::bobyqa::Options explicit_m = zero;
        explicit_m.interpolation_points = 2 * n + 1;
        const Outcome a = run(x0, zero);
        const Outcome b = run(x0, explicit_m);
        EXPECT_TRUE(same_trace(a.trace, b.trace)) << "n " << n;
        // A cap of 2n + 1 ends the default run after its starting points.
        flop::bobyqa::Options capped = v0121::options(0.25, 2 * n + 1);
        EXPECT_EQ(run(x0, capped).r.evaluations, 2 * n + 1);
    }
}

TEST(V0121Options, EveryLegalPointCountMinimisesAQuadratic) {
    // sphere is a quadratic with its minimum at sphere_centre; with the radius
    // taken down to 1e-8 every m reaches it to far better than 1e-4.
    for (std::size_t n = 1; n <= 4; ++n) {
        Vec c(n);
        for (std::size_t i = 0; i < n; ++i) c[i] = v0111::sphere_centre(i);
        const Vec x0(n, 0.5);
        for (std::size_t m = n + 2; m <= (n + 1) * (n + 2) / 2; ++m) {
            flop::bobyqa::Options o = v0121::options_xtol(0.25, 1e-8, 5000);
            o.interpolation_points = m;
            for (const bool batch : {false, true}) {
                const Outcome a = batch ? run_batch(x0, o) : run(x0, o);
                EXPECT_TRUE(flop::converged(a.r.status))
                    << "n " << n << " m " << m << " " << flop::to_string(a.r.status);
                EXPECT_LT(v0121::max_abs_diff(a.r.x, c), 1e-4) << "n " << n << " m " << m;
                EXPECT_LT(a.r.f, 1e-8) << "n " << n << " m " << m;
            }
        }
    }
}

TEST(V0121Options, ANarrowBoxShrinksTheFirstRadiusToHalfItsWidth) {
    // Box [0, 1] x [0, 4], x0 = (0.5, 2), initial_step 2: the first radius is
    // min(2, 1/2, 4/2) = 1/2. x0 is 1/2 from both bounds of the first
    // coordinate, so the starting points are x0 and x0 +- 1/2 along each
    // axis, every one exact.
    flop::bobyqa::Options o = v0121::options(2.0, 5);
    o.shrink_initial_step = true;
    o.bounds = flop::Bounds::box(Vec{0.0, 0.0}, Vec{1.0, 4.0});
    const Outcome a = run(Vec{0.5, 2.0}, o);
    ASSERT_EQ(a.trace.size(), 5u);
    std::set<std::pair<double, double>> got;
    for (const auto& p : a.trace) got.insert({p.x[0], p.x[1]});
    const std::set<std::pair<double, double>> want{
        {0.5, 2.0}, {1.0, 2.0}, {0.0, 2.0}, {0.5, 2.5}, {0.5, 1.5}};
    EXPECT_EQ(got, want);
    // The batch channel starts from the same five points.
    const Outcome b = run_batch(Vec{0.5, 2.0}, o);
    EXPECT_TRUE(same_trace(a.trace, b.trace));
}

TEST(V0121Options, TheNarrowestCoordinateSetsTheShrunkRadius) {
    // Widths 8 and 1: the radius is 1/2 whichever coordinate comes first.
    for (const bool swap : {false, true}) {
        flop::bobyqa::Options o = v0121::options(4.0, 200);
        o.stopping.xtol_rel = 1e-6;
        o.shrink_initial_step = true;
        const Vec lo = swap ? Vec{0.0, -4.0} : Vec{-4.0, 0.0};
        const Vec hi = swap ? Vec{1.0, 4.0} : Vec{4.0, 1.0};
        o.bounds = flop::Bounds::box(lo, hi);
        const Vec x0 = swap ? Vec{0.5, 0.0} : Vec{0.0, 0.5};
        const Outcome a = run(x0, o);
        const std::size_t narrow = swap ? 0 : 1;
        // x0 is the centre of the narrow side: the starting points on it are
        // exactly on its two bounds.
        std::set<double> seen;
        for (std::size_t i = 1; i < 5; ++i) seen.insert(a.trace[i].x[narrow]);
        EXPECT_TRUE(seen.contains(0.0)) << swap;
        EXPECT_TRUE(seen.contains(1.0)) << swap;
        for (const auto& p : a.trace)
            for (std::size_t i = 0; i < 2; ++i) {
                EXPECT_GE(p.x[i], lo[i]);
                EXPECT_LE(p.x[i], hi[i]);
            }
        EXPECT_LE(a.r.final_radius, 0.5);
    }
}

TEST(V0121Options, ShrinkingChangesNothingWhenTheBoxIsWideEnough) {
    flop::bobyqa::Options o = v0121::options_xtol(0.25, 1e-6, 500);
    o.bounds = flop::Bounds::box(Vec{-2.0, -2.0}, Vec{2.0, 2.0});
    const Outcome off = run(Vec{0.5, 0.5}, o);
    o.shrink_initial_step = true;
    const Outcome on = run(Vec{0.5, 0.5}, o);
    EXPECT_TRUE(same_trace(off.trace, on.trace));
    // A box exactly twice the step wide is wide enough, too.
    o.bounds = flop::Bounds::box(Vec{0.0, 0.0}, Vec{0.5, 0.5});
    o.shrink_initial_step = false;
    const Outcome exact_off = run(Vec{0.25, 0.25}, o);
    o.shrink_initial_step = true;
    const Outcome exact_on = run(Vec{0.25, 0.25}, o);
    EXPECT_TRUE(same_trace(exact_off.trace, exact_on.trace));
}

TEST(V0121Options, OneSidedCoordinatesDoNotShrinkTheStep) {
    flop::bobyqa::Options o = v0121::options(4.0, 5);
    o.shrink_initial_step = true;
    o.bounds = flop::Bounds::none(2);
    o.bounds->lower[0] = 0.0;
    o.bounds->upper[1] = 1.0;
    const Outcome a = run(Vec{0.5, 0.5}, o);
    // The step stays 4: some starting point is further than 1 from x0.
    double far = 0.0;
    for (const auto& p : a.trace)
        far = std::max({far, std::fabs(p.x[0] - 0.5), std::fabs(p.x[1] - 0.5)});
    EXPECT_GT(far, 1.0);
}

TEST(V0121Options, TheCallbackSeesEveryEvaluationInOrder) {
    const Vec x0{1.0, 1.0, 1.0};
    const flop::bobyqa::Options o = v0121::options_xtol(0.5, 1e-6, 500);
    for (const bool batch : {false, true}) {
        const Outcome a = batch ? run_batch(x0, o) : run(x0, o);
        ASSERT_EQ(a.trace.size(), a.r.evaluations) << batch;
        for (std::size_t i = 0; i < a.trace.size(); ++i) {
            EXPECT_EQ(a.trace[i].index, i) << batch;
            ASSERT_EQ(a.trace[i].x.size(), x0.size());
            // The value is the objective's own at the point reported.
            EXPECT_TRUE(v0121::same_bits(a.trace[i].f, v0111::sphere(a.trace[i].x))) << i;
        }
        // The first point is x0.
        EXPECT_TRUE(v0121::same_bits(a.trace[0].x, x0)) << batch;
        // The result is the best point evaluated: the first of the minima.
        std::size_t best = 0;
        for (std::size_t i = 1; i < a.trace.size(); ++i)
            if (a.trace[i].f < a.trace[best].f) best = i;
        EXPECT_TRUE(v0121::same_bits(a.r.f, a.trace[best].f)) << batch;
        EXPECT_TRUE(v0121::same_bits(a.r.x, a.trace[best].x)) << batch;
    }
}

TEST(V0121Options, AnEmptyCallbackCostsNothingAndChangesNothing) {
    const Vec x0{1.0, 1.0};
    flop::bobyqa::Options o = v0121::options_xtol(0.5, 1e-6, 500);
    const flop::Result a = flop::bobyqa::minimize(v0111::sphere, x0, o);
    const Outcome b = run(x0, o);
    EXPECT_TRUE(v0121::same_bits(a.x, b.r.x));
    EXPECT_TRUE(v0121::same_bits(a.f, b.r.f));
    EXPECT_EQ(a.evaluations, b.r.evaluations);
    EXPECT_EQ(a.status, b.r.status);
}
