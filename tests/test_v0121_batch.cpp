// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: BOBYQA's batch channel.
//
// The method has independent points in two places: the starting points, the
// first min(m, 2n + 1) of them as one call with x0 first and the remaining
// m - 2n - 1, which depend on the first values, as a second; and the new
// points of a RESCUE, one call per run of consecutive slots
// (docs/algorithms/bobyqa.md, "Deviations"). Every other evaluation is one
// point. The contract is on the shape of the calls, and on the points: the
// batch run evaluates what the scalar run evaluates, in the same order, to the
// bit, whatever the cap and whatever the box.
//
// A stop value met inside a call is the one place the channels differ, by
// design: the call has already evaluated every point in it, so all of them are
// counted, traced and eligible as the result.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

#include "flop/flop.hpp"
#include "v0121_problems.hpp"

namespace {

using v0121::Trace;
using v0121::Vec;

double quadratic(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double d = x[i] - 0.3 * static_cast<double>(i + 1);
        s += d * d;
    }
    return s;
}

double rosenbrock(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        const double a = 1.0 - x[i], b = x[i + 1] - x[i] * x[i];
        s += a * a + 100.0 * b * b;
    }
    return s;
}

struct Outcome {
    flop::Result r;
    Trace trace;
    std::vector<std::size_t> sizes;  // the size of every batch call
    std::vector<std::vector<double>> first_points;
};

Outcome run_scalar(double (*f)(std::span<const double>), const Vec& x0, flop::bobyqa::Options o) {
    Outcome out;
    o.on_evaluation = v0121::recorder(out.trace);
    out.r = flop::bobyqa::minimize(f, x0, o);
    return out;
}

Outcome run_batch(double (*f)(std::span<const double>), const Vec& x0, flop::bobyqa::Options o) {
    Outcome out;
    o.on_evaluation = v0121::recorder(out.trace);
    auto fb = [&](std::span<const std::span<const double>> xs, std::span<double> vals) {
        out.sizes.push_back(xs.size());
        out.first_points.emplace_back(xs[0].begin(), xs[0].end());
        for (std::size_t i = 0; i < xs.size(); ++i) vals[i] = f(xs[i]);
    };
    out.r = flop::bobyqa::minimize_batch(fb, x0, o);
    return out;
}

bool same_trace(const Trace& a, const Trace& b, std::size_t* at = nullptr) {
    const std::size_t k = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < k; ++i) {
        if (!v0121::same_bits(a[i].x, b[i].x) || !v0121::same_bits(a[i].f, b[i].f)) {
            if (at) *at = i;
            return false;
        }
    }
    if (a.size() != b.size()) {
        if (at) *at = k;
        return false;
    }
    return true;
}

std::size_t max_m(std::size_t n) {
    return (n + 1) * (n + 2) / 2;
}

}  // namespace

TEST(V0121Batch, TheFirstCallIsTheFirstMinOfMAndTwoNPlusOneStartingPointsWithX0First) {
    for (std::size_t n = 1; n <= 4; ++n) {
        for (std::size_t m = n + 2; m <= max_m(n); ++m) {
            flop::bobyqa::Options o = v0121::options(0.25, m);
            o.interpolation_points = m;
            const Vec x0(n, 0.5);
            const Outcome t = run_batch(quadratic, x0, o);
            ASSERT_EQ(t.trace.size(), m) << "n " << n << " m " << m;
            const std::size_t first = std::min(m, 2 * n + 1);
            ASSERT_FALSE(t.sizes.empty());
            EXPECT_EQ(t.sizes[0], first) << "n " << n << " m " << m;
            EXPECT_TRUE(v0121::same_bits(t.first_points[0], x0)) << "x0 is first, n " << n;
            if (m > 2 * n + 1) {
                ASSERT_EQ(t.sizes.size(), 2u) << "n " << n << " m " << m;
                EXPECT_EQ(t.sizes[1], m - 2 * n - 1);
            } else {
                EXPECT_EQ(t.sizes.size(), 1u);
            }
        }
    }
}

TEST(V0121Batch, EveryOtherEvaluationOfARunThatNeedsNoRescueGoesAlone) {
    // The calls after the starting points all have one point, and the sizes
    // add up to the evaluations.
    for (const std::size_t n : {std::size_t{2}, std::size_t{4}}) {
        flop::bobyqa::Options o = v0121::options_xtol(0.5, 1e-6, 2000);
        const Outcome t = run_batch(rosenbrock, Vec(n, 0.0), o);
        const std::size_t first_calls = 1;
        ASSERT_GT(t.sizes.size(), first_calls);
        EXPECT_EQ(t.sizes[0], 2 * n + 1);
        for (std::size_t i = first_calls; i < t.sizes.size(); ++i)
            EXPECT_EQ(t.sizes[i], 1u) << "call " << i << ", n " << n;
        EXPECT_EQ(std::accumulate(t.sizes.begin(), t.sizes.end(), std::size_t{0}), t.r.evaluations);
    }
}

TEST(V0121Batch, ARescueSendsItsNewPointsInCallsOfConsecutiveSlots) {
    // With rho_end = 1e-12 RESCUE is called, and the calls after the starting
    // points are of one point or, where it replaced consecutive slots, more:
    // never more than m - 1, and every point goes out exactly once. A call of
    // more than one point is rare (RESCUE keeps most old points); one was seen
    // at n = 8, m = 45, which is among the runs here, but the test does not
    // require it, since whether RESCUE replaces consecutive slots depends on the
    // rounding of the arithmetic.
    for (const std::size_t n : {std::size_t{2}, std::size_t{3}, std::size_t{5}, std::size_t{8}}) {
        for (const std::size_t m : {2 * n + 1, n + 3, max_m(n)}) {
            flop::bobyqa::Options o = v0121::options(0.5, 6000);
            o.stopping.xtol_abs = 1e-12;
            o.interpolation_points = m;
            const Outcome t = run_batch(quadratic, Vec(n, 0.1), o);
            const std::size_t start_calls = m > 2 * n + 1 ? 2 : 1;
            std::size_t total = 0;
            for (std::size_t i = 0; i < t.sizes.size(); ++i) {
                total += t.sizes[i];
                if (i < start_calls) continue;
                EXPECT_LE(t.sizes[i], m - 1) << "call " << i << " n " << n << " m " << m;
            }
            EXPECT_EQ(total, t.r.evaluations) << "n " << n << " m " << m;
            EXPECT_EQ(t.trace.size(), t.r.evaluations);
        }
    }
}

TEST(V0121Batch, TheBatchAndScalarChannelsRunTheSameTrajectoryToTheBit) {
    struct Problem {
        double (*f)(std::span<const double>);
        Vec x0;
        std::size_t m;
        double xtol;
        bool boxed;
    };
    const Problem problems[] = {
        {.f = quadratic, .x0 = {0.1, 0.1}, .m = 0, .xtol = 1e-9, .boxed = false},
        {.f = rosenbrock, .x0 = {-1.2, 1.0}, .m = 0, .xtol = 1e-8, .boxed = false},
        {.f = rosenbrock, .x0 = {-1.2, 1.0, 1.0}, .m = 5, .xtol = 1e-7, .boxed = false},
        {.f = rosenbrock, .x0 = {-1.2, 1.0, 1.0}, .m = 10, .xtol = 1e-7, .boxed = false},
        {.f = quadratic, .x0 = {0.1, 0.1, 0.1}, .m = 0, .xtol = 1e-12, .boxed = false},
        {.f = quadratic, .x0 = {0.1, 0.1, 0.1, 0.1}, .m = 7, .xtol = 1e-12, .boxed = false},
        {.f = quadratic, .x0 = {0.0, 0.0}, .m = 0, .xtol = 1e-9, .boxed = true},
        {.f = rosenbrock, .x0 = {-0.5, 0.2}, .m = 6, .xtol = 1e-8, .boxed = true},
    };
    for (const Problem& p : problems) {
        flop::bobyqa::Options o = v0121::options(0.4, 4000);
        o.stopping.xtol_abs = p.xtol;
        o.interpolation_points = p.m;
        if (p.boxed) o.bounds = flop::Bounds::box(Vec(p.x0.size(), -1.0), Vec(p.x0.size(), 0.6));
        const Outcome s = run_scalar(p.f, p.x0, o);
        const Outcome b = run_batch(p.f, p.x0, o);
        std::size_t at = 0;
        EXPECT_TRUE(same_trace(s.trace, b.trace, &at)) << "first difference at point " << at;
        EXPECT_TRUE(v0121::same_bits(s.r.x, b.r.x));
        EXPECT_TRUE(v0121::same_bits(s.r.f, b.r.f));
        EXPECT_TRUE(v0121::same_bits(s.r.final_radius, b.r.final_radius));
        EXPECT_EQ(s.r.evaluations, b.r.evaluations);
        EXPECT_EQ(s.r.status, b.r.status);
    }
}

TEST(V0121Batch, ACapTooSmallForACallSendsThePointsOneAtATimeAndKeepsTheTrajectory) {
    // A cap below the size of a call: the call is not made whole, and the
    // run is the scalar run's prefix.
    const std::size_t n = 3;
    const Vec x0(n, 0.1);
    for (const std::size_t m : {2 * n + 1, std::size_t{10}}) {
        flop::bobyqa::Options base = v0121::options(0.5, 1000);
        base.interpolation_points = m;
        base.stopping.xtol_abs = 1e-8;
        const Outcome whole = run_scalar(quadratic, x0, base);
        for (std::size_t cap = 1; cap <= m + 3; ++cap) {
            flop::bobyqa::Options o = base;
            o.stopping.max_evaluations = cap;
            const Outcome b = run_batch(quadratic, x0, o);
            ASSERT_EQ(b.trace.size(), cap) << "m " << m << " cap " << cap;
            EXPECT_EQ(b.r.status, flop::Status::MaxEvaluationsReached);
            for (std::size_t i = 0; i < cap; ++i) {
                if (!v0121::same_bits(b.trace[i].x, whole.trace[i].x)) {
                    ADD_FAILURE() << "m " << m << " cap " << cap << " differs at " << i;
                    break;
                }
            }
            // A cap below the first call's size sends nothing as a call; a cap that
            // holds the first call lets it go whole.
            const std::size_t first = std::min(m, 2 * n + 1);
            if (cap < first) {
                for (const std::size_t sz : b.sizes) EXPECT_EQ(sz, 1u) << "cap " << cap;
            } else {
                EXPECT_EQ(b.sizes[0], first) << "cap " << cap;
            }
        }
    }
}

TEST(V0121Batch, AStopValueMetInsideTheFirstCallCountsTheWholeCall) {
    // The second point of the first call is at or below the stop value; the call
    // has evaluated all five points, all are traced and eligible, and the result
    // is the best of them.
    flop::bobyqa::Options o = v0121::options(0.5, 1000);
    o.stopping.stop_value = 0.2;
    const Outcome t = run_batch(quadratic, Vec{0.1, 0.1}, o);
    EXPECT_EQ(t.r.status, flop::Status::StopValueReached);
    EXPECT_EQ(t.sizes[0], 5u);
    EXPECT_EQ(t.trace.size(), 5u);
    EXPECT_EQ(t.r.evaluations, 5u);
    std::size_t best = 0;
    for (std::size_t i = 1; i < t.trace.size(); ++i)
        if (t.trace[i].f < t.trace[best].f) best = i;
    EXPECT_TRUE(v0121::same_bits(t.r.x, t.trace[best].x));
    EXPECT_LE(t.r.f, 0.2);
}
