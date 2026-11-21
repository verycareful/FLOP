// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: BOBYQA and its box.
//
// Powell, DAMTP 2009/NA06, requires a step onto a bound to put the point
// exactly on it (after (3.2)); FLOP takes it further: the point evaluated has
// the bound's own value in a coordinate a step placed on that bound, so no
// point is a rounding error away from a bound it should be on, and none is
// outside the box (docs/algorithms/bobyqa.md, "Deviations"). A coordinate
// without a bound on a side has no constraint there. Held here: every
// evaluation of every kind of run is inside the box, a coordinate within a few
// units in the last place of a bound is that bound to the bit, a minimiser on
// a face or at a corner is returned with the bound's own value, and a box
// the run never reaches changes nothing.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "flop/flop.hpp"
#include "v0121_problems.hpp"
#include "v0121_reference.hpp"

namespace {

using v0121::Draw;
using v0121::Trace;
using v0121::Vec;

struct Outcome {
    flop::Result r;
    Trace trace;
};

template <class F>
Outcome run(F f, const Vec& x0, flop::bobyqa::Options o, bool batch = false) {
    Outcome out;
    o.on_evaluation = v0121::recorder(out.trace);
    if (batch) {
        auto fb = [&f](std::span<const std::span<const double>> xs, std::span<double> vals) {
            for (std::size_t i = 0; i < xs.size(); ++i) vals[i] = f(xs[i]);
        };
        out.r = flop::bobyqa::minimize_batch(fb, x0, o);
    } else {
        out.r = flop::bobyqa::minimize(f, x0, o);
    }
    return out;
}

double ulp(double v) {
    return std::nextafter(std::fabs(v), 1e300) - std::fabs(v);
}

// Every point of the trace inside the box.
void expect_inside(const Trace& t, const flop::Bounds& b, const char* what) {
    for (const auto& p : t) {
        for (std::size_t i = 0; i < p.x.size(); ++i) {
            const std::optional<double>& lo = b.lower[i];
            const std::optional<double>& hi = b.upper[i];
            if (lo.has_value()) {
                EXPECT_GE(p.x[i], *lo) << what << ": point " << p.index << " coordinate " << i;
            }
            if (hi.has_value()) {
                EXPECT_LE(p.x[i], *hi) << what << ": point " << p.index << " coordinate " << i;
            }
        }
    }
}

// How many coordinates of the trace are within four units in the last place of
// a bound without being that bound to the bit.
std::size_t near_misses(const Trace& t, const flop::Bounds& b) {
    std::size_t count = 0;
    for (const auto& p : t) {
        for (std::size_t i = 0; i < p.x.size(); ++i) {
            const std::optional<double>& lo = b.lower[i];
            const std::optional<double>& hi = b.upper[i];
            if (lo.has_value() && p.x[i] != *lo && p.x[i] - *lo <= 4.0 * ulp(*lo)) ++count;
            if (hi.has_value() && p.x[i] != *hi && *hi - p.x[i] <= 4.0 * ulp(*hi)) ++count;
        }
    }
    return count;
}

double shifted_quadratic(std::span<const double> x, const Vec& c) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        s += (x[i] - c[i]) * (x[i] - c[i]) * (1.0 + 0.5 * static_cast<double>(i));
    return s;
}

}  // namespace

TEST(V0121Bounds, EveryEvaluationOfEveryKindOfRunIsInsideTheBoxAndOnABoundWhereItIsOne) {
    Draw rng(0xB0D5ULL);
    std::size_t on_bound_points = 0, runs = 0, misses = 0;
    for (int trial = 0; trial < 150; ++trial) {
        const std::size_t n = 1 + rng.below(4);
        flop::Bounds b = flop::Bounds::none(n);
        Vec x0(n), c(n);
        const double step = 0.1 + 0.3 * std::fabs(rng.next());
        for (std::size_t i = 0; i < n; ++i) {
            // A box at least 2 step wide on a side, or one sided, or none.
            const std::size_t kind = rng.below(4);
            const double lo = -(2.0 * step + 1.5 * std::fabs(rng.next()));
            const double hi = 2.0 * step + 1.5 * std::fabs(rng.next());
            if (kind != 2) b.lower[i] = lo;
            if (kind != 3) b.upper[i] = hi;
            x0[i] = lo + (hi - lo) * 0.5 * (rng.next() + 1.0);
            // Sometimes on a bound.
            if (kind == 0 && rng.below(3) == 0) x0[i] = lo;
            if (kind == 1 && rng.below(3) == 0) x0[i] = hi;
            if (kind == 2) x0[i] = hi * 0.5 * (rng.next() + 1.0);
            if (kind == 3) x0[i] = lo * 0.5 * (rng.next() + 1.0);
            // A minimiser inside or outside the box.
            c[i] = 3.0 * rng.next();
        }
        flop::bobyqa::Options o = v0121::options(step, 1500);
        o.stopping.xtol_abs = 1e-7;
        o.bounds = b;
        auto f = [&c](std::span<const double> x) { return shifted_quadratic(x, c); };
        for (const bool batch : {false, true}) {
            const Outcome t = run(f, x0, o, batch);
            ASSERT_FALSE(t.trace.empty());
            expect_inside(t.trace, b, "random box");
            misses += near_misses(t.trace, b);
            for (std::size_t i = 0; i < n; ++i) {
                const std::optional<double>& lo = b.lower[i];
                const std::optional<double>& hi = b.upper[i];
                if (lo.has_value()) {
                    EXPECT_GE(t.r.x[i], *lo);
                }
                if (hi.has_value()) {
                    EXPECT_LE(t.r.x[i], *hi);
                }
            }
            for (const auto& p : t.trace) {
                for (std::size_t i = 0; i < n; ++i) {
                    const std::optional<double>& lo = b.lower[i];
                    const std::optional<double>& hi = b.upper[i];
                    if ((lo.has_value() && p.x[i] == *lo) || (hi.has_value() && p.x[i] == *hi))
                        ++on_bound_points;
                }
            }
            ++runs;
        }
    }
    EXPECT_GT(runs, 200u);
    EXPECT_GT(on_bound_points, 200u) << "points on a bound must have been exercised";
    // A coordinate within four units in the last place of a bound is that
    // bound to the bit. The offsets of the box from the origin are rounded when
    // the origin is set or moved, and a coordinate on a bound whose model
    // gradient is rounding noise steps by about that much.
    EXPECT_EQ(misses, 0u) << "coordinates within 4 ulp of a bound but not on it";
}

TEST(V0121Bounds, AnX0MovedToDeltaInsideABoundHasItsOppositePointExactlyOnTheBound) {
    // x0 = 0.2 is within the first radius 0.3 of the lower bound 0.1, so it is
    // moved to 0.1 + 0.3 (Powell, page 5, before (2.2)), and x0 - Delta e_1 is
    // then the bound itself; the point is 0.1 to the bit, not
    // 0.4 - 0.3 = 0.10000000000000003.
    flop::bobyqa::Options o = v0121::options(0.3, 3);
    o.bounds = flop::Bounds::box(Vec{0.1}, Vec{5.0});
    const Outcome t = run(v0111::sphere, Vec{0.2}, o);
    ASSERT_EQ(t.trace.size(), 3u);
    EXPECT_TRUE(v0121::same_bits(t.trace[0].x[0], 0.1 + 0.3)) << "x0 moved";
    EXPECT_TRUE(v0121::same_bits(t.trace[2].x[0], 0.1))
        << "x0 - Delta is the bound: " << t.trace[2].x[0];
    // The same at an upper bound.
    flop::bobyqa::Options u = v0121::options(0.3, 3);
    u.bounds = flop::Bounds::box(Vec{-5.0}, Vec{0.7});
    const Outcome w = run(v0111::sphere, Vec{0.6}, u);
    ASSERT_EQ(w.trace.size(), 3u);
    EXPECT_TRUE(v0121::same_bits(w.trace[1].x[0], 0.7))
        << "x0 + Delta is the bound: " << w.trace[1].x[0];
}

TEST(V0121Bounds, AMinimiserOnAFaceOrAtACornerIsReturnedWithTheBoundsOwnValue) {
    struct Case {
        Vec lo, hi, c, expected;
    };
    const Case cases[] = {
        // On the upper face of the first coordinate.
        {.lo = {-1.0, -1.0}, .hi = {1.0, 1.0}, .c = {2.0, 0.3}, .expected = {1.0, 0.3}},
        // On the lower face of the second.
        {.lo = {-1.0, -1.0}, .hi = {1.0, 1.0}, .c = {0.25, -3.0}, .expected = {0.25, -1.0}},
        // At a corner.
        {.lo = {-1.0, -1.0}, .hi = {1.0, 1.0}, .c = {3.0, -3.0}, .expected = {1.0, -1.0}},
        // Three variables, a face and a corner and an interior coordinate.
        {.lo = {0.0, -2.0, 0.5},
         .hi = {4.0, 2.0, 3.0},
         .c = {-1.0, 5.0, 1.5},
         .expected = {0.0, 2.0, 1.5}},
    };
    for (const Case& k : cases) {
        const std::size_t n = k.lo.size();
        flop::bobyqa::Options o = v0121::options(0.4, 3000);
        o.stopping.xtol_abs = 1e-9;
        o.bounds = flop::Bounds::box(k.lo, k.hi);
        Vec x0(n);
        for (std::size_t i = 0; i < n; ++i) x0[i] = 0.5 * (k.lo[i] + k.hi[i]);
        auto f = [&k](std::span<const double> x) { return shifted_quadratic(x, k.c); };
        for (const bool batch : {false, true}) {
            const Outcome t = run(f, x0, o, batch);
            EXPECT_TRUE(flop::converged(t.r.status)) << flop::to_string(t.r.status);
            for (std::size_t i = 0; i < n; ++i) {
                const bool on_bound = k.expected[i] == k.lo[i] || k.expected[i] == k.hi[i];
                if (on_bound) {
                    EXPECT_TRUE(v0121::same_bits(t.r.x[i], k.expected[i]))
                        << "coordinate " << i << " is " << t.r.x[i] << ", the bound is "
                        << k.expected[i];
                } else {
                    EXPECT_NEAR(t.r.x[i], k.expected[i], 1e-6);
                }
            }
            expect_inside(t.trace, *o.bounds, "face or corner");
        }
    }
}

TEST(V0121Bounds, ASideWithoutABoundHasNoConstraint) {
    // The minimiser is far outside on the unbounded side: the run goes there.
    flop::bobyqa::Options o = v0121::options(0.5, 5000);
    o.stopping.xtol_abs = 1e-8;
    flop::Bounds b = flop::Bounds::none(2);
    b.lower[0] = 0.0;  // only below
    b.upper[1] = 1.0;  // only above
    o.bounds = b;
    const Vec c{40.0, -40.0};
    auto f = [&c](std::span<const double> x) { return shifted_quadratic(x, c); };
    const Outcome t = run(f, Vec{1.0, 0.0}, o);
    EXPECT_TRUE(flop::converged(t.r.status));
    EXPECT_NEAR(t.r.x[0], 40.0, 1e-5);
    EXPECT_NEAR(t.r.x[1], -40.0, 1e-5);
    // And the other way round: the bound on the near side is the answer.
    const Vec d{-40.0, 40.0};
    auto g = [&d](std::span<const double> x) { return shifted_quadratic(x, d); };
    const Outcome u = run(g, Vec{1.0, 0.0}, o);
    EXPECT_TRUE(v0121::same_bits(u.r.x[0], 0.0));
    EXPECT_TRUE(v0121::same_bits(u.r.x[1], 1.0));
}

TEST(V0121Bounds, ABoxTheRunNeverReachesChangesNothing) {
    // A box far wider than where the run goes, on every side: the run is the
    // run without it, to the bit, and the box is a no-op in both channels.
    for (const std::size_t n : {std::size_t{1}, std::size_t{3}}) {
        Vec x0(n, 0.1), c(n);
        for (std::size_t i = 0; i < n; ++i) c[i] = 0.3 * static_cast<double>(i + 1);
        auto f = [&c](std::span<const double> x) { return shifted_quadratic(x, c); };
        flop::bobyqa::Options o = v0121::options(0.5, 2000);
        o.stopping.xtol_abs = 1e-9;
        const Outcome bare = run(f, x0, o);
        o.bounds = flop::Bounds::box(Vec(n, -1e6), Vec(n, 1e6));
        for (const bool batch : {false, true}) {
            const Outcome boxed = run(f, x0, o, batch);
            ASSERT_EQ(boxed.trace.size(), bare.trace.size()) << "n " << n;
            for (std::size_t i = 0; i < bare.trace.size(); ++i) {
                if (!v0121::same_bits(boxed.trace[i].x, bare.trace[i].x)) {
                    ADD_FAILURE() << "n " << n << " differs at point " << i;
                    break;
                }
            }
            EXPECT_EQ(boxed.r.status, bare.r.status);
        }
    }
}

TEST(V0121Bounds, ABoxExactlyTwiceTheStepWideIsUsedToItsEnds) {
    // Width 2 initial_step: the first points on a coordinate are at both ends
    // (x0 in the middle) or at the end and its double (x0 on a bound); none is
    // outside.
    flop::bobyqa::Options o = v0121::options(0.5, 5);
    o.bounds = flop::Bounds::box(Vec{0.0, 0.0}, Vec{1.0, 1.0});
    const Outcome mid = run(v0111::sphere, Vec{0.5, 0.5}, o);
    expect_inside(mid.trace, *o.bounds, "mid");
    // x0 at (0, 1): lower on the first, upper on the second.
    const Outcome corner = run(v0111::sphere, Vec{0.0, 1.0}, o);
    expect_inside(corner.trace, *o.bounds, "corner");
    ASSERT_EQ(corner.trace.size(), 5u);
    EXPECT_TRUE(v0121::same_bits(corner.trace[3].x, Vec{1.0, 1.0}))
        << "x0 + 2 Delta e_1 is the upper bound itself";
    EXPECT_TRUE(v0121::same_bits(corner.trace[4].x, Vec{0.0, 0.0}));
}
