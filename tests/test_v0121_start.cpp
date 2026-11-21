// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: BOBYQA's starting procedure.
//
// Powell, DAMTP 2009/NA06, section 2. x0 is moved onto a bound or inside it
// when it is closer to the bound than the first radius; the points are
// (2.2): x0 +- Delta e_i, or x0 + Delta e_i and x0 + 2 Delta e_i from a lower
// bound, or x0 - Delta e_i and x0 - 2 Delta e_i from an upper one; with more
// than 2n + 1 points the first 2n + 1 are reordered where the second value of
// an interior pair is the lower, and the rest are (2.3) with the pairs of
// (2.4), the report's example for n = 5 and m = 20 among them. The first
// model (2.5) is read off from the values: gradient and diagonal from the
// coordinate points, the off-diagonal pairs from the extra points, the rest
// zero.
//
// The points are read from the evaluation trace of a run capped at m
// evaluations, which ends after exactly the starting points; every
// coordinate is one addition of x0 and a multiplier, so the points are
// compared to the bit. The first model is read from the first step of the
// method: when the objective is a quadratic the model reproduces, the step is
// the minimiser of the quadratic, and when it is not, it is not.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "flop/flop.hpp"
#include "v0121_problems.hpp"
#include "v0121_reference.hpp"

namespace {

using v0121::Place;
using v0121::Trace;
using v0121::Vec;

template <class F>
Trace trace_of(F f, const Vec& x0, flop::bobyqa::Options o, std::size_t evaluations,
               bool batch = false) {
    Trace t;
    o.stopping.max_evaluations = evaluations;
    o.on_evaluation = v0121::recorder(t);
    if (batch) {
        auto fb = [&f](std::span<const std::span<const double>> xs, std::span<double> out) {
            for (std::size_t i = 0; i < xs.size(); ++i) out[i] = f(xs[i]);
        };
        (void)flop::bobyqa::minimize_batch(fb, x0, o);
    } else {
        (void)flop::bobyqa::minimize(f, x0, o);
    }
    return t;
}

flop::bobyqa::Options starting(double delta, std::size_t m) {
    flop::bobyqa::Options o = v0121::options(delta, m);
    o.interpolation_points = m;
    return o;
}

// f = -sum (1 + i/10) x_i: lower at x0 + Delta e_i than at x0 - Delta e_i, so
// the pair is never swapped.
double falling(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) s -= (1.0 + 0.1 * static_cast<double>(i)) * x[i];
    return s;
}

// f = +sum (1 + i/10) x_i: the other way, so the pair of every interior
// coordinate is swapped.
double rising(std::span<const double> x) {
    return -falling(x);
}

// The point x0 + sum of multiples, coordinate by coordinate as the library
// forms it: one addition of x0_i and the multiplier.
Vec point(const Vec& x0, const Vec& offset) {
    Vec p(x0.size());
    for (std::size_t i = 0; i < x0.size(); ++i) p[i] = x0[i] + offset[i];
    return p;
}

void expect_point(const Trace& t, std::size_t index, const Vec& want, const char* what) {
    ASSERT_LT(index, t.size()) << what;
    EXPECT_TRUE(v0121::same_bits(t[index].x, want)) << what << ": point " << index;
}

}  // namespace

TEST(V0121Start, AnInteriorX0GivesTheCoordinatePointsOf22) {
    for (std::size_t n = 1; n <= 4; ++n) {
        Vec x0(n);
        for (std::size_t i = 0; i < n; ++i) x0[i] = 0.3 + 0.17 * static_cast<double>(i);
        const double delta = 0.4;
        const Trace t = trace_of(v0111::sphere, x0, starting(delta, 2 * n + 1), 2 * n + 1);
        ASSERT_EQ(t.size(), 2 * n + 1);
        expect_point(t, 0, x0, "y_1 = x0");
        for (std::size_t i = 0; i < n; ++i) {
            Vec up(n, 0.0), down(n, 0.0);
            up[i] = delta;
            down[i] = -delta;
            expect_point(t, i + 1, point(x0, up), "x0 + Delta e_i");
            expect_point(t, n + i + 1, point(x0, down), "x0 - Delta e_i");
        }
    }
}

TEST(V0121Start, X0OnALowerBoundGivesPlusDeltaAndPlusTwoDelta) {
    for (std::size_t n = 1; n <= 3; ++n) {
        const Vec x0(n, 1.0);
        const double delta = 0.5;
        flop::bobyqa::Options o = starting(delta, 2 * n + 1);
        o.bounds = flop::Bounds::box(Vec(n, 1.0), Vec(n, 9.0));
        const Trace t = trace_of(v0111::sphere, x0, o, 2 * n + 1);
        ASSERT_EQ(t.size(), 2 * n + 1);
        expect_point(t, 0, x0, "y_1 = x0");
        for (std::size_t i = 0; i < n; ++i) {
            Vec one(n, 0.0), two(n, 0.0);
            one[i] = delta;
            two[i] = 2.0 * delta;
            expect_point(t, i + 1, point(x0, one), "x0 + Delta e_i");
            expect_point(t, n + i + 1, point(x0, two), "x0 + 2 Delta e_i");
        }
    }
}

TEST(V0121Start, X0OnAnUpperBoundGivesMinusDeltaAndMinusTwoDelta) {
    for (std::size_t n = 1; n <= 3; ++n) {
        const Vec x0(n, 9.0);
        const double delta = 0.5;
        flop::bobyqa::Options o = starting(delta, 2 * n + 1);
        o.bounds = flop::Bounds::box(Vec(n, 1.0), Vec(n, 9.0));
        const Trace t = trace_of(v0111::sphere, x0, o, 2 * n + 1);
        ASSERT_EQ(t.size(), 2 * n + 1);
        expect_point(t, 0, x0, "y_1 = x0");
        for (std::size_t i = 0; i < n; ++i) {
            Vec one(n, 0.0), two(n, 0.0);
            one[i] = -delta;
            two[i] = -2.0 * delta;
            expect_point(t, i + 1, point(x0, one), "x0 - Delta e_i");
            expect_point(t, n + i + 1, point(x0, two), "x0 - 2 Delta e_i");
        }
    }
}

TEST(V0121Start, X0CloserToABoundThanDeltaIsMovedToDeltaInsideIt) {
    // Delta = 1/2 on [0, 4]: 1/4 is moved to 1/2 and 15/4 to 7/2; 1/2 and 7/2
    // are not closer than Delta and stay, with interior pairs (the point at
    // 0 and 4 is then a bound itself).
    const double delta = 0.5;
    flop::bobyqa::Options o = starting(delta, 5);
    o.bounds = flop::Bounds::box(Vec{0.0, 0.0}, Vec{4.0, 4.0});
    struct Case {
        Vec x0;
        Vec moved;
    };
    for (const Case& c : {Case{.x0 = {0.25, 3.75}, .moved = {0.5, 3.5}},
                          Case{.x0 = {0.5, 3.5}, .moved = {0.5, 3.5}},
                          Case{.x0 = {2.0, 0.25}, .moved = {2.0, 0.5}}}) {
        const Trace t = trace_of(v0111::sphere, c.x0, o, 5);
        ASSERT_EQ(t.size(), 5u);
        expect_point(t, 0, c.moved, "x0 moved");
        for (std::size_t i = 0; i < 2; ++i) {
            Vec up(2, 0.0), down(2, 0.0);
            up[i] = delta;
            down[i] = -delta;
            expect_point(t, i + 1, point(c.moved, up), "x0 + Delta e_i");
            expect_point(t, 2 + i + 1, point(c.moved, down), "x0 - Delta e_i");
        }
    }
    // A coordinate bounded on one side only is moved too.
    o.bounds = flop::Bounds::none(2);
    o.bounds->lower[0] = 0.0;
    o.bounds->upper[1] = 4.0;
    const Trace t = trace_of(v0111::sphere, Vec{0.25, 3.75}, o, 5);
    expect_point(t, 0, Vec{0.5, 3.5}, "x0 moved on one-sided bounds");
}

TEST(V0121Start, FewerThanTwoNPlusOnePointsAreTheFirstOnes) {
    // m <= 2n: the points y_j with j > m are not formed or evaluated.
    const Vec x0{0.3, 0.6, 0.9};
    const double delta = 0.25;
    for (std::size_t m = 5; m <= 6; ++m) {
        const Trace t = trace_of(v0111::sphere, x0, starting(delta, m), m);
        ASSERT_EQ(t.size(), m);
        expect_point(t, 0, x0, "y_1");
        for (std::size_t i = 0; i < 3; ++i) {
            Vec up(3, 0.0);
            up[i] = delta;
            expect_point(t, i + 1, point(x0, up), "x0 + Delta e_i");
        }
        for (std::size_t i = 0; i + 4 < m; ++i) {
            Vec down(3, 0.0);
            down[i] = -delta;
            expect_point(t, 4 + i, point(x0, down), "x0 - Delta e_i");
        }
    }
}

TEST(V0121Start, TheExtraPointsFollowThePairsOf23And24) {
    // The report's example: n = 5, m = 20 has the pairs {1,2} {2,3} {3,4}
    // {4,5} {5,1} {1,3} {2,4} {3,5} {4,1}, from the 12th point on.
    const Vec x0(5, 0.25);
    const double delta = 0.5;
    const Trace t = trace_of(falling, x0, starting(delta, 20), 20);
    ASSERT_EQ(t.size(), 20u);
    const std::size_t pairs[9][2] = {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 1},
                                     {1, 3}, {2, 4}, {3, 5}, {4, 1}};
    for (std::size_t e = 0; e < 9; ++e) {
        Vec off(5, 0.0);
        off[pairs[e][0] - 1] = delta;
        off[pairs[e][1] - 1] = delta;
        expect_point(t, 11 + e, point(x0, off), "an extra point of (2.3)");
    }
}

TEST(V0121Start, TheExtraPointsOfEveryLegalCountAreTheOnesOf23) {
    // Every n from 2 to 5 and every m beyond 2n + 1, against the points the
    // report's equations give (v0121_reference.hpp writes them out again).
    for (std::size_t n = 2; n <= 5; ++n)
        for (std::size_t m = 2 * n + 2; m <= (n + 1) * (n + 2) / 2; ++m) {
            Vec x0(n);
            for (std::size_t i = 0; i < n; ++i) x0[i] = 0.25 * static_cast<double>(i + 1);
            const double delta = 0.5;
            const Trace t = trace_of(falling, x0, starting(delta, m), m);
            ASSERT_EQ(t.size(), m);
            std::vector<double> alpha(n, delta), beta(n, -delta);
            const v0121::Points y = v0121::coordinate_points(alpha, beta, m);
            for (std::size_t j = 2 * n + 1; j < m; ++j) {
                const Vec off(y.begin() + static_cast<std::ptrdiff_t>(j * n),
                              y.begin() + static_cast<std::ptrdiff_t>((j + 1) * n));
                expect_point(t, j, point(x0, off), "an extra point");
            }
        }
}

TEST(V0121Start, AnInteriorPairIsSwappedWhereTheSecondValueIsTheLower) {
    // With rising f the value at x0 - Delta e_i is the lower for every i, so
    // the extra points are built on the points at x0 - Delta e_i. The first
    // 2n + 1 points are evaluated before the swap and keep their order.
    const std::size_t n = 3, m = 8;
    const Vec x0(n, 0.25);
    const double delta = 0.5;
    const Trace t = trace_of(rising, x0, starting(delta, m), m);
    ASSERT_EQ(t.size(), m);
    for (std::size_t i = 0; i < n; ++i) {
        Vec up(n, 0.0), down(n, 0.0);
        up[i] = delta;
        down[i] = -delta;
        expect_point(t, i + 1, point(x0, up), "first call, y_{i+1}");
        expect_point(t, n + i + 1, point(x0, down), "first call, y_{n+i+1}");
    }
    // The one extra point is on {p, q} = {1, 2}, on the lower side of both.
    expect_point(t, 7, point(x0, Vec{-delta, -delta, 0.0}), "swapped extra point");
}

TEST(V0121Start, OnlyThePairsWithALowerSecondValueAreSwapped) {
    // Slopes of alternating sign: coordinates 1 and 3 fall towards +, 2 and 4
    // towards -. The extra points of n = 4, m = 12 are on {1,2} {2,3} {3,4},
    // and each coordinate of them is on the side of the lower value.
    const std::size_t n = 4, m = 12;
    const Vec x0(n, 0.25);
    const double delta = 0.5;
    auto f = [](std::span<const double> x) { return -x[0] + x[1] - x[2] + x[3]; };
    const Trace t = trace_of(f, x0, starting(delta, m), m);
    ASSERT_EQ(t.size(), m);
    const double side[4] = {delta, -delta, delta, -delta};  // the lower value's side
    const std::size_t pairs[3][2] = {{0, 1}, {1, 2}, {2, 3}};
    for (std::size_t e = 0; e < 3; ++e) {
        Vec off(n, 0.0);
        off[pairs[e][0]] = side[pairs[e][0]];
        off[pairs[e][1]] = side[pairs[e][1]];
        expect_point(t, 2 * n + 1 + e, point(x0, off), "an extra point with mixed sides");
    }
}

TEST(V0121Start, EqualValuesAreNotSwappedAndNeitherIsABoundCoordinate) {
    const std::size_t m = 6;
    const double delta = 0.5;
    {
        // f symmetric about x0: the two values of every pair are equal, and
        // equal is not lower.
        const Vec x0{0.25, 0.75};
        auto f = [&x0](std::span<const double> x) {
            return (x[0] - x0[0]) * (x[0] - x0[0]) + (x[1] - x0[1]) * (x[1] - x0[1]);
        };
        const Trace t = trace_of(f, x0, starting(delta, m), m);
        ASSERT_EQ(t.size(), m);
        EXPECT_TRUE(v0121::same_bits(t[1].f, t[3].f));
        expect_point(t, 5, point(x0, Vec{delta, delta}), "an unswapped extra point");
    }
    {
        // x0 on the lower bound of both coordinates, f decreasing so that the
        // value at x0 + 2 Delta e_i is the lower: a coordinate on a bound is
        // not swapped.
        const Vec x0{1.0, 1.0};
        flop::bobyqa::Options o = starting(delta, m);
        o.bounds = flop::Bounds::box(Vec{1.0, 1.0}, Vec{9.0, 9.0});
        const Trace t = trace_of(falling, x0, o, m);
        ASSERT_EQ(t.size(), m);
        EXPECT_LT(t[3].f, t[1].f) << "the second value must be the lower";
        expect_point(t, 5, point(x0, Vec{delta, delta}), "an extra point from a lower bound");
    }
}

// ---- the first model (2.5), read from the first step -------------------------

namespace {

// The distance between the point after the starting ones and the minimiser
// c, and the step's length from the best starting point.
struct FirstStep {
    double to_minimiser = 0.0;
    double length = 0.0;
    bool found = false;
};

template <class F>
FirstStep first_step(F f, const Vec& x0, const Vec& c, flop::bobyqa::Options o, std::size_t m,
                     bool batch = false) {
    const Trace t = trace_of(f, x0, std::move(o), m + 1, batch);
    FirstStep r;
    if (t.size() != m + 1) return r;
    std::size_t best = 0;
    for (std::size_t j = 1; j < m; ++j)
        if (t[j].f < t[best].f) best = j;
    for (std::size_t i = 0; i < c.size(); ++i) {
        r.to_minimiser = std::max(r.to_minimiser, std::fabs(t[m].x[i] - c[i]));
    }
    double len2 = 0.0;
    for (std::size_t i = 0; i < c.size(); ++i) {
        const double d = t[m].x[i] - t[best].x[i];
        len2 += d * d;
    }
    r.length = std::sqrt(len2);
    r.found = true;
    return r;
}

}  // namespace

TEST(V0121Start, ADiagonalQuadraticGivesAFirstModelThatIsExact) {
    // f = sum (x_i - c_i)^2 with c at 0.2 Delta from x0: the model of (2.5)
    // from 2n + 1 points is f itself, x0 is the best starting point, and the
    // first trust-region step, inside the radius, goes to c.
    for (std::size_t n = 1; n <= 4; ++n) {
        const double delta = 0.5;
        Vec x0(n), c(n);
        for (std::size_t i = 0; i < n; ++i) {
            x0[i] = 0.25 * static_cast<double>(i + 1);
            c[i] = x0[i] + 0.2 * delta * (i % 2 == 0 ? 1.0 : -1.0);
        }
        auto f = [&c](std::span<const double> x) {
            double s = 0.0;
            for (std::size_t i = 0; i < x.size(); ++i) s += (x[i] - c[i]) * (x[i] - c[i]);
            return s;
        };
        const FirstStep r = first_step(f, x0, c, starting(delta, 2 * n + 1), 2 * n + 1);
        ASSERT_TRUE(r.found) << n;
        EXPECT_LT(r.to_minimiser, 1e-9) << "n " << n;
        EXPECT_LT(r.length, delta);
    }
}

TEST(V0121Start, AnOffDiagonalPairIsInTheFirstModelOnlyWithAPointForIt) {
    // n = 3: the extra point of m = 8 gives the pair {1, 2} its second
    // derivative, so a quadratic whose only off-diagonal element is there is
    // reproduced, and its minimiser is the first step. With 2n + 1 points the
    // off-diagonal elements are zero and the first step is not the minimiser.
    const std::size_t n = 3;
    const double delta = 0.5;
    const Vec x0{0.25, 0.5, 0.75};
    const Vec c{x0[0] + 0.1, x0[1] - 0.1, x0[2] + 0.05};
    auto f = [&c](std::span<const double> x) {
        const double a = x[0] - c[0], b = x[1] - c[1], e = x[2] - c[2];
        return a * a + b * b + e * e + 0.8 * a * b;
    };
    const FirstStep with_pair = first_step(f, x0, c, starting(delta, 8), 8);
    ASSERT_TRUE(with_pair.found);
    EXPECT_LT(with_pair.to_minimiser, 1e-8);
    const FirstStep without = first_step(f, x0, c, starting(delta, 2 * n + 1), 2 * n + 1);
    ASSERT_TRUE(without.found);
    EXPECT_GT(without.to_minimiser, 1e-4);
}

TEST(V0121Start, X0OnALowerBoundGivesAnExactModelOfADiagonalQuadratic) {
    // The pair at x0 and x0 + Delta, x0 + 2 Delta of (2.2) determines the
    // diagonal element as well as the interior pair does.
    const double delta = 0.5;
    const Vec x0{1.0, 1.0};
    const Vec c{x0[0] + 0.1, x0[1] + 0.15};
    flop::bobyqa::Options o = starting(delta, 5);
    o.bounds = flop::Bounds::box(Vec{1.0, 1.0}, Vec{9.0, 9.0});
    auto f = [&c](std::span<const double> x) {
        return (x[0] - c[0]) * (x[0] - c[0]) + (x[1] - c[1]) * (x[1] - c[1]);
    };
    const FirstStep r = first_step(f, x0, c, o, 5);
    ASSERT_TRUE(r.found);
    EXPECT_LT(r.to_minimiser, 1e-9);
}

TEST(V0121Start, WithFewerPointsThanTwoNPlusOneTheMissingCurvatureIsZero) {
    // m = n + 2 = 5 at n = 3: the diagonal elements of coordinates 2 and 3
    // are zero (2.5), so the model is linear along them and a step along
    // them runs to the trust-region boundary: the first step has the length
    // of the radius, where a quadratic with its minimum at 0.2 Delta from
    // x0 would not.
    const double delta = 0.5;
    const Vec x0{0.25, 0.5, 0.75};
    const Vec c{x0[0] + 0.1, x0[1] + 0.1, x0[2] - 0.1};
    auto f = [&c](std::span<const double> x) {
        double s = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) s += (x[i] - c[i]) * (x[i] - c[i]);
        return s;
    };
    const FirstStep r = first_step(f, x0, c, starting(delta, 5), 5);
    ASSERT_TRUE(r.found);
    EXPECT_NEAR(r.length, delta, 1e-9 * delta);
    EXPECT_GT(r.to_minimiser, 1e-3);
}
