// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: how BOBYQA controls its radii.
//
// Powell, DAMTP 2009/NA06, section 6. The control is private to the solver, so
// it is held through runs built so that the rule's consequence is a number the
// evaluation trace shows, in dyadic arithmetic where the points can be
// compared exactly:
//
//   (6.4) the ratio of actual to predicted reduction sets the next radius
//         Delta: above 0.7 it is the larger of Delta/2 and 2|d|, up to 0.1 the
//         smaller of Delta/2 and |d|, and a Delta within 1.5 rho is rho. On a
//         linear objective the ratio is 1 and the step runs to the boundary,
//         so the radius doubles; on an objective that goes flat at the best
//         point the ratio is 0 and the radius halves, down to rho.
//   (6.5) a step shorter than rho / 2 is not evaluated at once: with the
//         points close and rho at its end it is evaluated last, and the run
//         ends at rho_end, which it reaches as the report says (6.6).
//   (6.13) the origin moves to x_k when the step is short against the distance
//         of x_k from it, so that a problem far from the origin is solved to
//         the accuracy of its coordinates and in the number of evaluations of
//         the same problem at the origin.
//
// RESCUE is called, in these runs and in others, when rounding makes the
// denominator of the update inadequate (5.2); that the runs still end where
// the report says, on the minimiser and at rho_end, is held at the end.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "flop/flop.hpp"
#include "v0121_problems.hpp"

namespace {

using v0121::Trace;
using v0121::Vec;

struct Outcome {
    flop::Result r;
    Trace trace;
};

template <class F>
Outcome run(F f, const Vec& x0, flop::bobyqa::Options o) {
    Outcome out;
    o.on_evaluation = v0121::recorder(out.trace);
    out.r = flop::bobyqa::minimize(f, x0, o);
    return out;
}

double dist(const Vec& a, const Vec& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(s);
}

}  // namespace

TEST(V0121Control, TheRadiusDoublesWhileTheReductionIsAsPredicted) {
    // f = -x: the model after the starting points is f itself, every step runs
    // to the trust-region boundary and the reduction is as predicted, so
    // Delta = max(Delta / 2, 2 |d|) = 2 Delta. From x0 = 0 with rho_beg = 1 the
    // starting points are 0, 1, -1; the steps then go to 2, 4, 8, ...
    for (const double scale : {1.0, 0.5, 4.0}) {
        flop::bobyqa::Options o = v0121::options(scale, 14);
        auto f = [scale](std::span<const double> x) { return -scale * (x[0] / scale); };
        const auto t = run(f, Vec{0.0}, o);
        ASSERT_EQ(t.trace.size(), 14u);
        double want = 2.0 * scale;
        for (std::size_t i = 3; i < t.trace.size(); ++i) {
            EXPECT_NEAR(t.trace[i].x[0], want, 1e-12 * want) << "scale " << scale << " point " << i;
            want *= 2.0;
        }
    }
}

TEST(V0121Control, TheStepsOfALinearObjectiveInSeveralVariablesDoubleInLength) {
    // The same in two and three variables: successive steps from the best point
    // have the lengths Delta, 2 Delta, 4 Delta, ...
    for (const std::size_t n : {std::size_t{2}, std::size_t{3}}) {
        Vec x0(n, 0.0);
        auto f = [](std::span<const double> x) {
            double s = 0.0;
            for (std::size_t i = 0; i < x.size(); ++i)
                s -= (1.0 + 0.5 * static_cast<double>(i)) * x[i];
            return s;
        };
        const double delta = 0.5;
        const auto t = run(f, x0, v0121::options(delta, 2 * n + 1 + 6));
        ASSERT_EQ(t.trace.size(), 2 * n + 7);
        // The best starting point is the one the first step leaves.
        std::size_t best = 0;
        for (std::size_t j = 1; j < 2 * n + 1; ++j)
            if (t.trace[j].f < t.trace[best].f) best = j;
        Vec from = t.trace[best].x;
        double want = delta;
        for (std::size_t i = 2 * n + 1; i < t.trace.size(); ++i) {
            EXPECT_NEAR(dist(t.trace[i].x, from), want, 1e-12 * want) << "n " << n << " step " << i;
            from = t.trace[i].x;
            want *= 2.0;
        }
    }
}

TEST(V0121Control, TheRadiusHalvesWhenTheObjectiveGoesFlatAtTheBestPoint) {
    // f = -min(x, 8 s) (s a power of two, so everything is exact): the radius
    // grows to 8 s and the step from 8 s to 16 s finds the same value, a ratio
    // of 0, so Delta = min(Delta / 2, |d|) = 4 s, 2 s, s: the steps from the best
    // point 8 s have the lengths 8 s, 4 s, 2 s, s. Delta then equals rho (= s,
    // the first radius, which a radius within 1.5 rho is made), and no point
    // is further than rho from the best one.
    for (const double s : {1.0, 0.5, 2.0}) {
        auto f = [s](std::span<const double> x) { return -std::min(x[0], 8.0 * s); };
        const auto t = run(f, Vec{0.0}, v0121::options(s, 18));
        ASSERT_EQ(t.trace.size(), 18u);
        const Vec best{8.0 * s};
        EXPECT_NEAR(t.trace[5].x[0], 8.0 * s, 1e-12 * s) << "the best point, scale " << s;
        EXPECT_NEAR(t.trace[6].x[0], 16.0 * s, 1e-12 * s);
        double length = 8.0 * s;
        for (std::size_t i = 6; i <= 9; ++i) {
            EXPECT_NEAR(dist(t.trace[i].x, best), length, 1e-9 * s)
                << "scale " << s << " point " << i;
            length *= 0.5;
        }
        for (std::size_t i = 10; i < t.trace.size(); ++i)
            EXPECT_LE(dist(t.trace[i].x, best), s * (1.0 + 1e-12))
                << "scale " << s << " point " << i;
        // Nothing improved on the best value after it was found.
        EXPECT_EQ(t.r.f, -8.0 * s);
    }
}

namespace {

// f = sum (x_i - c_i)^2 with c_i = 0.2 + 0.05 i: a minimiser at 0.32 from the
// origin, inside the first radius and short against it.
double near_origin(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double d = x[i] - (0.2 + 0.05 * static_cast<double>(i));
        s += d * d;
    }
    return s;
}

}  // namespace

TEST(V0121Control, AShortStepAtTheEndIsEvaluatedLastAndTheRunEndsAtRhoEnd) {
    // The model after the starting points is f, and the first step goes to its
    // minimiser, 0.32 from x_k: shorter than rho / 2 = 1/2. The points are all
    // within 10 rho, and rho_end = rho, so (6.6) ends the run, as the report
    // does, after evaluating that step last. Five starting points, one more.
    flop::bobyqa::Options o = v0121::options(1.0, 50);
    o.stopping.xtol_rel = 1.0;
    const auto t = run(near_origin, Vec{0.0, 0.0}, o);
    ASSERT_EQ(t.trace.size(), 6u);
    EXPECT_NEAR(t.trace[5].x[0], 0.2, 1e-9);
    EXPECT_NEAR(t.trace[5].x[1], 0.25, 1e-9);
    EXPECT_EQ(t.r.status, flop::Status::XtolReached);
    EXPECT_EQ(t.r.final_radius, 1.0);
    EXPECT_LT(t.r.f, 1e-15) << "the result is the step that was evaluated last";
}

TEST(V0121Control, WithRhoAboveRhoEndTheSameStepIsAnOrdinaryOneAfterTheCut) {
    // rho_end = 1e-3: rho is above 250 rho_end, so (6.6) cuts it to rho / 10 and
    // Delta becomes max(rho_old / 2, rho_new); the step to the minimiser,
    // 0.32 long, is then not short, and is evaluated as a step. The run goes on
    // to rho_end, which it reaches exactly, and ends on the minimiser.
    flop::bobyqa::Options o = v0121::options(1.0, 200);
    o.stopping.xtol_abs = 1e-3;
    const auto t = run(near_origin, Vec{0.0, 0.0}, o);
    ASSERT_GT(t.trace.size(), 6u);
    EXPECT_NEAR(t.trace[5].x[0], 0.2, 1e-9);
    EXPECT_NEAR(t.trace[5].x[1], 0.25, 1e-9);
    EXPECT_EQ(t.r.status, flop::Status::XtolReached);
    EXPECT_EQ(t.r.final_radius, 1e-3) << "rho reaches rho_end, not below it";
    EXPECT_NEAR(t.r.x[0], 0.2, 1e-9);
    EXPECT_NEAR(t.r.x[1], 0.25, 1e-9);
}

TEST(V0121Control, RhoEndAboveTheFirstRadiusIsTheFirstRadius) {
    // rho_end is never above rho_beg: asked for more, the run ends as soon as
    // the first step has been taken and evaluated.
    flop::bobyqa::Options o = v0121::options(1.0, 50);
    o.stopping.xtol_abs = 7.0;
    const auto t = run(near_origin, Vec{0.0, 0.0}, o);
    EXPECT_EQ(t.r.status, flop::Status::XtolReached);
    EXPECT_EQ(t.r.final_radius, 1.0);
    EXPECT_LE(t.trace.size(), 6u);
}

TEST(V0121Control, AProblemFarFromTheOriginIsSolvedAsWellAsOneAtIt) {
    // (6.13): the origin follows x_k, so the arithmetic of the model works on
    // differences of nearby points however far the problem is. The same
    // quadratic translated by T (a power of two, so the translation is exact)
    // reaches its minimiser to the spacing of the doubles at T, in about as
    // many evaluations as at T = 0.
    std::size_t base_evals = 0;
    for (const double translation : {0.0, 1024.0, 1048576.0, 1073741824.0}) {
        flop::bobyqa::Options o = v0121::options_xtol(0.5, 1e-9, 4000);
        auto f = [translation](std::span<const double> x) {
            double s = 0.0;
            for (std::size_t i = 0; i < x.size(); ++i) {
                const double d = (x[i] - translation) - 0.3 * static_cast<double>(i + 1);
                s += d * d;
            }
            return s;
        };
        const Vec x0(3, translation);
        const flop::Result r = flop::bobyqa::minimize(f, x0, o);
        EXPECT_EQ(r.status, flop::Status::XtolReached) << translation;
        const double ulp = std::nextafter(translation + 1.0, 1e300) - (translation + 1.0);
        for (std::size_t i = 0; i < 3; ++i)
            EXPECT_NEAR(r.x[i], translation + 0.3 * static_cast<double>(i + 1), 64.0 * ulp + 1e-8)
                << "translation " << translation << " coordinate " << i;
        if (translation == 0.0) base_evals = r.evaluations;
        EXPECT_LE(r.evaluations, 2 * base_evals) << "translation " << translation;
    }
}

TEST(V0121Control, RunsThatCallRescueStillEndOnTheMinimiserAtRhoEnd) {
    // With rho_end = 1e-12 the points are within a few ulps of each other
    // against the values of f, rounding spoils the denominator (5.2), and RESCUE
    // rebuilds H (it is called in these runs). The method goes on from the
    // rebuilt model and ends at rho_end on the minimiser.
    for (const std::size_t n : {std::size_t{2}, std::size_t{5}}) {
        flop::bobyqa::Options o = v0121::options(0.5, 3000);
        o.stopping.xtol_abs = 1e-12;
        auto f = [](std::span<const double> x) {
            double s = 0.0;
            for (std::size_t i = 0; i < x.size(); ++i) {
                const double d = x[i] - 0.3 * static_cast<double>(i + 1);
                s += d * d;
            }
            return s;
        };
        const auto t = run(f, Vec(n, 0.1), o);
        EXPECT_EQ(t.r.status, flop::Status::XtolReached) << "n " << n;
        EXPECT_EQ(t.r.final_radius, 1e-12) << "n " << n;
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(t.r.x[i], 0.3 * static_cast<double>(i + 1), 1e-9) << "n " << n;
        EXPECT_LT(t.r.f, 1e-18);
    }
}
