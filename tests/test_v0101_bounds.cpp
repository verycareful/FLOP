// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.0.1 test wave: box bounds.
//
// A box is a promise about every evaluated point, not only the returned
// one: an objective that is undefined outside the box (a circuit parameter
// that must stay in its range, a log of a quantity that must stay positive)
// is called at every point the optimizer chooses, including the initial
// simplex and every geometry repair. The promise is checked here through
// on_evaluation, which sees each point before the objective's value is
// used.
//
// The other half of the contract is that a bound that never binds changes
// nothing: the projection inside the step is the identity when the step
// stays inside, so the trajectory with a wide box is the unbounded
// trajectory to the bit.

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "flop/flop.hpp"
#include "v0101_problems.hpp"

namespace {

struct Watch {
    std::vector<std::optional<double>> lower, upper;
    std::size_t seen = 0;
    std::size_t outside = 0;
    void operator()(const flop::Evaluation& e) {
        ++seen;
        for (std::size_t i = 0; i < e.x.size(); ++i) {
            const std::optional<double>& lo = lower[i];
            const std::optional<double>& hi = upper[i];
            if (lo.has_value() && e.x[i] < *lo) ++outside;
            if (hi.has_value() && e.x[i] > *hi) ++outside;
        }
    }
};

// Every evaluation of `opts` on the sphere from x0 stays inside opts.bounds.
flop::Result run_watched(std::span<const double> x0, flop::cobyla::Options opts, Watch& w) {
    if (opts.bounds.has_value()) {
        w.lower = opts.bounds->lower;
        w.upper = opts.bounds->upper;
    }
    opts.on_evaluation = [&w](const flop::Evaluation& e) { w(e); };
    return flop::cobyla::minimize(v0101::sphere, x0, opts);
}

}  // namespace

TEST(V0101Bounds, AnOptimumOutsideTheBoxLandsOnItsFace) {
    // The sphere's centre is (0.3, 0.6, 0.9); the box tops out at
    // (0.2, 0.5, 0.7), so the constrained minimum is that corner and its
    // value the squared distance, 0.01 + 0.01 + 0.04.
    const std::vector<double> x0{0.0, 0.0, 0.0};
    const std::vector<double> lo{-1.0, -1.0, -1.0};
    const std::vector<double> hi{0.2, 0.5, 0.7};
    flop::cobyla::Options o = v0101::options(0.3, 1e-9, 2000);
    o.bounds = flop::Bounds::box(lo, hi);
    Watch w;
    const flop::Result r = run_watched(x0, o, w);
    EXPECT_TRUE(flop::converged(r.status)) << flop::to_string(r.status);
    EXPECT_LE(v0101::max_abs_diff(r.x, hi), 1e-8);
    EXPECT_NEAR(r.f, 0.06, 1e-12);
    EXPECT_EQ(w.outside, 0u);
    EXPECT_EQ(w.seen, r.evaluations);
}

TEST(V0101Bounds, EveryEvaluationIsInsideTheBoxIncludingTheInitialSimplex) {
    // x0 sits within one step of the upper bound on every coordinate, so
    // the +rho poke of the initial simplex would leave the box on each one
    // and the construction has to go the other way.
    const std::vector<double> x0{0.15, 0.45, 0.65};
    const std::vector<double> lo{-1.0, -1.0, -1.0};
    const std::vector<double> hi{0.2, 0.5, 0.7};
    flop::cobyla::Options o = v0101::options(0.3, 1e-9, 2000);
    o.bounds = flop::Bounds::box(lo, hi);
    Watch w;
    const flop::Result r = run_watched(x0, o, w);
    EXPECT_EQ(w.outside, 0u);
    EXPECT_EQ(w.seen, r.evaluations);
    EXPECT_LE(v0101::max_abs_diff(r.x, hi), 1e-8);
}

TEST(V0101Bounds, ABoxNarrowerThanTheStepOnEveryCoordinate) {
    // Room of 0.1 on each side against a step of 0.3: the initial simplex
    // has to shrink its offsets to the room available.
    const std::vector<double> x0{0.0, 0.0};
    const std::vector<double> lo{-0.1, -0.1};
    const std::vector<double> hi{0.1, 0.1};
    flop::cobyla::Options o = v0101::options(0.3, 1e-9, 2000);
    o.bounds = flop::Bounds::box(lo, hi);
    Watch w;
    const flop::Result r = run_watched(x0, o, w);
    EXPECT_EQ(w.outside, 0u);
    EXPECT_TRUE(flop::converged(r.status)) << flop::to_string(r.status);
    // The centre (0.3, 0.6) is outside on both coordinates: the corner
    // (0.1, 0.1) is the minimum.
    EXPECT_LE(v0101::max_abs_diff(r.x, hi), 1e-8);
}

TEST(V0101Bounds, OneSidedBoundsAreHonoured) {
    // Lower bound only on coordinate 0 (above the centre, so it binds),
    // upper only on coordinate 1 (above the centre, so it does not).
    const std::vector<double> x0{1.0, 0.0};
    flop::cobyla::Options o = v0101::options(0.3, 1e-9, 2000);
    o.bounds = flop::Bounds::none(2);
    o.bounds->lower[0] = 0.5;
    o.bounds->upper[1] = 2.0;
    Watch w;
    const flop::Result r = run_watched(x0, o, w);
    EXPECT_EQ(w.outside, 0u);
    EXPECT_TRUE(flop::converged(r.status)) << flop::to_string(r.status);
    EXPECT_NEAR(r.x[0], 0.5, 1e-8);
    EXPECT_NEAR(r.x[1], v0101::sphere_centre(1), 1e-7);
}

TEST(V0101Bounds, StartingInACornerWithTheOptimumOutside) {
    // x0 is the corner nearest the centre; no feasible direction improves,
    // and the run must recognise that and stop there rather than loop.
    const std::vector<double> x0{0.2, 0.5, 0.7};
    const std::vector<double> lo{-1.0, -1.0, -1.0};
    const std::vector<double> hi{0.2, 0.5, 0.7};
    flop::cobyla::Options o = v0101::options(0.3, 1e-9, 500);
    o.bounds = flop::Bounds::box(lo, hi);
    Watch w;
    const flop::Result r = run_watched(x0, o, w);
    EXPECT_EQ(w.outside, 0u);
    EXPECT_TRUE(flop::converged(r.status)) << flop::to_string(r.status);
    EXPECT_LE(v0101::max_abs_diff(r.x, x0), 1e-12);
    EXPECT_LT(r.evaluations, 500u);
}

TEST(V0101Bounds, AnInactiveBoxGivesTheUnboundedTrajectoryToTheBit) {
    const std::vector<double> x0{-1.2, 1.0};
    std::vector<std::vector<double>> free_points, boxed_points;
    flop::cobyla::Options free_o = v0101::options(0.5, 1e-9, 5000);
    free_o.on_evaluation = [&](const flop::Evaluation& e) {
        free_points.emplace_back(e.x.begin(), e.x.end());
    };
    flop::cobyla::Options boxed_o = v0101::options(0.5, 1e-9, 5000);
    const std::vector<double> lo{-100.0, -100.0};
    const std::vector<double> hi{100.0, 100.0};
    boxed_o.bounds = flop::Bounds::box(lo, hi);
    boxed_o.on_evaluation = [&](const flop::Evaluation& e) {
        boxed_points.emplace_back(e.x.begin(), e.x.end());
    };
    const flop::Result a = flop::cobyla::minimize(v0101::rosenbrock, x0, free_o);
    const flop::Result b = flop::cobyla::minimize(v0101::rosenbrock, x0, boxed_o);
    ASSERT_EQ(free_points.size(), boxed_points.size());
    for (std::size_t k = 0; k < free_points.size(); ++k)
        ASSERT_TRUE(v0101::same_bits(free_points[k], boxed_points[k])) << "evaluation " << k;
    EXPECT_TRUE(v0101::same_bits(a.x, b.x));
    EXPECT_TRUE(v0101::same_bits(a.f, b.f));
    EXPECT_EQ(a.status, b.status);
}

TEST(V0101Bounds, ABoxWithConstraintsKeepsBothPromises) {
    // HS35's constraint plus its x >= 0 box; the run must stay in the box at
    // every evaluation and end on the constraint.
    const std::vector<double> x0{0.5, 0.5, 0.5};
    flop::cobyla::Options o = v0101::options(0.5, 2e-9, 20000);
    o.bounds = flop::Bounds::none(3);
    for (std::size_t i = 0; i < 3; ++i) o.bounds->lower[i] = 0.0;
    std::size_t outside = 0;
    o.on_evaluation = [&](const flop::Evaluation& e) {
        for (const double v : e.x)
            if (v < 0.0) ++outside;
    };
    const flop::Result r = flop::cobyla::minimize(v0101::hs35_f, v0101::hs35_c, 1, x0, o);
    EXPECT_EQ(outside, 0u);
    EXPECT_NEAR(r.f, 1.0 / 9.0, 1e-9);
    EXPECT_LE(r.max_constraint_violation, 1e-9);
}
