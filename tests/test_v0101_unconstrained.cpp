// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.0.1 test wave: standard unconstrained problems.
//
// Each problem has an optimum in closed form and the test asks for it to
// the accuracy a derivative-free method with linear models can deliver at
// the requested trust radius, plus a converged status, plus a budget that
// was not exhausted. The budgets are wide: a problem that needs the whole
// budget is a finding about the method, and MaxEvaluationsReached is what
// reports it, so the assertion on status is the one that carries the
// information.
//
// The 96-dimensional quadratic is the VQE parameter count and the case the
// library exists for; it is separable, so a coordinate method would solve
// it, and the assertion on its evaluation count is what says the trust
// region step is doing the work rather than the geometry repair.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

#include "flop/flop.hpp"
#include "v0101_problems.hpp"

namespace {

void expect_at(const flop::Result& r, std::span<const double> x_opt, double f_opt, double xtol,
               double ftol) {
    ASSERT_EQ(r.x.size(), x_opt.size());
    EXPECT_LE(v0101::max_abs_diff(r.x, x_opt), xtol);
    EXPECT_NEAR(r.f, f_opt, ftol);
    EXPECT_TRUE(flop::converged(r.status)) << flop::to_string(r.status);
    EXPECT_EQ(r.max_constraint_violation, 0.0);
}

}  // namespace

TEST(V0101Unconstrained, SphereIn2D) {
    const std::vector<double> x0{0.0, 0.0};
    const std::vector<double> x_opt{v0101::sphere_centre(0), v0101::sphere_centre(1)};
    const flop::Result r =
        flop::cobyla::minimize(v0101::sphere, x0, v0101::options(0.3, 1e-9, 1000));
    expect_at(r, x_opt, 0.0, 1e-7, 1e-14);
    EXPECT_LT(r.evaluations, 1000u);
}

TEST(V0101Unconstrained, SphereIn16D) {
    const std::size_t n = 16;
    const std::vector<double> x0(n, 0.0);
    std::vector<double> x_opt(n);
    for (std::size_t i = 0; i < n; ++i) x_opt[i] = v0101::sphere_centre(i);
    const flop::Result r =
        flop::cobyla::minimize(v0101::sphere, x0, v0101::options(0.3, 1e-9, 5000));
    expect_at(r, x_opt, 0.0, 1e-7, 1e-12);
    EXPECT_LT(r.evaluations, 2000u);
}

TEST(V0101Unconstrained, SeparableQuadraticIn96D) {
    const std::size_t n = 96;
    const std::vector<double> x0(n, 0.0);
    std::vector<double> x_opt(n);
    for (std::size_t i = 0; i < n; ++i) x_opt[i] = v0101::sphere_centre(i);
    const flop::Result r =
        flop::cobyla::minimize(v0101::sphere, x0, v0101::options(0.3, 1e-6, 20000));
    expect_at(r, x_opt, 0.0, 1e-4, 1e-6);
    // The optimum is 164 units from x0 and a step is at most 0.3, so 550
    // steps is the floor for any method; 25 evaluations per unit of
    // distance covered leaves room for the simplex maintenance and the
    // final contraction and none for a coordinate sweep.
    EXPECT_LT(r.evaluations, 12000u);
}

TEST(V0101Unconstrained, RosenbrockIn2D) {
    const std::vector<double> x0{-1.2, 1.0};
    const std::vector<double> x_opt{1.0, 1.0};
    const flop::Result r =
        flop::cobyla::minimize(v0101::rosenbrock, x0, v0101::options(0.5, 1e-9, 50000));
    expect_at(r, x_opt, 0.0, 1e-4, 1e-8);
}

TEST(V0101Unconstrained, RosenbrockIn10D) {
    const std::vector<double> x0(10, -1.0);
    const std::vector<double> x_opt(10, 1.0);
    const flop::Result r =
        flop::cobyla::minimize(v0101::rosenbrock, x0, v0101::options(0.5, 1e-9, 200000));
    expect_at(r, x_opt, 0.0, 1e-3, 1e-6);
}

TEST(V0101Unconstrained, PowellSingular) {
    // The singular Hessian defeats the trust-region contraction: linear
    // models see a valley that is flat to first order in two directions,
    // so the radius shrinks slowly and the x tolerance is out of reach in
    // any budget a test can afford. What is held is the value the budget
    // buys: f is a quartic in the distance along the flat directions, so
    // f below 1e-6 is x within a few hundredths, and the run is expected
    // to be still going when the cap lands.
    const std::vector<double> x0{3.0, -1.0, 0.0, 1.0};
    const std::vector<double> x_opt(4, 0.0);
    const flop::Result r =
        flop::cobyla::minimize(v0101::powell_singular, x0, v0101::options(0.5, 1e-9, 50000));
    EXPECT_LE(r.f, 1e-6);
    EXPECT_LE(v0101::max_abs_diff(r.x, x_opt), 0.05);
    EXPECT_EQ(r.max_constraint_violation, 0.0);
}

TEST(V0101Unconstrained, Beale) {
    const std::vector<double> x0{1.0, 1.0};
    const std::vector<double> x_opt{3.0, 0.5};
    const flop::Result r =
        flop::cobyla::minimize(v0101::beale, x0, v0101::options(0.5, 1e-9, 50000));
    expect_at(r, x_opt, 0.0, 1e-4, 1e-8);
}

TEST(V0101Unconstrained, OneDimension) {
    // n = 1 is the degenerate simplex (a segment); the cosine has its
    // minimum at pi.
    auto f = [](std::span<const double> x) { return std::cos(x[0]); };
    const std::vector<double> x0{1.603};
    const std::vector<double> x_opt{std::numbers::pi};
    const flop::Result r = flop::cobyla::minimize(f, x0, v0101::options(1.0, 1e-9, 500));
    expect_at(r, x_opt, -1.0, 1e-6, 1e-12);
}

TEST(V0101Unconstrained, StartingAtTheOptimumStaysThere) {
    const std::vector<double> x0{v0101::sphere_centre(0), v0101::sphere_centre(1)};
    const flop::Result r =
        flop::cobyla::minimize(v0101::sphere, x0, v0101::options(0.3, 1e-9, 1000));
    expect_at(r, x0, 0.0, 1e-8, 1e-15);
}

TEST(V0101Unconstrained, TheReturnedFIsTheObjectiveAtTheReturnedX) {
    const std::vector<double> x0{-1.2, 1.0};
    const flop::Result r =
        flop::cobyla::minimize(v0101::rosenbrock, x0, v0101::options(0.5, 1e-6, 2000));
    EXPECT_TRUE(v0101::same_bits(r.f, v0101::rosenbrock(r.x)));
}
