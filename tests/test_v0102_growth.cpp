// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.0.2: the trust radius that grows, and the paper's radius that does not.
//
// The paper's rho is never increased, so along a valley the method crawls
// at rho per evaluation once the simplex has flattened onto the floor: each
// step is then a perfect step of length rho with ratio 1, nothing fails,
// and rho stays. FLOP's trust radius Delta doubles after a very successful
// step and shrinks back to rho on a failure, so the crawl lengthens its
// stride; rho keeps the paper's schedule as Delta's floor.
// trust_region_growth = false pins Delta to rho and gives the paper's
// method, which stays available and is held here beside the other.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "flop/flop.hpp"
#include "v0101_problems.hpp"

namespace {

flop::cobyla::Options valley_options(bool growth, std::size_t cap) {
    flop::cobyla::Options o = v0101::options(0.5, 1e-9, cap);
    o.trust_region_growth = growth;
    return o;
}

// The longest distance between two consecutive evaluated points, in units
// of the initial step: with the paper's radius no trial point is further
// than one radius from a vertex, so this stays at or below 1 + 1 = 2 (a
// step from the base after the base moved by a step).
double longest_stride(bool growth, std::size_t cap) {
    std::vector<double> previous;
    double longest = 0.0;
    flop::cobyla::Options o = valley_options(growth, cap);
    o.on_evaluation = [&](const flop::Evaluation& e) {
        if (!previous.empty()) longest = std::max(longest, v0101::max_abs_diff(e.x, previous));
        previous.assign(e.x.begin(), e.x.end());
    };
    const std::vector<double> x0{-1.2, 1.0};
    (void)flop::cobyla::minimize(v0101::rosenbrock, x0, o);
    return longest / 0.5;
}

TEST(V0102Growth, GrowthIsOnByDefault) {
    const flop::cobyla::Options o;
    EXPECT_TRUE(o.trust_region_growth);
}

// Rosenbrock 2-d from (-1.2, 1) at a cap of 5000: the paper's method is
// still on the valley floor with a gap above 1e-3, the growing radius is
// below 1e-4.
TEST(V0102Growth, TheGrowingRadiusLeavesThePaperBehindOnAValley) {
    const std::vector<double> x0{-1.2, 1.0};
    const flop::Result on =
        flop::cobyla::minimize(v0101::rosenbrock, x0, valley_options(true, 5000));
    const flop::Result off =
        flop::cobyla::minimize(v0101::rosenbrock, x0, valley_options(false, 5000));
    EXPECT_LT(on.f, 1e-4);
    EXPECT_LT(on.f, off.f);
}

// With growth off the radius is the paper's: no stride between consecutive
// evaluations exceeds two initial steps once the simplex is built. With
// growth on, strides beyond that appear, which is the whole point.
TEST(V0102Growth, GrowthOffKeepsEveryStrideWithinThePapersRadius) {
    EXPECT_LE(longest_stride(false, 2000), 2.0 + 1e-12);
}

// Growth never takes the radius past the initial step: the caller's scale
// for the problem is the cap.
TEST(V0102Growth, GrowthNeverPassesTheInitialStep) {
    EXPECT_LE(longest_stride(true, 2000), 2.0 + 1e-12);
}

// A problem with no valley: the sphere. Growth must not cost anything where
// the paper's radius was already right: both converge, to the same optimum,
// and the grown radius does not take more evaluations.
TEST(V0102Growth, GrowthDoesNotHurtWhereTheRadiusWasRight) {
    const std::vector<double> x0(16, 0.0);
    flop::cobyla::Options on = v0101::options(0.3, 1e-9, 5000);
    flop::cobyla::Options off = on;
    off.trust_region_growth = false;
    const flop::Result a = flop::cobyla::minimize(v0101::sphere, x0, on);
    const flop::Result b = flop::cobyla::minimize(v0101::sphere, x0, off);
    EXPECT_EQ(a.status, flop::Status::XtolReached);
    EXPECT_EQ(b.status, flop::Status::XtolReached);
    EXPECT_LT(a.f, 1e-12);
    EXPECT_LT(b.f, 1e-12);
    EXPECT_LE(a.evaluations, b.evaluations);
}

// The grown radius on a constrained problem: HS24 still ends on its
// constraints at the same optimum.
TEST(V0102Growth, GrowthKeepsAConstrainedOptimum) {
    const std::vector<double> x0{1.0, 0.5};
    flop::cobyla::Options o = v0101::options(0.5, 1e-9, 5000);
    o.trust_region_growth = true;
    const flop::Result r = flop::cobyla::minimize(v0101::hs24_f, v0101::hs24_c, 3, x0, o);
    EXPECT_NEAR(r.f, -1.0, 1e-6);
    EXPECT_LE(r.max_constraint_violation, 1e-8);
}

}  // namespace
