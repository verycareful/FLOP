// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.0.1 test wave: constrained problems.
//
// Powell's 1994 test set from the paper's own starting points, then two
// Hock-Schittkowski problems whose inequality constraints are active at
// the optimum and which also carry x >= 0 as a box. Every optimum is a
// closed form derived in v0101_problems.hpp (HS100 excepted, whose optimum
// is known to seven digits and is held to those).
//
// A constrained optimum is asked for three things: the objective, the
// point, and feasibility of the returned point to the accuracy the trust
// radius can deliver. The last is reported by max_constraint_violation and
// is asserted separately because an optimizer can be at the right f by
// standing slightly outside the feasible set.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

#include "flop/flop.hpp"
#include "v0101_problems.hpp"

namespace {

// The paper's runs use rhobeg 0.5 on every problem here. Its rhoend is
// 1e-6; the runs below go to 1e-9 (xtol_rel 2e-9 on the step of 0.5), which
// is what lets the point be held to 1e-5 on the curved problems.
constexpr double kStep = 0.5;

void check(const v0101::Constrained& p, double xtol, double ftol, double vtol) {
    SCOPED_TRACE(p.name);
    const flop::Result r =
        p.m ? flop::cobyla::minimize(p.f, p.c, p.m, p.x0, v0101::options(kStep, 2e-9, 20000))
            : flop::cobyla::minimize(p.f, p.x0, v0101::options(kStep, 2e-9, 20000));
    ASSERT_EQ(r.x.size(), p.x_opt.size());
    EXPECT_TRUE(flop::converged(r.status)) << flop::to_string(r.status);
    EXPECT_NEAR(r.f, p.f_opt, ftol);
    EXPECT_LE(v0101::max_abs_diff(r.x, p.x_opt), xtol);
    EXPECT_LE(r.max_constraint_violation, vtol);
}

}  // namespace

TEST(V0101Constrained, Powell1994Problem1SimpleQuadratic) {
    check(v0101::powell_1994_problems()[0], 1e-6, 1e-12, 0.0);
}

TEST(V0101Constrained, Powell1994Problem2UnitCircle) {
    // Two optima by symmetry (x1, x2) and (-x1, -x2); from (1, 1) the run
    // reaches the one with x1 > 0, and the point check allows either.
    const v0101::Constrained p = v0101::powell_1994_problems()[1];
    const flop::Result r =
        flop::cobyla::minimize(p.f, p.c, p.m, p.x0, v0101::options(kStep, 2e-9, 20000));
    EXPECT_TRUE(flop::converged(r.status)) << flop::to_string(r.status);
    // The run ends within the final radius (1e-9) of the circle, on either
    // side of it, so f and the violation are held to ten times that.
    EXPECT_NEAR(r.f, p.f_opt, 1e-8);
    const std::vector<double> mirror{-p.x_opt[0], -p.x_opt[1]};
    EXPECT_TRUE(v0101::max_abs_diff(r.x, p.x_opt) < 1e-5 ||
                v0101::max_abs_diff(r.x, mirror) < 1e-5);
    EXPECT_LE(r.max_constraint_violation, 1e-8);
}

TEST(V0101Constrained, Powell1994Problem3Ellipsoid) {
    // The sign pattern of the optimum is any with an odd number of negative
    // coordinates; the magnitudes are what is checked.
    const v0101::Constrained p = v0101::powell_1994_problems()[2];
    const flop::Result r =
        flop::cobyla::minimize(p.f, p.c, p.m, p.x0, v0101::options(kStep, 2e-9, 20000));
    EXPECT_TRUE(flop::converged(r.status)) << flop::to_string(r.status);
    EXPECT_NEAR(r.f, p.f_opt, 1e-10);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(std::fabs(r.x[i]), std::fabs(p.x_opt[i]), 1e-5) << i;
    EXPECT_LE(r.max_constraint_violation, 1e-10);
}

TEST(V0101Constrained, Powell1994Problem4WeakRosenbrock) {
    check(v0101::powell_1994_problems()[3], 1e-4, 1e-8, 0.0);
}

TEST(V0101Constrained, Powell1994Problem5IntermediateRosenbrock) {
    check(v0101::powell_1994_problems()[4], 1e-4, 1e-8, 0.0);
}

TEST(V0101Constrained, Powell1994Problem6Fletcher9115) {
    check(v0101::powell_1994_problems()[5], 1e-5, 1e-9, 1e-10);
}

TEST(V0101Constrained, Powell1994Problem7Fletcher1442) {
    check(v0101::powell_1994_problems()[6], 1e-5, 1e-8, 1e-9);
}

TEST(V0101Constrained, Powell1994Problem8RosenSuzuki) {
    check(v0101::powell_1994_problems()[7], 1e-5, 1e-8, 1e-9);
}

TEST(V0101Constrained, Powell1994Problem9HockSchittkowski100) {
    // The reference point carries seven digits, so 1e-5 on x is the
    // reference's own precision, and 1e-6 on f is the same relative to 680.
    check(v0101::powell_1994_problems()[8], 1e-5, 1e-6, 1e-8);
}

TEST(V0101Constrained, HS24TwoActiveConstraintsAndABox) {
    const std::vector<double> x0{1.0, 0.5};
    const std::vector<double> x_opt{3.0, std::numbers::sqrt3};
    flop::cobyla::Options o = v0101::options(kStep, 2e-9, 20000);
    o.bounds = flop::Bounds::none(2);
    o.bounds->lower[0] = 0.0;
    o.bounds->lower[1] = 0.0;
    const flop::Result r = flop::cobyla::minimize(v0101::hs24_f, v0101::hs24_c, 3, x0, o);
    EXPECT_TRUE(flop::converged(r.status)) << flop::to_string(r.status);
    EXPECT_NEAR(r.f, -1.0, 1e-8);
    EXPECT_LE(v0101::max_abs_diff(r.x, x_opt), 1e-5);
    EXPECT_LE(r.max_constraint_violation, 1e-9);
    EXPECT_GE(r.x[0], 0.0);
    EXPECT_GE(r.x[1], 0.0);
}

TEST(V0101Constrained, HS35ConvexQuadraticOnAnActiveConstraint) {
    const std::vector<double> x0{0.5, 0.5, 0.5};
    const std::vector<double> x_opt{4.0 / 3.0, 7.0 / 9.0, 4.0 / 9.0};
    flop::cobyla::Options o = v0101::options(kStep, 2e-9, 20000);
    o.bounds = flop::Bounds::none(3);
    for (std::size_t i = 0; i < 3; ++i) o.bounds->lower[i] = 0.0;
    const flop::Result r = flop::cobyla::minimize(v0101::hs35_f, v0101::hs35_c, 1, x0, o);
    EXPECT_TRUE(flop::converged(r.status)) << flop::to_string(r.status);
    EXPECT_NEAR(r.f, 1.0 / 9.0, 1e-9);
    EXPECT_LE(v0101::max_abs_diff(r.x, x_opt), 1e-5);
    EXPECT_LE(r.max_constraint_violation, 1e-9);
}

TEST(V0101Constrained, AnInfeasibleStartEndsFeasible) {
    // (x1 - 2)^2 + (x2 - 1)^2 under x1 <= 1 and x2 >= x1, from (3, -2)
    // where both constraints are violated; the optimum is (1, 1), f = 1.
    auto f = [](std::span<const double> x) {
        return (x[0] - 2.0) * (x[0] - 2.0) + (x[1] - 1.0) * (x[1] - 1.0);
    };
    auto c = [](std::span<const double> x, std::span<double> out) {
        out[0] = 1.0 - x[0];
        out[1] = x[1] - x[0];
    };
    const std::vector<double> x0{3.0, -2.0};
    const std::vector<double> x_opt{1.0, 1.0};
    const flop::Result r = flop::cobyla::minimize(f, c, 2, x0, v0101::options(kStep, 2e-9, 5000));
    EXPECT_TRUE(flop::converged(r.status)) << flop::to_string(r.status);
    EXPECT_NEAR(r.f, 1.0, 1e-8);
    EXPECT_LE(v0101::max_abs_diff(r.x, x_opt), 1e-5);
    EXPECT_LE(r.max_constraint_violation, 1e-8);
}

TEST(V0101Constrained, TheViolationReportedIsTheViolationAtTheReturnedPoint) {
    // Every evaluation is recorded as it happens through on_evaluation, and
    // the returned x is matched against the records bit for bit. That checks
    // two things: the returned point is one the optimizer evaluated, and the
    // violation it reports is that point's. The violation is rebuilt from the
    // recorded c(x), so the pin is on a copy of the library's own numbers
    // and not on a second evaluation of the constraint polynomials, whose
    // rounding a compiler may order differently at each call site.
    //
    // At the returned point the constraint values sit at the rounding floor
    // and one is often exactly zero, so the rebuilt maximum admits only
    // strictly violated constraints: every value it compares is strictly
    // positive, and a feasible point yields the literal 0.0. A maximum that
    // let -0.0 in would not do, because under -fno-signed-zeros a compiler
    // may compile a compare-and-assign as a max instruction that returns
    // either zero, while the library reports +0.0 by bits.
    const v0101::Constrained p = v0101::powell_1994_problems()[7];
    struct Record {
        std::vector<double> x;
        std::vector<double> c;
    };
    std::vector<Record> records;
    flop::cobyla::Options o = v0101::options(kStep, 2e-9, 20000);
    o.on_evaluation = [&records](const flop::Evaluation& ev) {
        records.push_back({.x = std::vector<double>(ev.x.begin(), ev.x.end()),
                           .c = std::vector<double>(ev.constraints.begin(), ev.constraints.end())});
    };
    const flop::Result r = flop::cobyla::minimize(p.f, p.c, p.m, p.x0, o);
    const auto hit = std::ranges::find_if(
        records, [&r](const Record& rec) { return v0101::same_bits(rec.x, r.x); });
    ASSERT_NE(hit, records.end()) << "the returned point was never evaluated";
    bool violated = false;
    double worst = 0.0;
    for (const double v : hit->c) {
        if (v < 0.0 && (!violated || -v > worst)) {
            worst = -v;
            violated = true;
        }
    }
    const double viol = violated ? worst : 0.0;
    EXPECT_TRUE(v0101::same_bits(r.max_constraint_violation, viol))
        << "reported " << r.max_constraint_violation << ", at the point " << viol;
}

TEST(V0101Constrained, AnInactiveConstraintChangesNothingButTheCount) {
    // The sphere with a constraint that holds everywhere near the path:
    // the run must reach the same optimum, and the constraint function
    // must have been called exactly once per evaluation.
    std::size_t f_calls = 0, c_calls = 0;
    auto f = [&](std::span<const double> x) {
        ++f_calls;
        return v0101::sphere(x);
    };
    auto c = [&](std::span<const double>, std::span<double> out) {
        ++c_calls;
        out[0] = 1.0;
    };
    const std::vector<double> x0{0.0, 0.0};
    const std::vector<double> x_opt{v0101::sphere_centre(0), v0101::sphere_centre(1)};
    const flop::Result r = flop::cobyla::minimize(f, c, 1, x0, v0101::options(0.3, 1e-9, 2000));
    EXPECT_TRUE(flop::converged(r.status)) << flop::to_string(r.status);
    EXPECT_LE(v0101::max_abs_diff(r.x, x_opt), 1e-7);
    EXPECT_EQ(f_calls, c_calls);
    EXPECT_EQ(f_calls, r.evaluations);
}
