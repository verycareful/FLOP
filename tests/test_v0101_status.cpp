// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.0.1 test wave: Status semantics and the evaluation trace.
//
// Each Status value is produced on a problem built to produce it and
// nothing else, and converged() is checked on every value, since the one
// decision every caller makes is whether to trust the point. The trace is
// here because it is what the status tests use to see the run: every
// evaluation once, in order, with an index equal to its position, the
// point, the value the objective returned, and the constraint values.

#include <gtest/gtest.h>

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

#include "flop/flop.hpp"
#include "v0101_problems.hpp"

TEST(V0101Status, ConvergedIsTrueForTheThreeTolerancesAndFalseOtherwise) {
    EXPECT_TRUE(flop::converged(flop::Status::XtolReached));
    EXPECT_TRUE(flop::converged(flop::Status::FtolReached));
    EXPECT_TRUE(flop::converged(flop::Status::StopValueReached));
    EXPECT_FALSE(flop::converged(flop::Status::MaxEvaluationsReached));
    EXPECT_FALSE(flop::converged(flop::Status::RoundoffLimited));
}

TEST(V0101Status, EveryValueHasAName) {
    using flop::Status;
    EXPECT_EQ(flop::to_string(Status::XtolReached), std::string_view("XtolReached"));
    EXPECT_EQ(flop::to_string(Status::FtolReached), std::string_view("FtolReached"));
    EXPECT_EQ(flop::to_string(Status::StopValueReached), std::string_view("StopValueReached"));
    EXPECT_EQ(flop::to_string(Status::MaxEvaluationsReached),
              std::string_view("MaxEvaluationsReached"));
    EXPECT_EQ(flop::to_string(Status::RoundoffLimited), std::string_view("RoundoffLimited"));
}

TEST(V0101Status, XtolReachedWhenTheRadiusReachesTheCallersFloor) {
    const std::vector<double> x0{0.0, 0.0};
    flop::cobyla::Options o;
    o.initial_step = 0.3;
    o.stopping.xtol_rel = 1e-6;
    const flop::Result r = flop::cobyla::minimize(v0101::sphere, x0, o);
    EXPECT_EQ(r.status, flop::Status::XtolReached);
    // The final radius is the floor the caller asked for, rho_end =
    // xtol_rel * initial_step, to the rounding of the halvings that reach it.
    EXPECT_NEAR(r.final_trust_radius, 1e-6 * 0.3, 1e-6 * 0.3 * 0.5);
}

TEST(V0101Status, XtolAbsIsAnAbsoluteFloorOnTheRadius) {
    const std::vector<double> x0{0.0, 0.0};
    flop::cobyla::Options o;
    o.initial_step = 0.3;
    o.stopping.xtol_abs = 1e-4;
    const flop::Result r = flop::cobyla::minimize(v0101::sphere, x0, o);
    EXPECT_EQ(r.status, flop::Status::XtolReached);
    EXPECT_NEAR(r.final_trust_radius, 1e-4, 0.5e-4);
}

TEST(V0101Status, TheLargerOfTheTwoXTolerancesWins) {
    const std::vector<double> x0{0.0, 0.0};
    flop::cobyla::Options o;
    o.initial_step = 0.3;
    o.stopping.xtol_rel = 1e-9;  // 3e-10 absolute
    o.stopping.xtol_abs = 1e-3;
    const flop::Result r = flop::cobyla::minimize(v0101::sphere, x0, o);
    EXPECT_EQ(r.status, flop::Status::XtolReached);
    EXPECT_NEAR(r.final_trust_radius, 1e-3, 0.5e-3);
}

TEST(V0101Status, FinalTrustRadiusOverridesTheTolerances) {
    const std::vector<double> x0{0.0, 0.0};
    flop::cobyla::Options o;
    o.initial_step = 0.3;
    o.stopping.xtol_rel = 1e-9;
    o.final_trust_radius = 1e-2;
    const flop::Result r = flop::cobyla::minimize(v0101::sphere, x0, o);
    EXPECT_EQ(r.status, flop::Status::XtolReached);
    EXPECT_NEAR(r.final_trust_radius, 1e-2, 0.5e-2);
}

TEST(V0101Status, FtolReachedWhenAnAcceptedStepChangesFByLessThanTheTolerance) {
    const std::vector<double> x0{0.0, 0.0};
    std::vector<double> best;
    flop::cobyla::Options o;
    o.initial_step = 0.3;
    o.stopping.ftol_abs = 1e-4;
    o.on_evaluation = [&](const flop::Evaluation& e) {
        if (best.empty() || e.f < best.back()) best.push_back(e.f);
    };
    const flop::Result r = flop::cobyla::minimize(v0101::sphere, x0, o);
    EXPECT_EQ(r.status, flop::Status::FtolReached);
    EXPECT_TRUE(flop::converged(r.status));
    ASSERT_GE(best.size(), 2u);
    // The last improvement of the best value was smaller than the tolerance.
    EXPECT_LE(best[best.size() - 2] - best.back(), 1e-4);
}

TEST(V0101Status, FtolRelIsRelativeToTheValue) {
    // A sphere lifted by 1000: an absolute tolerance of 1e-4 would need
    // the same absolute fall, a relative one of 1e-7 stops at a fall of
    // about 1e-4 in a value near 1000.
    auto f = [](std::span<const double> x) { return 1000.0 + v0101::sphere(x); };
    const std::vector<double> x0{0.0, 0.0};
    flop::cobyla::Options o;
    o.initial_step = 0.3;
    o.stopping.ftol_rel = 1e-7;
    const flop::Result r = flop::cobyla::minimize(f, x0, o);
    EXPECT_EQ(r.status, flop::Status::FtolReached);
    EXPECT_NEAR(r.f, 1000.0, 1e-3);
}

TEST(V0101Status, StopValueReachedAtTheFirstEvaluationAtOrBelowIt) {
    const std::vector<double> x0{0.0, 0.0};
    std::vector<double> seen;
    flop::cobyla::Options o;
    o.initial_step = 0.3;
    o.stopping.stop_value = 0.1;
    o.stopping.max_evaluations = 1000;
    o.on_evaluation = [&](const flop::Evaluation& e) { seen.push_back(e.f); };
    const flop::Result r = flop::cobyla::minimize(v0101::sphere, x0, o);
    EXPECT_EQ(r.status, flop::Status::StopValueReached);
    EXPECT_TRUE(flop::converged(r.status));
    EXPECT_LE(r.f, 0.1);
    // Nothing before the last evaluation was at or below the value, and the
    // last one is the point returned.
    ASSERT_EQ(seen.size(), r.evaluations);
    for (std::size_t k = 0; k + 1 < seen.size(); ++k) EXPECT_GT(seen[k], 0.1) << k;
    EXPECT_TRUE(v0101::same_bits(seen.back(), r.f));
}

TEST(V0101Status, StopValueAlreadyMetAtX0StopsAfterOneEvaluation) {
    const std::vector<double> x0{v0101::sphere_centre(0), v0101::sphere_centre(1)};
    flop::cobyla::Options o;
    o.stopping.stop_value = 1.0;
    const flop::Result r = flop::cobyla::minimize(v0101::sphere, x0, o);
    EXPECT_EQ(r.status, flop::Status::StopValueReached);
    EXPECT_EQ(r.evaluations, 1u);
    EXPECT_TRUE(v0101::same_bits(r.x, x0));
}

TEST(V0101Status, MaxEvaluationsReachedIsExactAndNotConverged) {
    for (const std::size_t cap : {1u, 2u, 3u, 10u, 57u}) {
        const std::vector<double> x0{0.0, 0.0, 0.0};
        std::size_t calls = 0;
        auto f = [&](std::span<const double> x) {
            ++calls;
            return v0101::sphere(x);
        };
        flop::cobyla::Options o = v0101::options(0.3, 1e-9, cap);
        const flop::Result r = flop::cobyla::minimize(f, x0, o);
        EXPECT_EQ(r.status, flop::Status::MaxEvaluationsReached) << cap;
        EXPECT_FALSE(flop::converged(r.status));
        EXPECT_EQ(r.evaluations, cap);
        EXPECT_EQ(calls, cap);
    }
}

TEST(V0101Status, TheCapReturnsTheBestPointSeen) {
    const std::vector<double> x0{0.0, 0.0, 0.0};
    double best = 0.0;
    bool have = false;
    flop::cobyla::Options o = v0101::options(0.3, 1e-9, 25);
    o.on_evaluation = [&](const flop::Evaluation& e) {
        if (!have || e.f < best) {
            best = e.f;
            have = true;
        }
    };
    const flop::Result r = flop::cobyla::minimize(v0101::sphere, x0, o);
    EXPECT_EQ(r.status, flop::Status::MaxEvaluationsReached);
    EXPECT_TRUE(v0101::same_bits(r.f, best));
    EXPECT_TRUE(v0101::same_bits(r.f, v0101::sphere(r.x)));
}

TEST(V0101Status, RoundoffLimitedWhenNoXToleranceStopsTheRadiusFirst) {
    // Only a cap, and a wide one: the radius halves until the simplex can
    // no longer be told from a point, which is a floor the method has to
    // recognise on its own.
    const std::vector<double> x0{0.0, 0.0};
    flop::cobyla::Options o;
    o.initial_step = 0.3;
    o.stopping.max_evaluations = 100000;
    const flop::Result r = flop::cobyla::minimize(v0101::sphere, x0, o);
    EXPECT_EQ(r.status, flop::Status::RoundoffLimited);
    EXPECT_FALSE(flop::converged(r.status));
    EXPECT_LT(r.evaluations, 100000u);
    const std::vector<double> x_opt{v0101::sphere_centre(0), v0101::sphere_centre(1)};
    EXPECT_LE(v0101::max_abs_diff(r.x, x_opt), 1e-7);
}

TEST(V0101Status, TheTraceSeesEveryEvaluationOnceInOrderWithItsValue) {
    const std::vector<double> x0{0.0, 0.0, 0.0};
    std::vector<std::size_t> indices;
    std::vector<std::vector<double>> points;
    std::vector<double> values;
    flop::cobyla::Options o = v0101::options(0.3, 1e-6, 500);
    o.on_evaluation = [&](const flop::Evaluation& e) {
        indices.push_back(e.index);
        points.emplace_back(e.x.begin(), e.x.end());
        values.push_back(e.f);
        EXPECT_TRUE(e.constraints.empty());
    };
    const flop::Result r = flop::cobyla::minimize(v0101::sphere, x0, o);
    ASSERT_EQ(indices.size(), r.evaluations);
    for (std::size_t k = 0; k < indices.size(); ++k) {
        EXPECT_EQ(indices[k], k);
        EXPECT_EQ(points[k].size(), 3u);
        EXPECT_TRUE(v0101::same_bits(values[k], v0101::sphere(points[k]))) << k;
    }
}

TEST(V0101Status, TheTraceCarriesTheConstraintValues) {
    const v0101::Constrained p = v0101::powell_1994_problems()[7];  // Rosen-Suzuki, m = 3
    std::size_t seen = 0;
    flop::cobyla::Options o = v0101::options(0.5, 1e-6, 500);
    o.on_evaluation = [&](const flop::Evaluation& e) {
        ++seen;
        ASSERT_EQ(e.constraints.size(), p.m);
        std::vector<double> cv(p.m);
        p.c(e.x, cv);
        EXPECT_TRUE(v0101::same_bits(e.constraints, cv));
    };
    const flop::Result r = flop::cobyla::minimize(p.f, p.c, p.m, p.x0, o);
    EXPECT_EQ(seen, r.evaluations);
}
