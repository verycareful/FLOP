// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: why a BOBYQA run stops, and what it returns.
//
// docs/algorithms/bobyqa.md, "Options" and "Status values": rho_end is
// final_trust_radius when that is positive, else the larger of xtol_abs and
// xtol_rel * initial_step, never above initial_step, with a floor of
// epsilon * initial_step below which the points cannot be told apart. Reaching
// the caller's own rho_end is XtolReached; reaching the floor is
// RoundoffLimited; Result::final_radius is rho at the end. StopValueReached is
// met by the evaluation that reaches it, and MaxEvaluationsReached is exact:
// a run capped at c is the first c evaluations of the run without the cap.
// Whatever the status, the result is the best point evaluated with its own
// value. The FtolReached rule is held in the tolerance suite.
//
// The error return of (6.3), a predicted reduction that is not positive when
// a step has been taken, is not reached by a problem a test can build:
// a model with no descent gives a step of zero, which is a short step.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
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

double rosenbrock3(std::span<const double> x) {
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

constexpr double kEps = std::numeric_limits<double>::epsilon();

}  // namespace

TEST(V0121Status, RhoEndIsFinalTrustRadiusOrTheLargerOfTheXTolerancesNeverAboveTheFirstRadius) {
    struct Case {
        double final_radius, xtol_abs, xtol_rel, step;
        double want;
    };
    for (const Case& c :
         {Case{.final_radius = 0.0, .xtol_abs = 1e-5, .xtol_rel = 0.0, .step = 0.5, .want = 1e-5},
          Case{.final_radius = 0.0, .xtol_abs = 0.0, .xtol_rel = 1e-5, .step = 0.5, .want = 5e-6},
          Case{.final_radius = 0.0, .xtol_abs = 1e-5, .xtol_rel = 1e-4, .step = 0.5, .want = 5e-5},
          Case{.final_radius = 0.0, .xtol_abs = 1e-3, .xtol_rel = 1e-9, .step = 0.5, .want = 1e-3},
          Case{.final_radius = 2e-5, .xtol_abs = 1e-2, .xtol_rel = 1e-2, .step = 0.5, .want = 2e-5},
          Case{.final_radius = 3e-6, .xtol_abs = 0.0, .xtol_rel = 0.0, .step = 0.5, .want = 3e-6},
          Case{.final_radius = 0.0, .xtol_abs = 9.0, .xtol_rel = 0.0, .step = 0.5, .want = 0.5},
          Case{.final_radius = 0.0, .xtol_abs = 0.0, .xtol_rel = 4.0, .step = 0.5, .want = 0.5}}) {
        for (const bool batch : {false, true}) {
            flop::bobyqa::Options o = v0121::options(c.step, 5000);
            o.final_trust_radius = c.final_radius;
            o.stopping.xtol_abs = c.xtol_abs;
            o.stopping.xtol_rel = c.xtol_rel;
            const Outcome t = run(quadratic, Vec(3, 0.1), o, batch);
            EXPECT_EQ(t.r.status, flop::Status::XtolReached)
                << "want " << c.want << " batch " << batch;
            EXPECT_EQ(t.r.final_radius, c.want) << "the radius reached is rho_end exactly";
        }
    }
}

TEST(V0121Status, TheFloorIsEpsilonTimesTheFirstRadiusAndReachingItIsRoundoffLimited) {
    // No x tolerance, or one below the floor: the run cannot tell its points
    // apart below epsilon * initial_step and ends there.
    for (const double step : {0.5, 8.0, 0.001}) {
        for (const double xtol : {0.0, 1e-300}) {
            flop::bobyqa::Options o = v0121::options(step, 100000);
            o.stopping.xtol_abs = xtol;
            for (const bool batch : {false, true}) {
                const Outcome t = run(quadratic, Vec{0.1, 0.1}, o, batch);
                EXPECT_EQ(t.r.status, flop::Status::RoundoffLimited)
                    << "step " << step << " xtol " << xtol;
                EXPECT_EQ(t.r.final_radius, step * kEps) << "step " << step;
                EXPECT_LT(t.r.evaluations, 400u);
            }
        }
    }
}

TEST(V0121Status, AConstantObjectiveEndsAtRhoEndOrAtTheFloor) {
    // Nothing to descend: the model is flat, every step is short, and rho falls
    // by (6.6) to rho_end, or to the floor when there is no x tolerance.
    flop::bobyqa::Options o = v0121::options(0.5, 1000);
    o.stopping.xtol_abs = 1e-6;
    auto flat = [](std::span<const double>) { return 3.0; };
    const Outcome a = run(flat, Vec{0.1, 0.1}, o);
    EXPECT_EQ(a.r.status, flop::Status::XtolReached);
    EXPECT_EQ(a.r.final_radius, 1e-6);
    EXPECT_EQ(a.r.f, 3.0);
    EXPECT_TRUE(v0121::same_bits(a.r.x, Vec{0.1, 0.1})) << "the first point evaluated is the best";
    o.stopping.xtol_abs = 0.0;
    const Outcome b = run(flat, Vec{0.1, 0.1}, o);
    EXPECT_EQ(b.r.status, flop::Status::RoundoffLimited);
    EXPECT_EQ(b.r.final_radius, 0.5 * kEps);
}

TEST(V0121Status, StopValueEndsTheRunAtTheEvaluationThatMeetsItAfterTheStartingPoints) {
    // A stop value met by an evaluation after the starting points: the scalar
    // run ends at it, every earlier value is above it; the batch run ends at it
    // too, and the call that holds it is counted whole.
    const Vec x0{0.1, 0.1};
    for (const double stop : {1e-3, 1e-6}) {
        flop::bobyqa::Options o = v0121::options(0.5, 1000);
        o.stopping.stop_value = stop;
        const Outcome t = run(quadratic, x0, o);
        ASSERT_GT(t.trace.size(), 5u);
        EXPECT_EQ(t.r.status, flop::Status::StopValueReached) << stop;
        EXPECT_EQ(t.r.evaluations, t.trace.size());
        EXPECT_LE(t.trace.back().f, stop);
        for (std::size_t i = 0; i + 1 < t.trace.size(); ++i) EXPECT_GT(t.trace[i].f, stop);
        EXPECT_LE(t.r.f, stop);
        const Outcome b = run(quadratic, x0, o, true);
        EXPECT_EQ(b.r.status, flop::Status::StopValueReached) << stop;
        EXPECT_EQ(b.r.evaluations, b.trace.size());
        EXPECT_LE(b.r.f, stop);
    }
}

TEST(V0121Status, StopValueEndsAScalarRunAtTheStartingPointThatMeetsIt) {
    // Red on purpose: the scalar channel evaluates all 2n + 1 starting points
    // even when the second of them is at or below the stop value, because the
    // loop over them does not look at the stop; the batch channel does that
    // by design (the call is made whole) but the scalar one has no call to
    // make whole. docs/algorithms/bobyqa.md says the run stops on stop_value
    // after every evaluation.
    const Vec x0{0.1, 0.1};
    for (const double stop : {0.2, 0.05, quadratic(x0)}) {
        flop::bobyqa::Options o = v0121::options(0.5, 1000);
        o.stopping.stop_value = stop;
        const Outcome t = run(quadratic, x0, o);
        std::size_t first = t.trace.size();
        for (std::size_t i = 0; i < t.trace.size(); ++i)
            if (t.trace[i].f <= stop) {
                first = i;
                break;
            }
        ASSERT_LT(first, t.trace.size()) << stop;
        EXPECT_EQ(t.trace.size(), first + 1) << "stop value " << stop;
        EXPECT_EQ(t.r.status, flop::Status::StopValueReached);
    }
    // The batch run counts the whole first call.
    flop::bobyqa::Options o = v0121::options(0.5, 1000);
    o.stopping.stop_value = 0.2;
    const Outcome b = run(quadratic, x0, o, true);
    EXPECT_EQ(b.trace.size(), 5u);
    EXPECT_EQ(b.r.status, flop::Status::StopValueReached);
    EXPECT_LE(b.r.f, 0.2);
}

TEST(V0121Status, EveryCapGivesAPrefixOfTheUncappedRunAndExactlyThatManyEvaluations) {
    // One run ends by the radius and one by the floor, after calls of RESCUE:
    // for every cap below the length of the uncapped run, the capped run is the
    // first c evaluations of it, ends MaxEvaluationsReached, and has c of them.
    struct Problem {
        double (*f)(std::span<const double>);
        Vec x0;
        double xtol;
    };
    const Problem problems[] = {{.f = rosenbrock3, .x0 = {-1.2, 1.0, 1.0}, .xtol = 1e-7},
                                {.f = quadratic, .x0 = {0.1, 0.1}, .xtol = 1e-12}};
    for (const Problem& p : problems) {
        flop::bobyqa::Options o = v0121::options(0.5, 100000);
        o.stopping.xtol_abs = p.xtol;
        const Outcome whole = run(p.f, p.x0, o);
        const std::size_t total = whole.trace.size();
        ASSERT_GT(total, 20u);
        for (std::size_t cap = 1; cap < total; ++cap) {
            flop::bobyqa::Options c = o;
            c.stopping.max_evaluations = cap;
            const Outcome t = run(p.f, p.x0, c);
            ASSERT_EQ(t.trace.size(), cap) << "cap " << cap;
            EXPECT_EQ(t.r.evaluations, cap);
            EXPECT_EQ(t.r.status, flop::Status::MaxEvaluationsReached) << "cap " << cap;
            for (std::size_t i = 0; i < cap; ++i) {
                if (!v0121::same_bits(t.trace[i].x, whole.trace[i].x)) {
                    ADD_FAILURE() << "cap " << cap << " differs from the whole run at point " << i;
                    break;
                }
            }
            EXPECT_GE(t.r.final_radius, whole.r.final_radius);
        }
        // A cap of the whole length is not a cap: the run ends as it ended.
        flop::bobyqa::Options exact = o;
        exact.stopping.max_evaluations = total;
        const Outcome same = run(p.f, p.x0, exact);
        EXPECT_EQ(same.r.status, whole.r.status);
        EXPECT_EQ(same.r.evaluations, total);
    }
}

TEST(V0121Status, TheResultIsTheBestPointEvaluatedWithItsOwnValue) {
    // In every kind of run the result is the first of the least values in the
    // trace, with the value the objective returns there.
    struct Run1 {
        flop::bobyqa::Options o;
        Vec x0;
    };
    std::vector<Run1> runs;
    for (const double stop : {0.0, 0.2}) {
        Run1 r{.o = v0121::options(0.5, 400), .x0 = {0.1, 0.1}};
        if (stop > 0.0) r.o.stopping.stop_value = stop;
        runs.push_back(r);
    }
    {
        Run1 r{.o = v0121::options(0.5, 17), .x0 = {0.1, 0.1}};
        runs.push_back(r);
    }
    {
        Run1 r{.o = v0121::options(0.5, 400), .x0 = {0.1, 0.4}};
        r.o.initial_step = 0.25;
        r.o.bounds = flop::Bounds::box(Vec{0.0, 0.0}, Vec{1.0, 0.55});
        r.o.stopping.xtol_abs = 1e-8;
        runs.push_back(r);
    }
    for (const Run1& r : runs) {
        for (const bool batch : {false, true}) {
            const Outcome t = run(quadratic, r.x0, r.o, batch);
            std::size_t best = 0;
            for (std::size_t i = 1; i < t.trace.size(); ++i)
                if (t.trace[i].f < t.trace[best].f) best = i;
            EXPECT_TRUE(v0121::same_bits(t.r.x, t.trace[best].x)) << batch;
            EXPECT_TRUE(v0121::same_bits(t.r.f, t.trace[best].f)) << batch;
            EXPECT_TRUE(v0121::same_bits(t.r.f, quadratic(t.r.x)))
                << "the value is the objective's";
        }
    }
}

TEST(V0121Status, FinalRadiusIsTheFirstRadiusUntilRhoIsReduced) {
    // A run capped inside the starting points has not reduced rho: final_radius
    // is initial_step. One that has converged to rho_end
    // has final_radius rho_end, and a capped one is between.
    for (const std::size_t cap : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4}}) {
        flop::bobyqa::Options o = v0121::options(0.75, cap);
        const Outcome t = run(quadratic, Vec{0.1, 0.1}, o);
        EXPECT_EQ(t.r.final_radius, 0.75) << "cap " << cap;
    }
    flop::bobyqa::Options conv = v0121::options(0.75, 5000);
    conv.stopping.xtol_abs = 1e-7;
    const Outcome whole = run(rosenbrock3, Vec{-1.2, 1.0, 1.0}, conv);
    ASSERT_EQ(whole.r.status, flop::Status::XtolReached);
    for (std::size_t cap = 10; cap < whole.trace.size(); cap += 7) {
        flop::bobyqa::Options o = conv;
        o.stopping.max_evaluations = cap;
        const Outcome t = run(rosenbrock3, Vec{-1.2, 1.0, 1.0}, o);
        EXPECT_GE(t.r.final_radius, 1e-7) << "cap " << cap;
        EXPECT_LE(t.r.final_radius, 0.75) << "cap " << cap;
    }
}

TEST(V0121Status, ConvergedIsTrueOfExactlyTheThreeSuccessfulStatuses) {
    EXPECT_TRUE(flop::converged(flop::Status::XtolReached));
    EXPECT_TRUE(flop::converged(flop::Status::FtolReached));
    EXPECT_TRUE(flop::converged(flop::Status::StopValueReached));
    EXPECT_FALSE(flop::converged(flop::Status::MaxEvaluationsReached));
    EXPECT_FALSE(flop::converged(flop::Status::RoundoffLimited));
    // And a run capped before it finishes is not converged.
    const Outcome t = run(quadratic, Vec{0.1, 0.1}, v0121::options(0.5, 9));
    EXPECT_FALSE(flop::converged(t.r.status));
}
