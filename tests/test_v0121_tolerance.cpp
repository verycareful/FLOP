// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: BOBYQA's f tolerances.
//
// BOBYQA ends with FtolReached when a trust-region or alternative step that
// lowers the best value lowers it by no more than ftol_abs, or by no more
// than ftol_rel |f_new| (docs/algorithms/bobyqa.md, "Stopping rules beyond
// rho"). The two comparisons are the shared helpers of
// flop/detail/tolerance.hpp, which are held at the ends of the finite range
// in V0112Tolerance; here the method is held to the rule they serve: only a
// step that lowers the best value counts, the fall is compared as written,
// each tolerance is off at zero, and objective values at the ends of the
// finite range are handled without forming a value outside it.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include "flop/detail/fp.hpp"
#include "flop/flop.hpp"
#include "v0121_problems.hpp"

namespace {

using v0121::Vec;

constexpr double kMax = std::numeric_limits<double>::max();

struct Outcome {
    flop::Result r;
    v0121::Trace trace;
};

template <class F>
Outcome run_of(F f, const Vec& x0, flop::bobyqa::Options o) {
    Outcome out;
    o.on_evaluation = v0121::recorder(out.trace);
    out.r = flop::bobyqa::minimize(f, x0, o);
    return out;
}

Outcome run(const Vec& x0, const flop::bobyqa::Options& o) {
    return run_of(v0111::sphere, x0, o);
}

// 100 + sum (x_i - c_i)^4, c the sphere's centre: the value at the minimum is
// 100, so a tolerance relative to |f| stays meaningful there, and the falls
// of a quadratic model on it shrink step by step instead of vanishing at the
// first one, as they do on a quadratic.
double lifted_quartic(std::span<const double> x) {
    double s = 100.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double d = x[i] - v0111::sphere_centre(i);
        s += d * d * d * d;
    }
    return s;
}

// The successive improvements of the running best value among the
// evaluations from index first on: (index, fall, new best).
struct Improvement {
    std::size_t index;
    double fall;
    double value;
};

std::vector<Improvement> improvements(const v0121::Trace& t, std::size_t first) {
    std::vector<Improvement> out;
    double best = t.empty() ? 0.0 : t[0].f;
    for (std::size_t i = 1; i < t.size(); ++i) {
        if (t[i].f < best) {
            if (i >= first) out.push_back({.index = i, .fall = best - t[i].f, .value = t[i].f});
            best = t[i].f;
        }
    }
    return out;
}

}  // namespace

TEST(V0121Tolerance, ATolerancesOfZeroAreOff) {
    // Neither f tolerance set: the run ends on the radius, never FtolReached.
    flop::bobyqa::Options o = v0121::options_xtol(0.5, 1e-8, 2000);
    const Outcome a = run(Vec{2.0, 2.0, 2.0}, o);
    EXPECT_EQ(a.r.status, flop::Status::XtolReached);
    // A tolerance below every fall the run makes does not end it either.
    o.stopping.ftol_abs = std::numeric_limits<double>::denorm_min();
    o.stopping.ftol_rel = std::numeric_limits<double>::denorm_min();
    const Outcome b = run(Vec{2.0, 2.0, 2.0}, o);
    EXPECT_EQ(b.r.status, flop::Status::XtolReached);
}

TEST(V0121Tolerance, AnAbsoluteToleranceEndsTheRunOnAFallNoLargerThanIt) {
    // f(x0) is about 13: a tolerance far above any fall the run can make ends
    // it at the first step that lowers the best value.
    const Vec x0{2.0, 2.0, 2.0};
    flop::bobyqa::Options o = v0121::options(0.5, 2000);
    o.stopping.ftol_abs = 1e6;
    const Outcome big = run(x0, o);
    EXPECT_EQ(big.r.status, flop::Status::FtolReached);
    const std::size_t init = 2 * x0.size() + 1;
    EXPECT_GT(big.r.evaluations, init) << "a step must have been taken";
    // A small tolerance lets the run go on until a fall is that small.
    o.stopping.ftol_abs = 1e-6;
    const Outcome small = run(x0, o);
    EXPECT_EQ(small.r.status, flop::Status::FtolReached);
    EXPECT_GT(small.r.evaluations, big.r.evaluations);
    // Every improvement after the starting points is above the tolerance but
    // the last, which is at most the tolerance: the rule as stated.
    const std::vector<Improvement> imp = improvements(small.trace, init);
    ASSERT_FALSE(imp.empty());
    for (std::size_t k = 0; k + 1 < imp.size(); ++k)
        EXPECT_GT(imp[k].fall, o.stopping.ftol_abs) << "improvement " << k;
    EXPECT_LE(imp.back().fall, o.stopping.ftol_abs);
    // The result is the best point evaluated, and it is where the run ended.
    EXPECT_TRUE(v0121::same_bits(small.r.f, imp.back().value));
    EXPECT_EQ(imp.back().index, small.trace.size() - 1);
}

TEST(V0121Tolerance, ARelativeToleranceIsMeasuredOnTheNewValue) {
    // The test is fall <= ftol_rel |f_new|, the value after the step.
    const Vec x0{2.0, 2.0, 2.0};
    flop::bobyqa::Options o = v0121::options(0.5, 2000);
    o.stopping.ftol_rel = 1e-3;
    const Outcome a = run_of(lifted_quartic, x0, o);
    EXPECT_EQ(a.r.status, flop::Status::FtolReached);
    const std::size_t init = 2 * x0.size() + 1;
    const std::vector<Improvement> imp = improvements(a.trace, init);
    ASSERT_GE(imp.size(), 2u) << "the rule needs a step above the tolerance and one below it";
    for (std::size_t k = 0; k + 1 < imp.size(); ++k)
        EXPECT_GT(imp[k].fall, o.stopping.ftol_rel * std::fabs(imp[k].value)) << k;
    EXPECT_LE(imp.back().fall, o.stopping.ftol_rel * std::fabs(imp.back().value));
}

TEST(V0121Tolerance, ARelativeToleranceAboveOneIsAccepted) {
    // ftol_rel > 1 is decided by a division, never by the product, so a
    // tolerance of 4 on a run whose fall is a fraction of |f_new| ends it at
    // the first improving step.
    const Vec x0{2.0, 2.0, 2.0};
    flop::bobyqa::Options o = v0121::options(0.5, 2000);
    o.stopping.ftol_rel = 4.0;
    const Outcome a = run(x0, o);
    EXPECT_EQ(a.r.status, flop::Status::FtolReached);
    o.stopping.ftol_rel = 1e300;
    const Outcome b = run(x0, o);
    EXPECT_EQ(b.r.status, flop::Status::FtolReached);
    EXPECT_EQ(b.r.evaluations, a.r.evaluations) << "both end at the first improving step";
}

TEST(V0121Tolerance, ASetOfBothToleranceKindsEndsOnWhicheverIsMetFirst) {
    const Vec x0{2.0, 2.0, 2.0};
    flop::bobyqa::Options o = v0121::options(0.5, 2000);
    o.stopping.ftol_abs = 1e-12;
    o.stopping.ftol_rel = 1e300;
    const Outcome both = run(x0, o);
    o.stopping.ftol_abs = 0.0;
    const Outcome rel_only = run(x0, o);
    EXPECT_EQ(both.r.status, flop::Status::FtolReached);
    EXPECT_EQ(both.r.evaluations, rel_only.r.evaluations);
}

TEST(V0121Tolerance, ValuesAtTheEndsOfTheRangeAreHandledWithoutNonFiniteArithmetic) {
    // f is +0.75 DBL_MAX on one side of x = 1 and -0.75 DBL_MAX on the other:
    // two values here differ by 1.5 DBL_MAX, which is not a double, and a
    // relative tolerance above one would form a product beyond the range.
    // Nothing is handed to the objective or returned that is not finite, and
    // no improving step follows the starting points, so FtolReached does not
    // end the run for any of the tolerances. The run is given a deadline:
    // one that does not end fails the test.
    struct Seen {
        std::size_t non_finite = 0;
        bool ftol = false;
        bool result_finite = true;
    };
    struct Case {
        double tol;
        bool relative;
    };
    for (const Case c : {Case{.tol = 1.0, .relative = false}, Case{.tol = 4.0, .relative = true},
                         Case{.tol = kMax, .relative = false}}) {
        auto seen = std::make_shared<Seen>();
        const bool ended = v0121::finishes_within(std::chrono::milliseconds(5000), [seen, c] {
            auto f = [seen](std::span<const double> x) {
                if (flop::detail::any_bad(x)) ++seen->non_finite;
                return x[0] > 1.0 ? -0.75 * kMax : 0.75 * kMax;
            };
            flop::bobyqa::Options o = v0121::options(0.5, 200);
            (c.relative ? o.stopping.ftol_rel : o.stopping.ftol_abs) = c.tol;
            const flop::Result r = flop::bobyqa::minimize(f, Vec{1.0}, o);
            seen->ftol = r.status == flop::Status::FtolReached;
            seen->result_finite = !flop::detail::fp_bad(r.f) && !flop::detail::any_bad(r.x);
        });
        ASSERT_TRUE(ended) << "the run did not end: tolerance " << c.tol
                           << (c.relative ? " rel" : " abs");
        EXPECT_FALSE(seen->ftol) << c.tol << (c.relative ? " rel" : " abs");
        EXPECT_EQ(seen->non_finite, 0u) << c.tol;
        EXPECT_TRUE(seen->result_finite) << c.tol;
    }
}
