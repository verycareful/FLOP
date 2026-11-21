// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: BOBYQA and floating point.
//
// Held here: the same inputs give the same trajectory to the bit, in both
// channels; the method is scale covariant, so that multiplying the objective
// by a power of two (which changes no rounding) changes no point of the run, to
// the bit, in either model; the extreme but ordinary values, signed zeros,
// subnormals and coordinates far from the origin, are ordinary values that
// give a finite result; and a run that reports success reports the right
// answer.
//
// The last is red on purpose. An objective multiplied by 1e140 or more, or by
// 1e-200 or less, ends with XtolReached after 23 evaluations at a point 0.19
// from the minimiser: a success that is wrong. The values are far inside the
// range of a double, and the same objective scaled by anything from 1e-50 to
// 1e100 is solved. Runs that may not end are given a deadline, and one that
// does not fails its test.

#include <gtest/gtest.h>

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

bool same_run(const Outcome& a, const Outcome& b) {
    if (a.trace.size() != b.trace.size()) return false;
    for (std::size_t i = 0; i < a.trace.size(); ++i)
        if (!v0121::same_bits(a.trace[i].x, b.trace[i].x)) return false;
    return v0121::same_bits(a.r.x, b.r.x) && a.r.status == b.r.status &&
           v0121::same_bits(a.r.final_radius, b.r.final_radius);
}

}  // namespace

TEST(V0121Fp, TheSameInputsGiveTheSameTrajectoryToTheBit) {
    flop::bobyqa::Options o = v0121::options_xtol(0.4, 1e-9, 3000);
    for (const bool batch : {false, true}) {
        const Outcome a = run(quadratic, Vec{0.1, 0.2, 0.3}, o, batch);
        const Outcome b = run(quadratic, Vec{0.1, 0.2, 0.3}, o, batch);
        EXPECT_TRUE(same_run(a, b)) << "batch " << batch;
    }
    o.bounds = flop::Bounds::box(Vec(3, -0.5), Vec(3, 0.4));
    const Outcome c = run(quadratic, Vec{0.1, 0.2, 0.3}, o);
    const Outcome d = run(quadratic, Vec{0.1, 0.2, 0.3}, o);
    EXPECT_TRUE(same_run(c, d));
}

TEST(V0121Fp, MultiplyingTheObjectiveByAPowerOfTwoChangesNoPointOfTheRunToTheBit) {
    // Every value, difference and product in the method is homogeneous in f, so
    // a power of two (exact in every operation) moves nothing: the points,
    // the status and the radius are those of the unscaled run, in both models
    // since reassociation cannot touch an exact scaling.
    struct Case {
        Vec x0;
        std::size_t m;
        bool boxed;
        double xtol;
    };
    for (const Case& c : {Case{.x0 = {0.1, 0.1, 0.1}, .m = 0, .boxed = false, .xtol = 1e-8},
                          Case{.x0 = {0.1, 0.1, 0.1, 0.1}, .m = 6, .boxed = false, .xtol = 1e-10},
                          Case{.x0 = {0.1, 0.4}, .m = 0, .boxed = true, .xtol = 1e-8},
                          Case{.x0 = {0.1, 0.1}, .m = 0, .boxed = false, .xtol = 1e-12}}) {
        flop::bobyqa::Options o = v0121::options(0.25, 3000);
        o.stopping.xtol_abs = c.xtol;
        o.interpolation_points = c.m;
        if (c.boxed) o.bounds = flop::Bounds::box(Vec(c.x0.size(), 0.0), Vec(c.x0.size(), 0.55));
        const Outcome base = run(quadratic, c.x0, o);
        for (const int e : {-60, -30, -10, 10, 30, 60, 100}) {
            const double scale = std::ldexp(1.0, e);
            auto f = [scale](std::span<const double> x) { return scale * quadratic(x); };
            for (const bool batch : {false, true}) {
                const Outcome t = run(f, c.x0, o, batch);
                EXPECT_TRUE(same_run(base, t)) << "2^" << e << " batch " << batch;
                // The value scales too.
                EXPECT_EQ(t.r.f, scale * base.r.f) << "2^" << e;
            }
        }
    }
}

TEST(V0121Fp, SignedZerosAndSubnormalsAreOrdinaryValues) {
    // f(x) = quadratic - quadratic(c) is -0.0 or +0.0 at the minimiser; and
    // f = 2^-1060 quadratic has subnormal values. Both runs end with a finite
    // point and value.
    const Vec c{0.3, 0.6};
    const double fc = quadratic(c);
    auto signed_zero = [fc](std::span<const double> x) {
        const double v = quadratic(x) - fc;
        return v == 0.0 ? -0.0 : v;
    };
    flop::bobyqa::Options o = v0121::options_xtol(0.4, 1e-9, 3000);
    const Outcome z = run(signed_zero, Vec{0.1, 0.1}, o);
    EXPECT_FALSE(flop::detail::any_bad(z.r.x));
    EXPECT_FALSE(flop::detail::fp_bad(z.r.f));
    EXPECT_NEAR(z.r.x[0], 0.3, 1e-6);
    auto tiny = [](std::span<const double> x) { return std::ldexp(quadratic(x), -1060); };
    const Outcome s = run(tiny, Vec{0.1, 0.1}, v0121::options(0.4, 400));
    EXPECT_FALSE(flop::detail::any_bad(s.r.x));
    EXPECT_FALSE(flop::detail::fp_bad(s.r.f));
    for (const auto& p : s.trace) {
        EXPECT_FALSE(flop::detail::any_bad(p.x));
    }
}

namespace {

struct Seen {
    std::size_t non_finite = 0;
    bool finished_normally = false;
    flop::Result r;
};

// Runs minimize of f from x0 on a thread with a deadline; null when it did not
// end in time.
std::shared_ptr<Seen> deadline_run(double scale, double centre, std::size_t n,
                                   std::chrono::milliseconds limit) {
    auto seen = std::make_shared<Seen>();
    const bool ended = v0121::finishes_within(limit, [seen, scale, centre, n] {
        auto f = [seen, scale, centre](std::span<const double> x) {
            if (flop::detail::any_bad(x)) ++seen->non_finite;
            double s = 0.0;
            for (std::size_t i = 0; i < x.size(); ++i) {
                const double d = (x[i] - centre) - 0.3 * static_cast<double>(i + 1);
                s += d * d;
            }
            return scale * s;
        };
        flop::bobyqa::Options o = v0121::options_xtol(0.5, 1e-7, 3000);
        seen->r = flop::bobyqa::minimize(f, Vec(n, centre), o);
        seen->finished_normally = true;
    });
    return ended ? seen : nullptr;
}

}  // namespace

TEST(V0121Fp, ProblemsFarFromTheOriginAreSolvedWithoutAnyNonFiniteCoordinate) {
    for (const double centre : {1e3, 1e9, -1e9, 1e15}) {
        const auto seen = deadline_run(1.0, centre, 3, std::chrono::milliseconds(10000));
        ASSERT_NE(seen, nullptr) << "the run did not end: centre " << centre;
        EXPECT_EQ(seen->non_finite, 0u) << centre;
        EXPECT_FALSE(flop::detail::any_bad(seen->r.x)) << centre;
        EXPECT_FALSE(flop::detail::fp_bad(seen->r.f)) << centre;
    }
}

TEST(V0121Fp, AReportedSuccessIsTheRightAnswerAtEveryScaleOfTheObjective) {
    // Red on purpose, see the file header. The scales from 1e-50 to 1e100 are
    // here as well, and pass.
    for (const double scale : {1e-50, 1e-10, 1.0, 1e10, 1e100, 1e140, 1e200, 1e-200}) {
        const auto seen = deadline_run(scale, 0.0, 2, std::chrono::milliseconds(10000));
        ASSERT_NE(seen, nullptr) << "the run did not end: scale " << scale;
        if (!flop::converged(seen->r.status)) continue;  // not claiming success
        EXPECT_NEAR(seen->r.x[0], 0.3, 1e-5)
            << "scale " << scale << " reported " << flop::to_string(seen->r.status);
        EXPECT_NEAR(seen->r.x[1], 0.6, 1e-5) << "scale " << scale;
    }
}
