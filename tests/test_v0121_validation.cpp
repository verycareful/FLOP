// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: BOBYQA's input validation and run-time errors.
//
// Every std::invalid_argument docs/api/bobyqa.md lists, on both entry points
// and through the facade, each with the guarantee that goes with it: a
// rejected call has evaluated nothing. The rules that are BOBYQA's own are
// the number of interpolation points, in [n + 2, (n + 1)(n + 2) / 2] (Powell,
// DAMTP 2009/NA06, section 1), the final radius, and the width of a box,
// b_i - a_i >= 2 initial_step (2.1) unless shrink_initial_step is set. Then
// the things that end a well-formed run with an exception: a non-finite
// objective value (std::runtime_error) on both channels, and an exception
// thrown by the objective or by the on_evaluation callback, which propagates
// unchanged.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "flop/flop.hpp"
#include "v0121_problems.hpp"

namespace {

using v0121::Vec;

constexpr double kMax = std::numeric_limits<double>::max();

flop::bobyqa::Options good() {
    return v0121::options_xtol(0.5, 1e-6, 200);
}

// The text of the std::invalid_argument a call throws, empty when it throws
// nothing.
template <class Call>
std::string rejection(Call&& call) {
    try {
        call();
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return {};
}

}  // namespace

TEST(V0121Validation, EmptyX0IsRejected) {
    v0121::expect_rejected_everywhere(Vec{}, good(), "empty x0");
}

TEST(V0121Validation, NonFiniteX0IsRejected) {
    for (const std::uint64_t bits : {v0121::kQuietNaN, v0121::kInfinity, v0121::kNegInfinity}) {
        Vec x0{1.0, 1.0};
        v0121::write_bits(&x0[1], bits);
        v0121::expect_rejected_everywhere(x0, good(), "non-finite x0");
    }
}

TEST(V0121Validation, NoStoppingCriterionIsRejected) {
    flop::bobyqa::Options o;  // every criterion at its off default
    v0121::expect_rejected_everywhere(Vec{1.0, 1.0}, o, "no stopping criterion");
}

TEST(V0121Validation, EachStoppingCriterionAloneIsEnough) {
    const Vec x0{1.0, 1.0};
    for (int which = 0; which < 6; ++which) {
        flop::bobyqa::Options o;
        if (which == 0) o.stopping.max_evaluations = 5;
        if (which == 1) o.stopping.xtol_rel = 1e-3;
        if (which == 2) o.stopping.xtol_abs = 1e-3;
        if (which == 3) o.stopping.ftol_rel = 1e-3;
        if (which == 4) o.stopping.ftol_abs = 1e-3;
        if (which == 5) o.stopping.stop_value = 1.0;
        EXPECT_NO_THROW((void)flop::bobyqa::minimize(v0111::sphere, x0, o)) << which;
    }
}

TEST(V0121Validation, NegativeOrNonFiniteTolerancesAreRejected) {
    const Vec x0{1.0, 1.0};
    for (int which = 0; which < 4; ++which) {
        flop::bobyqa::Options o = good();
        double* t[4] = {&o.stopping.xtol_rel, &o.stopping.xtol_abs, &o.stopping.ftol_rel,
                        &o.stopping.ftol_abs};
        *t[which] = -1e-9;
        v0121::expect_rejected_everywhere(x0, o, "negative tolerance");
        for (const std::uint64_t bits : {v0121::kQuietNaN, v0121::kInfinity}) {
            flop::bobyqa::Options p = good();
            double* u[4] = {&p.stopping.xtol_rel, &p.stopping.xtol_abs, &p.stopping.ftol_rel,
                            &p.stopping.ftol_abs};
            v0121::write_bits(u[which], bits);
            v0121::expect_rejected_everywhere(x0, p, "non-finite tolerance");
        }
    }
    flop::bobyqa::Options o = good();
    o.stopping.stop_value = 0.0;
    v0121::write_bits(&*o.stopping.stop_value, v0121::kNegInfinity);
    v0121::expect_rejected_everywhere(x0, o, "non-finite stop_value");
}

TEST(V0121Validation, InitialStepMustBePositiveAndFinite) {
    const Vec x0{1.0, 1.0};
    for (const double bad : {0.0, -0.5}) {
        flop::bobyqa::Options o = good();
        o.initial_step = bad;
        v0121::expect_rejected_everywhere(x0, o, "non-positive initial_step");
    }
    for (const std::uint64_t bits : {v0121::kInfinity, v0121::kQuietNaN}) {
        flop::bobyqa::Options o = good();
        v0121::write_bits(&o.initial_step, bits);
        v0121::expect_rejected_everywhere(x0, o, "non-finite initial_step");
    }
}

TEST(V0121Validation, MalformedBoundsAreRejected) {
    const Vec x0{1.0, 1.0};
    {
        flop::bobyqa::Options o = good();
        o.bounds = flop::Bounds::none(3);
        v0121::expect_rejected_everywhere(x0, o, "bounds of the wrong length");
    }
    {
        flop::bobyqa::Options o = good();
        o.bounds = flop::Bounds::none(2);
        o.bounds->upper.emplace_back(1.0);
        v0121::expect_rejected_everywhere(x0, o, "lower and upper of different lengths");
    }
    {
        flop::bobyqa::Options o = good();
        o.bounds = flop::Bounds::box(Vec{0.0, 1.0}, Vec{2.0, 1.0});
        v0121::expect_rejected_everywhere(x0, o, "a fixed coordinate");
    }
    {
        flop::bobyqa::Options o = good();
        o.bounds = flop::Bounds::box(Vec{2.0, 0.0}, Vec{0.0, 2.0});
        v0121::expect_rejected_everywhere(x0, o, "lower above upper");
    }
    {
        flop::bobyqa::Options o = good();
        o.bounds = flop::Bounds::none(2);
        v0121::write_bits(&o.bounds->lower[0].emplace(0.0), v0121::kNegInfinity);
        v0121::expect_rejected_everywhere(x0, o, "a non-finite lower bound");
    }
    {
        flop::bobyqa::Options o = good();
        o.bounds = flop::Bounds::none(2);
        v0121::write_bits(&o.bounds->upper[1].emplace(0.0), v0121::kQuietNaN);
        v0121::expect_rejected_everywhere(x0, o, "a non-finite upper bound");
    }
}

TEST(V0121Validation, X0OutsideTheBoxIsRejectedAndOnItsBoundaryAccepted) {
    {
        flop::bobyqa::Options o = good();
        o.bounds = flop::Bounds::none(2);
        o.bounds->upper[0] = 0.5;
        v0121::expect_rejected_everywhere(Vec{1.0, 1.0}, o, "x0 above the box");
    }
    {
        flop::bobyqa::Options o = good();
        o.bounds = flop::Bounds::none(2);
        o.bounds->lower[1] = 1.5;
        v0121::expect_rejected_everywhere(Vec{1.0, 1.0}, o, "x0 below the box");
    }
    flop::bobyqa::Options o = good();
    o.bounds = flop::Bounds::box(Vec{0.0, -1.0}, Vec{2.0, 2.0});
    EXPECT_NO_THROW((void)flop::bobyqa::minimize(v0111::sphere, Vec{0.0, 2.0}, o));
}

// ---- the rules that are BOBYQA's own ----------------------------------------

TEST(V0121Validation, TheInterpolationPointCountIsInTheReportsRange) {
    // Section 1: n + 2 <= m <= (n + 1)(n + 2) / 2, and 0 selects 2n + 1.
    // At n = 1 the range holds the one value 3.
    for (std::size_t n = 1; n <= 4; ++n) {
        const std::size_t lo = n + 2;
        const std::size_t hi = (n + 1) * (n + 2) / 2;
        const Vec x0(n, 0.5);
        for (std::size_t m = lo; m <= hi; ++m) {
            flop::bobyqa::Options o = good();
            o.interpolation_points = m;
            EXPECT_NO_THROW((void)flop::bobyqa::minimize(v0111::sphere, x0, o))
                << "n " << n << " m " << m;
        }
        flop::bobyqa::Options zero = good();
        zero.interpolation_points = 0;
        EXPECT_NO_THROW((void)flop::bobyqa::minimize(v0111::sphere, x0, zero)) << "n " << n;
        for (const std::size_t m :
             {std::size_t{1}, n + 1, hi + 1, hi + 2, std::numeric_limits<std::size_t>::max()}) {
            if (m == 0 || (m >= lo && m <= hi)) continue;
            flop::bobyqa::Options o = good();
            o.interpolation_points = m;
            v0121::expect_rejected_everywhere(x0, o, "interpolation_points outside the range");
        }
    }
}

TEST(V0121Validation, TheFinalRadiusIsZeroOrPositiveAndFinite) {
    const Vec x0{1.0, 1.0};
    for (const double bad : {-1e-9, -1.0}) {
        flop::bobyqa::Options o = good();
        o.final_trust_radius = bad;
        v0121::expect_rejected_everywhere(x0, o, "negative final_trust_radius");
    }
    for (const std::uint64_t bits : {v0121::kInfinity, v0121::kNegInfinity, v0121::kQuietNaN}) {
        flop::bobyqa::Options o = good();
        v0121::write_bits(&o.final_trust_radius, bits);
        v0121::expect_rejected_everywhere(x0, o, "non-finite final_trust_radius");
    }
    // Zero derives it from the tolerances; a positive value, even one above
    // initial_step or at the bottom of the range, is input.
    for (const double ok : {0.0, std::numeric_limits<double>::denorm_min(), 1e-8, 0.5, 3.0}) {
        flop::bobyqa::Options o = good();
        o.final_trust_radius = ok;
        EXPECT_NO_THROW((void)flop::bobyqa::minimize(v0111::sphere, x0, o)) << ok;
    }
}

TEST(V0121Validation, ABoxNarrowerThanTwiceTheFirstStepIsRefused) {
    // (2.1): b_i - a_i >= 2 initial_step on every coordinate bounded on both
    // sides. Widths and steps here are dyadic, so the comparison is exact.
    const Vec x0{0.5, 0.5};
    flop::bobyqa::Options o = good();
    o.bounds = flop::Bounds::box(Vec{0.0, 0.0}, Vec{1.0, 1.0});
    o.initial_step = 0.75;
    v0121::expect_rejected_everywhere(x0, o, "a box narrower than 2 initial_step");
    // Exactly twice the step is wide enough.
    o.initial_step = 0.5;
    EXPECT_NO_THROW((void)flop::bobyqa::minimize(v0111::sphere, x0, o));
    // One unit in the last place above it is not.
    o.initial_step = std::nextafter(0.5, 1.0);
    v0121::expect_rejected_everywhere(x0, o, "a box one ulp too narrow");
}

TEST(V0121Validation, OnlyACoordinateBoundedOnBothSidesIsMeasured) {
    // A one-sided coordinate has no width, however close x0 is to its bound.
    const Vec x0{0.5, 0.5};
    flop::bobyqa::Options o = good();
    o.initial_step = 4.0;
    o.bounds = flop::Bounds::none(2);
    o.bounds->lower[0] = 0.0;
    o.bounds->upper[1] = 1.0;
    EXPECT_NO_THROW((void)flop::bobyqa::minimize(v0111::sphere, x0, o));
}

TEST(V0121Validation, TheNarrowCoordinateIsNamed) {
    flop::bobyqa::Options o = good();
    o.initial_step = 0.5;
    o.bounds = flop::Bounds::box(Vec{-8.0, 0.0, -8.0}, Vec{8.0, 0.75, 8.0});
    const Vec x0{0.0, 0.5, 0.0};
    const std::string scalar =
        rejection([&] { (void)flop::bobyqa::minimize(v0111::sphere, x0, o); });
    EXPECT_NE(scalar.find("flop::bobyqa::minimize: "), std::string::npos) << scalar;
    EXPECT_NE(scalar.find("coordinate 1"), std::string::npos) << scalar;
    EXPECT_NE(scalar.find("shrink_initial_step"), std::string::npos) << scalar;
    auto fb = [](std::span<const std::span<const double>> xs, std::span<double> out) {
        for (std::size_t i = 0; i < xs.size(); ++i) out[i] = v0111::sphere(xs[i]);
    };
    const std::string batch = rejection([&] { (void)flop::bobyqa::minimize_batch(fb, x0, o); });
    EXPECT_NE(batch.find("flop::bobyqa::minimize_batch: "), std::string::npos) << batch;
    EXPECT_NE(batch.find("coordinate 1"), std::string::npos) << batch;
}

TEST(V0121Validation, ShrinkingTheFirstStepAcceptsANarrowBoxButNotAFixedCoordinate) {
    const Vec x0{0.5, 0.5};
    flop::bobyqa::Options o = good();
    o.initial_step = 4.0;
    o.shrink_initial_step = true;
    o.bounds = flop::Bounds::box(Vec{0.0, 0.0}, Vec{1.0, 1.0});
    EXPECT_NO_THROW((void)flop::bobyqa::minimize(v0111::sphere, x0, o));
    // A coordinate with a_i = b_i is not a variable, with or without the flag.
    o.bounds = flop::Bounds::box(Vec{0.0, 0.5}, Vec{1.0, 0.5});
    v0121::expect_rejected_everywhere(x0, o, "a fixed coordinate with shrink_initial_step");
    o.shrink_initial_step = false;
    v0121::expect_rejected_everywhere(x0, o, "a fixed coordinate");
    // Nor is a lower bound above the upper one.
    o.shrink_initial_step = true;
    o.bounds = flop::Bounds::box(Vec{1.0, 0.0}, Vec{0.0, 1.0});
    v0121::expect_rejected_everywhere(x0, o, "lower above upper with shrink_initial_step");
}

TEST(V0121Validation, TheWidthOfABoxCannotOverflowTheTest) {
    // b - a = 1.5 DBL_MAX is not a double; the test compares b/2 - a/2.
    flop::bobyqa::Options o = v0121::options(1.0, 3);
    o.bounds = flop::Bounds::box(Vec{-0.75 * kMax}, Vec{0.75 * kMax});
    EXPECT_NO_THROW(
        (void)flop::bobyqa::minimize([](std::span<const double>) { return 0.0; }, Vec{0.0}, o));
    // A first step beyond half of that width is still refused.
    o.initial_step = 0.9 * kMax;
    v0121::expect_rejected_everywhere(Vec{0.0}, o, "a step beyond half the width");
}

// ---- what ends a run that was accepted ---------------------------------------

TEST(V0121Validation, ANonFiniteObjectiveValueEndsTheRunWithRuntimeError) {
    // A NaN or infinity written into the library's memory by a batch call of
    // more than one point is caught under every compiler and model. One
    // handed back by value (the scalar channel, and the batch channel's
    // single-point calls, which return the value it wrote) is undefined
    // under clang's -ffinite-math-only before FLOP sees it; GCC keeps the
    // bits and the check catches them.
    for (const std::uint64_t bits : {v0121::kQuietNaN, v0121::kInfinity, v0121::kNegInfinity}) {
        std::size_t calls = 0;
        auto fb = [&](std::span<const std::span<const double>> xs, std::span<double> out) {
            for (std::size_t i = 0; i < xs.size(); ++i) {
                ++calls;
                out[i] = v0111::sphere(xs[i]);
                if (calls == 2) v0121::write_bits(&out[i], bits);
            }
        };
        EXPECT_THROW((void)flop::bobyqa::minimize_batch(fb, Vec{1.0, 1.0, 1.0}, good()),
                     std::runtime_error);
        EXPECT_EQ(calls, 7u);  // the whole first call, 2n + 1 points, ran
    }
#if defined(__clang__) && defined(__FAST_MATH__)
    GTEST_SKIP() << "clang -ffinite-math-only: a NaN returned by value is poison, see the "
                    "algorithm page's limits";
#else
    // At the first point, at a starting point, at the last starting point, and
    // from the first trust-region step on.
    for (const std::size_t at : {std::size_t{1}, std::size_t{3}, std::size_t{5}, std::size_t{6},
                                 std::size_t{9}, std::size_t{14}}) {
        std::size_t calls = 0;
        auto f = [&](std::span<const double> x) {
            ++calls;
            if (calls == at) {
                double nan = 0.0;
                v0121::write_bits(&nan, v0121::kQuietNaN);
                return nan;
            }
            return v0111::sphere(x);
        };
        EXPECT_THROW((void)flop::bobyqa::minimize(f, Vec{1.0, 1.0}, good()), std::runtime_error)
            << "at " << at;
        EXPECT_EQ(calls, at) << "an evaluation followed the error";
        std::size_t bcalls = 0;
        auto fb = [&](std::span<const std::span<const double>> xs, std::span<double> out) {
            for (std::size_t i = 0; i < xs.size(); ++i) {
                ++bcalls;
                out[i] = v0111::sphere(xs[i]);
                if (bcalls == at) v0121::write_bits(&out[i], v0121::kInfinity);
            }
        };
        EXPECT_THROW((void)flop::bobyqa::minimize_batch(fb, Vec{1.0, 1.0}, good()),
                     std::runtime_error)
            << "at " << at;
    }
#endif
}

namespace {

// An exception of the test's own type, so that propagating it unchanged
// cannot be confused with the library throwing one of its own.
struct Thrown : std::exception {
    explicit Thrown(int t) : tag(t) {}
    int tag;
};

}  // namespace

TEST(V0121Validation, WhatTheObjectiveOrTheCallbackThrowsPropagatesUnchanged) {
    for (const std::size_t at : {std::size_t{1}, std::size_t{5}, std::size_t{12}}) {
        std::size_t calls = 0;
        auto f = [&](std::span<const double> x) -> double {
            if (++calls == at) throw Thrown(7);
            return v0111::sphere(x);
        };
        try {
            (void)flop::bobyqa::minimize(f, Vec{1.0, 1.0}, good());
            ADD_FAILURE() << "no exception";
        } catch (const Thrown& t) {
            EXPECT_EQ(t.tag, 7);
        }
        EXPECT_EQ(calls, at);
    }
    {
        auto fb = [](std::span<const std::span<const double>>, std::span<double>) {
            throw Thrown(8);
        };
        try {
            (void)flop::bobyqa::minimize_batch(fb, Vec{1.0, 1.0}, good());
            ADD_FAILURE() << "no exception";
        } catch (const Thrown& t) {
            EXPECT_EQ(t.tag, 8);
        }
    }
    {
        flop::bobyqa::Options o = good();
        o.on_evaluation = [](const flop::Evaluation& e) {
            if (e.index == 3) throw Thrown(9);
        };
        try {
            (void)flop::bobyqa::minimize(v0111::sphere, Vec{1.0, 1.0}, o);
            ADD_FAILURE() << "no exception";
        } catch (const Thrown& t) {
            EXPECT_EQ(t.tag, 9);
        }
    }
}
