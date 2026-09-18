// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.0.1 test wave: input validation.
//
// Every std::invalid_argument an entry point can throw, one test each, and
// the guarantee that goes with them: a rejected call has evaluated nothing.
// The objective in every test counts its calls and the count is asserted
// zero after the throw. The runtime_error for a non-finite objective value
// is here too, since it is the one exception a well-formed call can raise.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "flop/flop.hpp"
#include "v0101_problems.hpp"

namespace {

struct Counting {
    std::size_t calls = 0;
    double operator()(std::span<const double> x) {
        ++calls;
        return v0101::sphere(x);
    }
};

flop::cobyla::Options good() {
    return v0101::options(0.5, 1e-6, 100);
}

}  // namespace

TEST(V0101Validation, EmptyX0Throws) {
    Counting f;
    const std::vector<double> x0;
    EXPECT_THROW((void)flop::cobyla::minimize(f, x0, good()), std::invalid_argument);
    EXPECT_EQ(f.calls, 0u);
}

TEST(V0101Validation, NonFiniteX0Throws) {
    Counting f;
    std::vector<double> nan_x0{1.0, 1.0};
    std::vector<double> inf_x0{1.0, 1.0};
    v0101::write_bits(&nan_x0[1], v0101::kQuietNaN);
    v0101::write_bits(&inf_x0[0], v0101::kInfinity);
    EXPECT_THROW((void)flop::cobyla::minimize(f, nan_x0, good()), std::invalid_argument);
    EXPECT_THROW((void)flop::cobyla::minimize(f, inf_x0, good()), std::invalid_argument);
    EXPECT_EQ(f.calls, 0u);
}

TEST(V0101Validation, NoStoppingCriterionThrows) {
    Counting f;
    const std::vector<double> x0{1.0, 1.0};
    flop::cobyla::Options o;  // every criterion at its "off" default
    EXPECT_THROW((void)flop::cobyla::minimize(f, x0, o), std::invalid_argument);
    EXPECT_EQ(f.calls, 0u);
}

TEST(V0101Validation, EachCriterionAloneIsEnough) {
    const std::vector<double> x0{1.0, 1.0};
    {
        flop::cobyla::Options o;
        o.stopping.max_evaluations = 5;
        EXPECT_NO_THROW((void)flop::cobyla::minimize(v0101::sphere, x0, o));
    }
    {
        flop::cobyla::Options o;
        o.stopping.xtol_rel = 1e-3;
        EXPECT_NO_THROW((void)flop::cobyla::minimize(v0101::sphere, x0, o));
    }
    {
        flop::cobyla::Options o;
        o.stopping.xtol_abs = 1e-3;
        EXPECT_NO_THROW((void)flop::cobyla::minimize(v0101::sphere, x0, o));
    }
    {
        flop::cobyla::Options o;
        o.stopping.ftol_rel = 1e-3;
        EXPECT_NO_THROW((void)flop::cobyla::minimize(v0101::sphere, x0, o));
    }
    {
        flop::cobyla::Options o;
        o.stopping.ftol_abs = 1e-3;
        EXPECT_NO_THROW((void)flop::cobyla::minimize(v0101::sphere, x0, o));
    }
    {
        flop::cobyla::Options o;
        o.stopping.stop_value = 1.0;
        EXPECT_NO_THROW((void)flop::cobyla::minimize(v0101::sphere, x0, o));
    }
}

TEST(V0101Validation, NegativeToleranceThrows) {
    Counting f;
    const std::vector<double> x0{1.0, 1.0};
    for (int which = 0; which < 4; ++which) {
        flop::cobyla::Options o = good();
        if (which == 0) o.stopping.xtol_rel = -1e-9;
        if (which == 1) o.stopping.xtol_abs = -1e-9;
        if (which == 2) o.stopping.ftol_rel = -1e-9;
        if (which == 3) o.stopping.ftol_abs = -1e-9;
        EXPECT_THROW((void)flop::cobyla::minimize(f, x0, o), std::invalid_argument) << which;
    }
    EXPECT_EQ(f.calls, 0u);
}

TEST(V0101Validation, NonFiniteToleranceOrStopValueThrows) {
    Counting f;
    const std::vector<double> x0{1.0, 1.0};
    {
        flop::cobyla::Options o = good();
        v0101::write_bits(&o.stopping.xtol_rel, v0101::kQuietNaN);
        EXPECT_THROW((void)flop::cobyla::minimize(f, x0, o), std::invalid_argument);
    }
    {
        flop::cobyla::Options o = good();
        v0101::write_bits(&o.stopping.ftol_abs, v0101::kInfinity);
        EXPECT_THROW((void)flop::cobyla::minimize(f, x0, o), std::invalid_argument);
    }
    {
        flop::cobyla::Options o = good();
        o.stopping.stop_value = 0.0;
        v0101::write_bits(&*o.stopping.stop_value, v0101::kNegInfinity);
        EXPECT_THROW((void)flop::cobyla::minimize(f, x0, o), std::invalid_argument);
    }
    EXPECT_EQ(f.calls, 0u);
}

TEST(V0101Validation, InitialStepMustBePositiveAndFinite) {
    Counting f;
    const std::vector<double> x0{1.0, 1.0};
    for (const double bad : {0.0, -0.5}) {
        flop::cobyla::Options o = good();
        o.initial_step = bad;
        EXPECT_THROW((void)flop::cobyla::minimize(f, x0, o), std::invalid_argument);
    }
    for (const std::uint64_t bits : {v0101::kInfinity, v0101::kQuietNaN}) {
        flop::cobyla::Options o = good();
        v0101::write_bits(&o.initial_step, bits);
        EXPECT_THROW((void)flop::cobyla::minimize(f, x0, o), std::invalid_argument);
    }
    EXPECT_EQ(f.calls, 0u);
}

TEST(V0101Validation, BoundsOfTheWrongLengthThrow) {
    Counting f;
    const std::vector<double> x0{1.0, 1.0};
    flop::cobyla::Options o = good();
    o.bounds = flop::Bounds::none(3);
    EXPECT_THROW((void)flop::cobyla::minimize(f, x0, o), std::invalid_argument);
    o.bounds = flop::Bounds::none(2);
    o.bounds->upper.emplace_back(1.0);  // lower has 2 entries, upper 3
    EXPECT_THROW((void)flop::cobyla::minimize(f, x0, o), std::invalid_argument);
    EXPECT_EQ(f.calls, 0u);
}

TEST(V0101Validation, LowerNotBelowUpperThrows) {
    Counting f;
    const std::vector<double> x0{1.0, 1.0};
    flop::cobyla::Options o = good();
    const std::vector<double> lo{0.0, 1.0};
    const std::vector<double> hi{2.0, 1.0};  // coordinate 1 is fixed
    o.bounds = flop::Bounds::box(lo, hi);
    EXPECT_THROW((void)flop::cobyla::minimize(f, x0, o), std::invalid_argument);
    o.bounds = flop::Bounds::box(hi, lo);  // reversed on coordinate 0
    EXPECT_THROW((void)flop::cobyla::minimize(f, x0, o), std::invalid_argument);
    EXPECT_EQ(f.calls, 0u);
}

TEST(V0101Validation, NonFiniteBoundThrows) {
    Counting f;
    const std::vector<double> x0{1.0, 1.0};
    flop::cobyla::Options o = good();
    o.bounds = flop::Bounds::none(2);
    v0101::write_bits(&o.bounds->lower[0].emplace(0.0), v0101::kNegInfinity);
    EXPECT_THROW((void)flop::cobyla::minimize(f, x0, o), std::invalid_argument);
    o.bounds = flop::Bounds::none(2);
    v0101::write_bits(&o.bounds->upper[1].emplace(0.0), v0101::kQuietNaN);
    EXPECT_THROW((void)flop::cobyla::minimize(f, x0, o), std::invalid_argument);
    EXPECT_EQ(f.calls, 0u);
}

TEST(V0101Validation, X0OutsideTheBoxThrows) {
    Counting f;
    const std::vector<double> x0{1.0, 1.0};
    flop::cobyla::Options o = good();
    o.bounds = flop::Bounds::none(2);
    o.bounds->upper[0] = 0.5;
    EXPECT_THROW((void)flop::cobyla::minimize(f, x0, o), std::invalid_argument);
    o.bounds = flop::Bounds::none(2);
    o.bounds->lower[1] = 1.5;
    EXPECT_THROW((void)flop::cobyla::minimize(f, x0, o), std::invalid_argument);
    EXPECT_EQ(f.calls, 0u);
}

TEST(V0101Validation, X0OnTheBoundaryIsAccepted) {
    const std::vector<double> x0{0.0, 2.0};
    flop::cobyla::Options o = good();
    const std::vector<double> lo{0.0, -1.0};
    const std::vector<double> hi{1.0, 2.0};
    o.bounds = flop::Bounds::box(lo, hi);
    EXPECT_NO_THROW((void)flop::cobyla::minimize(v0101::sphere, x0, o));
}

TEST(V0101Validation, TheSameChecksGuardEveryEntryPoint) {
    Counting f;
    const std::vector<double> x0;  // empty: rejected everywhere
    auto fb = [&](std::span<const std::span<const double>> xs, std::span<double> out) {
        for (std::size_t i = 0; i < xs.size(); ++i) out[i] = f(xs[i]);
    };
    auto c = [](std::span<const double> x, std::span<double> out) { out[0] = x[0]; };
    EXPECT_THROW((void)flop::cobyla::minimize(f, c, 1, x0, good()), std::invalid_argument);
    EXPECT_THROW((void)flop::cobyla::minimize_batch(fb, x0, good()), std::invalid_argument);
    EXPECT_THROW((void)flop::cobyla::minimize_batch(fb, c, 1, x0, good()), std::invalid_argument);
    const flop::Minimizer m = flop::Minimizer::create("COBYLA");
    const flop::Minimizer::Objective fo = [&](std::span<const double> x) { return f(x); };
    EXPECT_THROW((void)m.minimize(fo, x0, good()), std::invalid_argument);
    EXPECT_EQ(f.calls, 0u);
}

TEST(V0101Validation, UnknownAlgorithmNameThrows) {
    EXPECT_THROW((void)flop::Minimizer::create("NELDERMEAD"), std::invalid_argument);
    EXPECT_THROW((void)flop::Minimizer::create("cobyla"), std::invalid_argument);  // case matters
    EXPECT_THROW((void)flop::Minimizer::create(""), std::invalid_argument);
}

TEST(V0101Validation, ANonFiniteObjectiveValueThrowsMidRun) {
    // The objective hands its value back by value, and that is the one path
    // the check cannot cover under every compiler: with -ffinite-math-only
    // clang declares every double a function returns free of NaN, so a NaN
    // returned there is undefined before the library sees it. GCC keeps the
    // bits and the check catches them. The test stands where the value can
    // arrive and says why where it cannot.
#if defined(__clang__) && defined(__FAST_MATH__)
    GTEST_SKIP() << "clang -ffinite-math-only: a NaN returned by value is poison, see the "
                    "algorithm page's limits";
#else
    std::size_t calls = 0;
    auto f = [&](std::span<const double> x) {
        ++calls;
        if (calls == 3) {
            double nan = 0.0;
            v0101::write_bits(&nan, v0101::kQuietNaN);
            return nan;
        }
        return v0101::sphere(x);
    };
    const std::vector<double> x0{1.0, 1.0};
    EXPECT_THROW((void)flop::cobyla::minimize(f, x0, good()), std::runtime_error);
    EXPECT_EQ(calls, 3u);
#endif
}

TEST(V0101Validation, ANonFiniteConstraintValueThrowsMidRun) {
    // A constraint writes into memory the library owns, so this path is
    // covered under every compiler and model.
    std::size_t calls = 0;
    auto c = [&](std::span<const double> x, std::span<double> out) {
        ++calls;
        out[0] = 1.0 - x[0] * x[0];
        if (calls == 2) v0101::write_bits(&out[0], v0101::kInfinity);
    };
    const std::vector<double> x0{0.5, 0.5};
    EXPECT_THROW((void)flop::cobyla::minimize(v0101::sphere, c, 1, x0, good()), std::runtime_error);
    EXPECT_EQ(calls, 2u);
}
