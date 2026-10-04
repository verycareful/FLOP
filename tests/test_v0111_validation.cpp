// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.1.1: Nelder-Mead's input validation and run-time errors.
//
// Every std::invalid_argument docs/api/nelder_mead.md lists, on both entry
// points and through the facade, each with the guarantee that goes with it:
// a rejected call has evaluated nothing. Then the two things that end a
// well-formed run with an exception: a non-finite objective value
// (std::runtime_error) on both channels, and an exception thrown by the
// objective or by the on_evaluation callback, which propagates unchanged.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

#include "flop/flop.hpp"
#include "v0111_problems.hpp"

namespace {

using Vec = std::vector<double>;

struct Counting {
    std::size_t calls = 0;
    double operator()(std::span<const double> x) {
        ++calls;
        return v0111::sphere(x);
    }
};

flop::nelder_mead::Options good() {
    flop::nelder_mead::Options o = v0111::options(0.5, 100);
    o.stopping.xtol_rel = 1e-6;
    return o;
}

// Calls every entry point that takes an objective without constraints with
// the same x0 and options, expects each to throw std::invalid_argument, and
// expects no evaluation from any of them.
void expect_rejected_everywhere(const Vec& x0, const flop::nelder_mead::Options& o,
                                const char* what) {
    Counting f;
    auto fb = [&f](std::span<const std::span<const double>> xs, std::span<double> out) {
        for (std::size_t i = 0; i < xs.size(); ++i) out[i] = f(xs[i]);
    };
    EXPECT_THROW((void)flop::nelder_mead::minimize(f, x0, o), std::invalid_argument) << what;
    EXPECT_THROW((void)flop::nelder_mead::minimize_batch(fb, x0, o), std::invalid_argument) << what;
    const flop::Minimizer m = flop::Minimizer::create("NELDER_MEAD");
    const flop::Minimizer::Objective fo = [&f](std::span<const double> x) { return f(x); };
    const flop::Minimizer::BatchObjective fbo = fb;
    const flop::Options& shared = o;
    EXPECT_THROW((void)m.minimize(fo, x0, shared), std::invalid_argument) << what;
    EXPECT_THROW((void)m.minimize(fbo, x0, shared), std::invalid_argument) << what;
    EXPECT_EQ(f.calls, 0u) << what;
}

}  // namespace

TEST(V0111Validation, EmptyX0IsRejected) {
    expect_rejected_everywhere(Vec{}, good(), "empty x0");
}

TEST(V0111Validation, NonFiniteX0IsRejected) {
    for (const std::uint64_t bits : {v0111::kQuietNaN, v0111::kInfinity, v0111::kNegInfinity}) {
        Vec x0{1.0, 1.0};
        v0111::write_bits(&x0[1], bits);
        expect_rejected_everywhere(x0, good(), "non-finite x0");
    }
}

TEST(V0111Validation, NoStoppingCriterionIsRejected) {
    flop::nelder_mead::Options o;  // every criterion at its off default
    expect_rejected_everywhere(Vec{1.0, 1.0}, o, "no stopping criterion");
}

TEST(V0111Validation, EachStoppingCriterionAloneIsEnough) {
    const Vec x0{1.0, 1.0};
    for (int which = 0; which < 6; ++which) {
        flop::nelder_mead::Options o;
        if (which == 0) o.stopping.max_evaluations = 5;
        if (which == 1) o.stopping.xtol_rel = 1e-3;
        if (which == 2) o.stopping.xtol_abs = 1e-3;
        if (which == 3) o.stopping.ftol_rel = 1e-3;
        if (which == 4) o.stopping.ftol_abs = 1e-3;
        if (which == 5) o.stopping.stop_value = 1.0;
        EXPECT_NO_THROW((void)flop::nelder_mead::minimize(v0111::sphere, x0, o)) << which;
    }
}

TEST(V0111Validation, NegativeOrNonFiniteTolerancesAreRejected) {
    const Vec x0{1.0, 1.0};
    for (int which = 0; which < 4; ++which) {
        flop::nelder_mead::Options o = good();
        double* t[4] = {&o.stopping.xtol_rel, &o.stopping.xtol_abs, &o.stopping.ftol_rel,
                        &o.stopping.ftol_abs};
        *t[which] = -1e-9;
        expect_rejected_everywhere(x0, o, "negative tolerance");
        for (const std::uint64_t bits : {v0111::kQuietNaN, v0111::kInfinity}) {
            flop::nelder_mead::Options p = good();
            double* u[4] = {&p.stopping.xtol_rel, &p.stopping.xtol_abs, &p.stopping.ftol_rel,
                            &p.stopping.ftol_abs};
            v0111::write_bits(u[which], bits);
            expect_rejected_everywhere(x0, p, "non-finite tolerance");
        }
    }
    flop::nelder_mead::Options o = good();
    o.stopping.stop_value = 0.0;
    v0111::write_bits(&*o.stopping.stop_value, v0111::kNegInfinity);
    expect_rejected_everywhere(x0, o, "non-finite stop_value");
}

TEST(V0111Validation, InitialStepMustBePositiveAndFinite) {
    const Vec x0{1.0, 1.0};
    for (const double bad : {0.0, -0.5}) {
        flop::nelder_mead::Options o = good();
        o.initial_step = bad;
        expect_rejected_everywhere(x0, o, "non-positive initial_step");
    }
    for (const std::uint64_t bits : {v0111::kInfinity, v0111::kQuietNaN}) {
        flop::nelder_mead::Options o = good();
        v0111::write_bits(&o.initial_step, bits);
        expect_rejected_everywhere(x0, o, "non-finite initial_step");
    }
}

TEST(V0111Validation, MalformedBoundsAreRejected) {
    const Vec x0{1.0, 1.0};
    {
        flop::nelder_mead::Options o = good();
        o.bounds = flop::Bounds::none(3);
        expect_rejected_everywhere(x0, o, "bounds of the wrong length");
    }
    {
        flop::nelder_mead::Options o = good();
        o.bounds = flop::Bounds::none(2);
        o.bounds->upper.emplace_back(1.0);
        expect_rejected_everywhere(x0, o, "lower and upper of different lengths");
    }
    {
        flop::nelder_mead::Options o = good();
        o.bounds = flop::Bounds::box(Vec{0.0, 1.0}, Vec{2.0, 1.0});
        expect_rejected_everywhere(x0, o, "a fixed coordinate");
    }
    {
        flop::nelder_mead::Options o = good();
        o.bounds = flop::Bounds::box(Vec{2.0, 0.0}, Vec{0.0, 2.0});
        expect_rejected_everywhere(x0, o, "lower above upper");
    }
    {
        flop::nelder_mead::Options o = good();
        o.bounds = flop::Bounds::none(2);
        v0111::write_bits(&o.bounds->lower[0].emplace(0.0), v0111::kNegInfinity);
        expect_rejected_everywhere(x0, o, "a non-finite lower bound");
    }
    {
        flop::nelder_mead::Options o = good();
        o.bounds = flop::Bounds::none(2);
        v0111::write_bits(&o.bounds->upper[1].emplace(0.0), v0111::kQuietNaN);
        expect_rejected_everywhere(x0, o, "a non-finite upper bound");
    }
}

TEST(V0111Validation, X0OutsideTheBoxIsRejectedAndOnItsBoundaryAccepted) {
    {
        flop::nelder_mead::Options o = good();
        o.bounds = flop::Bounds::none(2);
        o.bounds->upper[0] = 0.5;
        expect_rejected_everywhere(Vec{1.0, 1.0}, o, "x0 above the box");
    }
    {
        flop::nelder_mead::Options o = good();
        o.bounds = flop::Bounds::none(2);
        o.bounds->lower[1] = 1.5;
        expect_rejected_everywhere(Vec{1.0, 1.0}, o, "x0 below the box");
    }
    flop::nelder_mead::Options o = good();
    o.bounds = flop::Bounds::box(Vec{0.0, -1.0}, Vec{1.0, 2.0});
    EXPECT_NO_THROW((void)flop::nelder_mead::minimize(v0111::sphere, Vec{0.0, 2.0}, o));
}

TEST(V0111Validation, ANonFiniteObjectiveValueEndsTheRunWithRuntimeError) {
    // A NaN or infinity written into the library's memory by a batch call of
    // more than one point is caught under every compiler and model. One
    // handed back by value (the scalar channel, and the batch channel's
    // single-point calls, which return the value it wrote) is undefined
    // under clang's -ffinite-math-only before FLOP sees it; GCC keeps the
    // bits and the check catches them.
    for (const std::uint64_t bits : {v0111::kQuietNaN, v0111::kInfinity}) {
        std::size_t calls = 0;
        auto fb = [&](std::span<const std::span<const double>> xs, std::span<double> out) {
            for (std::size_t i = 0; i < xs.size(); ++i) {
                ++calls;
                out[i] = v0111::sphere(xs[i]);
                if (calls == 2) v0111::write_bits(&out[i], bits);
            }
        };
        EXPECT_THROW((void)flop::nelder_mead::minimize_batch(fb, Vec{1.0, 1.0, 1.0}, good()),
                     std::runtime_error);
        EXPECT_EQ(calls, 4u);  // the whole first call ran
    }
#if defined(__clang__) && defined(__FAST_MATH__)
    GTEST_SKIP() << "clang -ffinite-math-only: a NaN returned by value is poison, see the "
                    "algorithm page's limits";
#else
    for (const std::size_t at : {std::size_t{1}, std::size_t{4}, std::size_t{7}}) {
        std::size_t calls = 0;
        auto f = [&](std::span<const double> x) {
            ++calls;
            if (calls == at) {
                double nan = 0.0;
                v0111::write_bits(&nan, v0111::kQuietNaN);
                return nan;
            }
            return v0111::sphere(x);
        };
        EXPECT_THROW((void)flop::nelder_mead::minimize(f, Vec{1.0, 1.0}, good()),
                     std::runtime_error);
        EXPECT_EQ(calls, at);
        std::size_t bcalls = 0;
        auto fb = [&](std::span<const std::span<const double>> xs, std::span<double> out) {
            for (std::size_t i = 0; i < xs.size(); ++i) {
                ++bcalls;
                out[i] = v0111::sphere(xs[i]);
                if (bcalls == at + 3) v0111::write_bits(&out[i], v0111::kInfinity);
            }
        };
        EXPECT_THROW((void)flop::nelder_mead::minimize_batch(fb, Vec{1.0, 1.0}, good()),
                     std::runtime_error);
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

TEST(V0111Validation, WhatTheObjectiveOrTheCallbackThrowsPropagatesUnchanged) {
    {
        std::size_t calls = 0;
        auto f = [&](std::span<const double> x) -> double {
            if (++calls == 5) throw Thrown(7);
            return v0111::sphere(x);
        };
        try {
            (void)flop::nelder_mead::minimize(f, Vec{1.0, 1.0}, good());
            ADD_FAILURE() << "no exception";
        } catch (const Thrown& t) {
            EXPECT_EQ(t.tag, 7);
        }
        EXPECT_EQ(calls, 5u);
    }
    {
        auto fb = [](std::span<const std::span<const double>>, std::span<double>) {
            throw Thrown(8);
        };
        try {
            (void)flop::nelder_mead::minimize_batch(fb, Vec{1.0, 1.0}, good());
            ADD_FAILURE() << "no exception";
        } catch (const Thrown& t) {
            EXPECT_EQ(t.tag, 8);
        }
    }
    {
        flop::nelder_mead::Options o = good();
        o.on_evaluation = [](const flop::Evaluation& e) {
            if (e.index == 3) throw Thrown(9);
        };
        try {
            (void)flop::nelder_mead::minimize(v0111::sphere, Vec{1.0, 1.0}, o);
            ADD_FAILURE() << "no exception";
        } catch (const Thrown& t) {
            EXPECT_EQ(t.tag, 9);
        }
    }
}
