// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.1.1: Nelder-Mead's floating-point discipline.
//
// The rules (CLAUDE.md, "Floating point"): the same inputs give the same
// trajectory to the bit; no value is a sentinel, so the largest and smallest
// finite values, signed zeros and subnormals are ordinary objective values;
// and nothing non-finite is ever computed or handed on. The last is pinned
// red for 0.1.1.2: nothing checks a trial point, so a simplex near the top
// of the range hands the objective an infinite coordinate (x0 + h overflows,
// or a reflection does), and COBYLA's initial simplex does the same. A
// spread between values of -DBL_MAX and +DBL_MAX also overflows inside the
// stopping test; that cannot be seen from outside, since the true spread
// exceeds every tolerance too, so the test below pins the behaviour only.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "flop/detail/fp.hpp"
#include "flop/flop.hpp"
#include "v0111_problems.hpp"

namespace {

using Vec = std::vector<double>;
using v0111::Table;
using v0111::Traced;

constexpr double kMax = std::numeric_limits<double>::max();

double staircase(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        s += std::floor(4.0 * std::fabs(x[i] - v0111::sphere_centre(i) - 0.11));
    return s;
}

void expect_identical(const Traced& a, const Traced& b, const char* what) {
    ASSERT_EQ(a.trace.size(), b.trace.size()) << what;
    for (std::size_t k = 0; k < a.trace.size(); ++k) {
        ASSERT_TRUE(v0111::same_bits(a.trace[k].x, b.trace[k].x)) << what << ", evaluation " << k;
        ASSERT_TRUE(v0111::same_bits(a.trace[k].f, b.trace[k].f)) << what << ", evaluation " << k;
    }
    EXPECT_EQ(a.r.status, b.r.status) << what;
    EXPECT_TRUE(v0111::same_bits(a.r.x, b.r.x)) << what;
    EXPECT_TRUE(v0111::same_bits(a.r.final_radius, b.r.final_radius)) << what;
}

// An objective that records whether it was ever handed a non-finite
// coordinate, read from memory by reference, and returns a finite value
// either way so the run goes on to show every point it would evaluate.
struct Watchful {
    double (*f)(std::span<const double>);
    std::size_t non_finite = 0;
    double operator()(std::span<const double> x) {
        if (flop::detail::any_bad(x)) {
            ++non_finite;
            return 0.0;
        }
        return f(x);
    }
};

double falling(std::span<const double> x) {
    // Decreasing in x[0], scaled by a power of two so the values stay finite
    // and exact at the top of the range.
    return -std::ldexp(x[0], -1023);
}

}  // namespace

TEST(V0111Fp, TheSameInputsGiveTheSameTrajectoryToTheBit) {
    for (const bool adaptive : {false, true}) {
        flop::nelder_mead::Options o = v0111::options(0.5, 3000, adaptive);
        o.stopping.xtol_abs = 1e-12;
        expect_identical(v0111::traced(v0111::extended_rosenbrock, Vec{-1.2, 1.0}, o),
                         v0111::traced(v0111::extended_rosenbrock, Vec{-1.2, 1.0}, o),
                         "rosenbrock");
        expect_identical(v0111::traced(staircase, Vec(4, 0.0), o),
                         v0111::traced(staircase, Vec(4, 0.0), o), "staircase");
        flop::nelder_mead::Options b = o;
        b.bounds = flop::Bounds::box(Vec(3, -1.0), Vec(3, 0.25));
        expect_identical(v0111::traced(v0111::sphere, Vec(3, 0.0), b),
                         v0111::traced(v0111::sphere, Vec(3, 0.0), b), "bounded");
        auto f1 = v0111::batched(v0111::powell_singular);
        auto f2 = v0111::batched(v0111::powell_singular);
        expect_identical(v0111::traced_batch(f1, Vec{3.0, -1.0, 0.0, 1.0}, o),
                         v0111::traced_batch(f2, Vec{3.0, -1.0, 0.0, 1.0}, o), "batch");
    }
}

TEST(V0111Fp, TheInitialSimplexIsOneAdditionPerCoordinate) {
    // Values with no short binary expansion, so a rounding difference would
    // show: vertex i + 1 is x0 with x0[i] + h in coordinate i, one addition,
    // which no reassociation can change.
    const Vec x0{0.1, -0.7, 3.3, 1e-3};
    const double h = 0.37;
    const Traced run = v0111::traced(v0111::sphere, x0, v0111::options(h, x0.size() + 1));
    ASSERT_EQ(run.trace.size(), x0.size() + 1);
    EXPECT_TRUE(v0111::same_bits(run.trace[0].x, x0));
    for (std::size_t i = 0; i < x0.size(); ++i) {
        Vec v = x0;
        v[i] = x0[i] + h;
        EXPECT_TRUE(v0111::same_bits(run.trace[i + 1].x, v)) << i;
    }
}

TEST(V0111Fp, TheExtremeFiniteValuesAreOrdinaryValues) {
    {
        // The worst finite value first still becomes the best point seen.
        Table t;
        t.set({0.0, 0.0}, kMax);
        const Traced run = v0111::traced(t, Vec{0.0, 0.0}, v0111::options(1.0, 1));
        EXPECT_TRUE(v0111::same_bits(run.r.f, kMax));
        EXPECT_TRUE(v0111::same_bits(run.r.x, Vec{0.0, 0.0}));
    }
    {
        // A constant objective at DBL_MAX: every comparison ties, every
        // iteration shrinks, and the run ends at the precision floor with x0.
        auto f = [](std::span<const double>) { return kMax; };
        const Traced run = v0111::traced(f, Vec{0.5, 0.5}, v0111::options(1.0, 100000));
        EXPECT_EQ(run.r.status, flop::Status::RoundoffLimited);
        EXPECT_TRUE(v0111::same_bits(run.r.x, Vec{0.5, 0.5}));
    }
    {
        // -DBL_MAX is the best value there is, and is kept.
        Table t;
        t.set({0.0}, 1.0).set({1.0}, -kMax);
        const Traced run = v0111::traced(t, Vec{0.0}, v0111::options(1.0, 10));
        EXPECT_TRUE(v0111::same_bits(run.r.f, -kMax));
        EXPECT_TRUE(v0111::same_bits(run.r.x, Vec{1.0}));
    }
    {
        // Values at both ends: the spread is 2 DBL_MAX, beyond every
        // tolerance, so neither f tolerance stops the run.
        for (const bool relative : {false, true}) {
            Table t;
            t.set({0.0}, -kMax).set({1.0}, kMax);
            flop::nelder_mead::Options o = v0111::options(1.0, 3);
            if (relative)
                o.stopping.ftol_rel = 1.0;
            else
                o.stopping.ftol_abs = 1.0;
            const Traced run = v0111::traced(t, Vec{0.0}, o);
            EXPECT_EQ(run.r.status, flop::Status::MaxEvaluationsReached);
            EXPECT_EQ(run.r.evaluations, 3u);
            EXPECT_TRUE(v0111::same_bits(run.r.f, -kMax));
        }
    }
}

TEST(V0111Fp, SignedZerosAndSubnormalsAreOrdinaryValues) {
    {
        // +0 and -0 compare equal; the first one evaluated is the result,
        // with its own bits.
        for (const double first : {0.0, -0.0}) {
            Table t;
            t.set({0.0}, first).set({1.0}, -first);
            const Traced run = v0111::traced(t, Vec{0.0}, v0111::options(1.0, 2));
            EXPECT_TRUE(v0111::same_bits(run.r.f, first));
            EXPECT_TRUE(v0111::same_bits(run.r.x, Vec{0.0}));
        }
    }
    {
        // An objective whose values are subnormal: the run completes, and the
        // result is one of the evaluated points with the value it returned.
        // (Under -ffast-math the processor may treat subnormals as zero, so
        // which branch each comparison takes is not pinned.)
        auto f = [](std::span<const double> x) { return v0111::sphere(x) * 1e-310; };
        const Traced run = v0111::traced(f, Vec(3, 0.0), v0111::options(1.0, 300));
        bool found = false;
        for (const v0111::Point& p : run.trace)
            found = found || (v0111::same_bits(p.x, run.r.x) && v0111::same_bits(p.f, run.r.f));
        EXPECT_TRUE(found);
        EXPECT_EQ(run.r.evaluations, run.trace.size());
    }
}

TEST(V0111Fp, NoNonFiniteCoordinateIsEverHandedToTheObjective) {
    {
        // x0 + h overflows: x0 = 0.75 DBL_MAX, h = 0.5 DBL_MAX. The initial
        // simplex is input, so both algorithms refuse it at entry, before the
        // objective is called at all.
        Watchful f{.f = falling};
        EXPECT_THROW(
            (void)flop::nelder_mead::minimize(f, Vec{0.75 * kMax}, v0111::options(0.5 * kMax, 10)),
            std::invalid_argument);
        EXPECT_EQ(f.non_finite, 0u) << "nelder_mead, initial simplex";
        Watchful g{.f = falling};
        flop::cobyla::Options co;
        co.initial_step = 0.5 * kMax;
        co.stopping.max_evaluations = 10;
        EXPECT_THROW((void)flop::cobyla::minimize(g, Vec{0.75 * kMax}, co), std::invalid_argument);
        EXPECT_EQ(g.non_finite, 0u) << "cobyla, initial simplex";
    }
    {
        // x0 = 0.6 DBL_MAX, h = 0.3 DBL_MAX: both initial points are finite,
        // but x0 is beyond DBL_MAX / (n + 5), the range in which Nelder-Mead's
        // centroid and trial points cannot overflow, so it is refused too.
        Watchful f{.f = falling};
        EXPECT_THROW(
            (void)flop::nelder_mead::minimize(f, Vec{0.6 * kMax}, v0111::options(0.3 * kMax, 10)),
            std::invalid_argument);
        EXPECT_EQ(f.non_finite, 0u) << "nelder_mead, x0 beyond the range";
    }
    {
        // From 0 with h = 1, f falls without bound: every iteration expands,
        // the simplex grows geometrically, and the run ends with an error at
        // the first point beyond the range instead of handing on an
        // overflowed one.
        Watchful f{.f = falling};
        EXPECT_THROW((void)flop::nelder_mead::minimize(f, Vec{0.0}, v0111::options(1.0, 100000)),
                     std::runtime_error);
        EXPECT_EQ(f.non_finite, 0u) << "nelder_mead, an objective unbounded below";
    }
}
