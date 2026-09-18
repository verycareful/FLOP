// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.0.1 test wave: the floating-point discipline and determinism.
//
// fp_bad reads the exponent field of a double it takes by reference, so it
// classifies a NaN or an infinity that sits in memory whatever the compiler
// has been told to assume about arithmetic. The values under test are
// written into memory as bit patterns and never pass through a double by
// value: under -ffinite-math-only clang declares every double a function
// returns or takes by value free of NaN and infinity, and a NaN produced by
// arithmetic is undefined by the same rule, so neither is a thing a test
// can hold. The test runs in both binaries; the fast-math one is the one
// that matters.
//
// Determinism is the other promise of the discipline: the same inputs give
// the same trajectory to the bit, run after run, in the same binary.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "flop/detail/fp.hpp"
#include "flop/flop.hpp"
#include "v0101_problems.hpp"

namespace {

// The value under test is written into memory and read by reference; it is
// never a double passed or returned by value, which under clang's
// -ffinite-math-only would make it poison.
bool bad_bits(std::uint64_t bits) {
    double d = 0.0;
    v0101::write_bits(&d, bits);
    return flop::detail::fp_bad(d);
}

}  // namespace

TEST(V0101Fp, NaNAndInfinityAreBad) {
    EXPECT_TRUE(bad_bits(v0101::kQuietNaN));
    EXPECT_TRUE(bad_bits(v0101::kInfinity));
    EXPECT_TRUE(bad_bits(v0101::kNegInfinity));
    // Every NaN payload, both signs, the quiet bit set or clear.
    EXPECT_TRUE(bad_bits(UINT64_C(0x7FF0000000000001)));
    EXPECT_TRUE(bad_bits(UINT64_C(0x7FF4000000000000)));
    EXPECT_TRUE(bad_bits(UINT64_C(0xFFF8000000000000)));
    EXPECT_TRUE(bad_bits(UINT64_C(0x7FFFFFFFFFFFFFFF)));
    EXPECT_TRUE(bad_bits(UINT64_C(0xFFFFFFFFFFFFFFFF)));
}

TEST(V0101Fp, EveryFiniteValueIsGood) {
    using L = std::numeric_limits<double>;
    EXPECT_FALSE(flop::detail::fp_bad(0.0));
    EXPECT_FALSE(flop::detail::fp_bad(-0.0));
    EXPECT_FALSE(flop::detail::fp_bad(1.0));
    EXPECT_FALSE(flop::detail::fp_bad(-1.0));
    EXPECT_FALSE(flop::detail::fp_bad(L::max()));
    EXPECT_FALSE(flop::detail::fp_bad(L::lowest()));
    EXPECT_FALSE(flop::detail::fp_bad(L::min()));
    EXPECT_FALSE(flop::detail::fp_bad(L::denorm_min()));
    EXPECT_FALSE(flop::detail::fp_bad(L::epsilon()));
    // The largest exponent below the all-ones field: finite.
    EXPECT_FALSE(bad_bits(UINT64_C(0x7FEFFFFFFFFFFFFF)));
    EXPECT_FALSE(bad_bits(UINT64_C(0x0010000000000000)));
}

TEST(V0101Fp, AnyBadFindsOneBadEntryAnywhere) {
    std::vector<double> v(10, 1.0);
    EXPECT_FALSE(flop::detail::any_bad(v));
    v0101::write_bits(&v[7], v0101::kQuietNaN);
    EXPECT_TRUE(flop::detail::any_bad(v));
    v[7] = 1.0;
    v0101::write_bits(&v[0], v0101::kInfinity);
    EXPECT_TRUE(flop::detail::any_bad(v));
    v[0] = 1.0;
    v0101::write_bits(&v[9], v0101::kNegInfinity);
    EXPECT_TRUE(flop::detail::any_bad(v));
    const std::vector<double> empty;
    EXPECT_FALSE(flop::detail::any_bad(empty));
}

TEST(V0101Fp, TheSameInputsGiveTheSameTrajectoryToTheBit) {
    const std::vector<double> x0{-1.2, 1.0};
    std::vector<std::vector<double>> first, second;
    std::vector<double> f_first, f_second;
    flop::cobyla::Options o1 = v0101::options(0.5, 1e-9, 5000);
    o1.on_evaluation = [&](const flop::Evaluation& e) {
        first.emplace_back(e.x.begin(), e.x.end());
        f_first.push_back(e.f);
    };
    flop::cobyla::Options o2 = v0101::options(0.5, 1e-9, 5000);
    o2.on_evaluation = [&](const flop::Evaluation& e) {
        second.emplace_back(e.x.begin(), e.x.end());
        f_second.push_back(e.f);
    };
    const flop::Result a = flop::cobyla::minimize(v0101::rosenbrock, x0, o1);
    const flop::Result b = flop::cobyla::minimize(v0101::rosenbrock, x0, o2);
    ASSERT_EQ(first.size(), second.size());
    for (std::size_t k = 0; k < first.size(); ++k) {
        ASSERT_TRUE(v0101::same_bits(first[k], second[k])) << k;
        ASSERT_TRUE(v0101::same_bits(f_first[k], f_second[k])) << k;
    }
    EXPECT_TRUE(v0101::same_bits(a.x, b.x));
    EXPECT_TRUE(v0101::same_bits(a.f, b.f));
    EXPECT_EQ(a.status, b.status);
    EXPECT_EQ(a.evaluations, b.evaluations);
}

TEST(V0101Fp, ConstrainedRunsAreDeterministicToo) {
    const v0101::Constrained p = v0101::powell_1994_problems()[8];  // HS100
    const flop::Result a =
        flop::cobyla::minimize(p.f, p.c, p.m, p.x0, v0101::options(0.5, 2e-9, 20000));
    const flop::Result b =
        flop::cobyla::minimize(p.f, p.c, p.m, p.x0, v0101::options(0.5, 2e-9, 20000));
    EXPECT_TRUE(v0101::same_bits(a.x, b.x));
    EXPECT_TRUE(v0101::same_bits(a.f, b.f));
    EXPECT_EQ(a.evaluations, b.evaluations);
    EXPECT_EQ(a.status, b.status);
}

TEST(V0101Fp, TheInitialSimplexIsASingleAdditionPerCoordinate) {
    // x0 + rho e_i is one addition, which no floating-point model can
    // reassociate, so the first n + 1 points are the same bits in both
    // binaries and in any consumer's build.
    const std::vector<double> x0{0.123456789, -0.987654321, 0.75, 0.0};
    const double rho = 0.3;
    std::vector<std::vector<double>> points;
    flop::cobyla::Options o = v0101::options(rho, 1e-9, 5);
    o.on_evaluation = [&](const flop::Evaluation& e) {
        points.emplace_back(e.x.begin(), e.x.end());
    };
    (void)flop::cobyla::minimize(v0101::sphere, x0, o);
    ASSERT_EQ(points.size(), 5u);
    EXPECT_TRUE(v0101::same_bits(points[0], x0));
    // The base walks to any vertex that improves, so each poke is one
    // addition on the base of the moment.
    std::vector<double> base = x0;
    double f_base = v0101::sphere(base);
    for (std::size_t i = 0; i < 4; ++i) {
        std::vector<double> expected = base;
        expected[i] += rho;
        EXPECT_TRUE(v0101::same_bits(points[i + 1], expected)) << i;
        const double f = v0101::sphere(points[i + 1]);
        if (f < f_base) {
            base = points[i + 1];
            f_base = f;
        }
    }
}
