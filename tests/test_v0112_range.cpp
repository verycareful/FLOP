// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.1.2: the range of coordinates the methods evaluate.
//
// Nothing non-finite is ever handed to the objective. Both algorithms
// refuse at entry an initial_step for which x0 +- initial_step overflows.
// Nelder-Mead evaluates no coordinate beyond DBL_MAX / (n + 5), the range in
// which its running sum, centroid and trial points cannot overflow: an x0 or
// an initial vertex beyond it is refused at entry (both are input), and a
// run that would evaluate a point beyond it ends with std::runtime_error,
// which in practice means an objective unbounded below.

#include <gtest/gtest.h>

#include <algorithm>
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

constexpr double kMax = std::numeric_limits<double>::max();

// The library's own value is used where a test sits exactly on the limit, so
// that both sides of the comparison are the same rounded quotient whatever
// the floating-point model makes of the division.
double range_limit(std::size_t n) {
    return flop::detail::nelder_mead_range_limit(n);
}

// Decreasing in x[0] without bound, scaled by a power of two so the values
// stay finite and exact at the top of the range.
double falling(std::span<const double> x) {
    return -std::ldexp(x[0], -1023);
}

// Records the largest coordinate magnitude it was handed and whether any
// coordinate was non-finite, read from memory by reference.
struct Watch {
    double largest = 0.0;
    std::size_t non_finite = 0;
    double operator()(std::span<const double> x) {
        if (flop::detail::any_bad(x)) {
            ++non_finite;
            return 0.0;
        }
        for (const double& v : x) largest = std::max(largest, std::fabs(v));
        return falling(x);
    }
};

}  // namespace

TEST(V0112Range, AStepThatOverflowsX0IsRefusedByBothAlgorithms) {
    for (const double x0 : {0.5 * kMax, -0.5 * kMax}) {
        Watch f;
        flop::nelder_mead::Options nm = v0111::options(0.75 * kMax, 10);
        EXPECT_THROW((void)flop::nelder_mead::minimize(f, Vec{x0}, nm), std::invalid_argument);
        flop::cobyla::Options co;
        co.initial_step = 0.75 * kMax;
        co.stopping.max_evaluations = 10;
        EXPECT_THROW((void)flop::cobyla::minimize(f, Vec{x0}, co), std::invalid_argument);
        EXPECT_EQ(f.largest, 0.0) << "the objective was called";
    }
    // initial_step = max - |x0| exactly: both sums are finite, so it is
    // accepted. One evaluation, x0 itself.
    flop::cobyla::Options co;
    co.initial_step = 0.5 * kMax;
    co.stopping.max_evaluations = 1;
    Watch f;
    EXPECT_NO_THROW((void)flop::cobyla::minimize(f, Vec{0.5 * kMax}, co));
    EXPECT_EQ(f.non_finite, 0u);
}

TEST(V0112Range, NelderMeadRefusesAnX0BeyondItsRange) {
    // DBL_MAX / 4 is beyond DBL_MAX / 6 and DBL_MAX / 8. Written as a
    // product with DBL_MAX rather than a multiple of the limit: under
    // -ffast-math the compiler may fold 2 (DBL_MAX / k) into (2 DBL_MAX) / k,
    // an infinity, which -ffinite-math-only makes undefined.
    for (const std::size_t n : {std::size_t{1}, std::size_t{3}}) {
        Vec x0(n, 0.0);
        x0[n - 1] = 0.25 * kMax;
        Watch f;
        EXPECT_THROW((void)flop::nelder_mead::minimize(f, x0, v0111::options(1.0, 10)),
                     std::invalid_argument)
            << "n " << n;
        auto g = v0111::batched(falling);
        EXPECT_THROW((void)flop::nelder_mead::minimize_batch(g, x0, v0111::options(1.0, 10)),
                     std::invalid_argument)
            << "n " << n;
        EXPECT_EQ(f.largest, 0.0);
    }
    // x0 at the limit itself is inside the range. With the upper bound there,
    // x0 + h is outside the box and the vertex is x0 - h, inside the range.
    const double limit = range_limit(1);
    flop::nelder_mead::Options o = v0111::options(0.5 * limit, 2);
    o.bounds = flop::Bounds::box(Vec{-limit}, Vec{limit});
    Watch f;
    EXPECT_NO_THROW((void)flop::nelder_mead::minimize(f, Vec{limit}, o));
    EXPECT_EQ(f.largest, limit);
}

TEST(V0112Range, NelderMeadRefusesAnInitialVertexBeyondItsRange) {
    // x0 = 0, h = DBL_MAX / 4, beyond DBL_MAX / 6 at n = 1.
    Watch f;
    EXPECT_THROW((void)flop::nelder_mead::minimize(f, Vec{0.0}, v0111::options(0.25 * kMax, 10)),
                 std::invalid_argument);
    // With an upper bound at 1 the vertex is x0 - h, still beyond.
    flop::nelder_mead::Options o = v0111::options(0.25 * kMax, 10);
    flop::Bounds b = flop::Bounds::none(1);
    b.upper[0] = 1.0;
    o.bounds = b;
    EXPECT_THROW((void)flop::nelder_mead::minimize(f, Vec{0.0}, o), std::invalid_argument);
    // With both bounds at -1 and 1, neither x0 + h nor x0 - h fits and the
    // vertex is the bound with more room (the upper on a tie), 1.
    o.bounds = flop::Bounds::box(Vec{-1.0}, Vec{1.0});
    o.stopping.max_evaluations = 2;
    EXPECT_NO_THROW((void)flop::nelder_mead::minimize(f, Vec{0.0}, o));
    EXPECT_EQ(f.largest, 1.0);
    EXPECT_EQ(f.non_finite, 0u);
}

TEST(V0112Range, AnObjectiveUnboundedBelowEndsWithAnErrorInsideTheRange) {
    // From 0 with h = 1, f falls without bound along x[0]: the method keeps
    // expanding, and the first point it would evaluate beyond the range ends
    // the run. Every point evaluated is inside the range, and the two
    // channels evaluate the same points before the error.
    for (const std::size_t n : {std::size_t{1}, std::size_t{2}}) {
        const Vec x0(n, 0.0);
        v0111::Trace scalar_trace;
        Watch f;
        flop::nelder_mead::Options o = v0111::options(1.0, 100000);
        o.on_evaluation = v0111::recorder(scalar_trace);
        EXPECT_THROW((void)flop::nelder_mead::minimize(f, x0, o), std::runtime_error) << "n " << n;
        EXPECT_EQ(f.non_finite, 0u);
        EXPECT_LE(f.largest, range_limit(n)) << "n " << n;
        EXPECT_LT(scalar_trace.size(), 100000u);

        v0111::Trace batch_trace;
        auto g = v0111::batched(falling);
        flop::nelder_mead::Options ob = v0111::options(1.0, 100000);
        ob.on_evaluation = v0111::recorder(batch_trace);
        EXPECT_THROW((void)flop::nelder_mead::minimize_batch(g, x0, ob), std::runtime_error)
            << "n " << n;
        ASSERT_EQ(batch_trace.size(), scalar_trace.size()) << "n " << n;
        for (std::size_t k = 0; k < scalar_trace.size(); ++k)
            ASSERT_TRUE(v0111::same_bits(batch_trace[k].x, scalar_trace[k].x))
                << "n " << n << ", " << k;
    }
}
