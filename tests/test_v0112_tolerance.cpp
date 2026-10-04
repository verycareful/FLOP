// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.1.2: the f tolerance tests at the ends of the finite range.
//
// Both algorithms compare a difference of two objective values a - b with
// ftol_abs or with ftol_rel |b| (flop/detail/tolerance.hpp): Nelder-Mead's
// spread, COBYLA's fall. Two finite values can differ by more than the
// largest double, and ftol_rel |b| can exceed it when ftol_rel > 1; forming
// either is an infinity, undefined under -ffinite-math-only. Both were formed
// before this release. The helpers decide each inequality without them, and
// inside the range with ftol_rel <= 1 they are the plain comparison.
//
// Every expectation below is exact: the values are powers of two times
// the largest double, or small dyadic numbers, and every relative tolerance
// above 1 is a power of two, so each division is exact as well.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "flop/detail/tolerance.hpp"

namespace {

constexpr double kMax = std::numeric_limits<double>::max();

using flop::detail::difference_within_abs;
using flop::detail::difference_within_rel;

}  // namespace

TEST(V0112Tolerance, ADifferenceAboveTheRangeExceedsEveryFiniteAbsoluteTolerance) {
    // DBL_MAX - (-DBL_MAX) = 2 DBL_MAX.
    for (const double t : {0.0, 1.0, kMax})
        EXPECT_FALSE(difference_within_abs(kMax, -kMax, t)) << t;
    // 0.75 DBL_MAX - (-0.5 DBL_MAX) = 1.25 DBL_MAX.
    EXPECT_FALSE(difference_within_abs(0.75 * kMax, -0.5 * kMax, kMax));
}

TEST(V0112Tolerance, ADifferenceBelowTheRangeIsWithinEveryTolerance) {
    // -DBL_MAX - DBL_MAX = -2 DBL_MAX, below every tolerance >= 0.
    EXPECT_TRUE(difference_within_abs(-kMax, kMax, 0.0));
    EXPECT_TRUE(difference_within_rel(-kMax, kMax, 0.5));
    EXPECT_TRUE(difference_within_rel(-kMax, kMax, 4.0));
}

TEST(V0112Tolerance, ARelativeToleranceDecidesBeyondTheRangeExactly) {
    // a - b = 2 DBL_MAX against r |b| = r DBL_MAX: within exactly for r >= 2.
    EXPECT_FALSE(difference_within_rel(kMax, -kMax, 0.5));
    EXPECT_FALSE(difference_within_rel(kMax, -kMax, 1.0));
    EXPECT_TRUE(difference_within_rel(kMax, -kMax, 2.0));
    EXPECT_TRUE(difference_within_rel(kMax, -kMax, 4.0));
    // a - b = 1.5 DBL_MAX against r * 0.5 DBL_MAX: within exactly for r >= 3,
    // so 2 is out and 4 is in.
    EXPECT_FALSE(difference_within_rel(kMax, -0.5 * kMax, 2.0));
    EXPECT_TRUE(difference_within_rel(kMax, -0.5 * kMax, 4.0));
}

TEST(V0112Tolerance, ARelativeToleranceAboveOneNeverFormsItsProduct) {
    // b = -DBL_MAX / 2, a = 0: the difference DBL_MAX / 2 is in range, but
    // 4 |b| = 2 DBL_MAX is not. Within: DBL_MAX / 2 <= 2 DBL_MAX.
    EXPECT_TRUE(difference_within_rel(0.0, -0.5 * kMax, 4.0));
    // The boundary: a = DBL_MAX / 4, b = -DBL_MAX / 4 gives a - b = DBL_MAX / 2
    // = 2 |b|, within at r = 2; with b = -DBL_MAX / 8 the difference is
    // 0.375 DBL_MAX against 0.25 DBL_MAX, out.
    EXPECT_TRUE(difference_within_rel(0.25 * kMax, -0.25 * kMax, 2.0));
    EXPECT_FALSE(difference_within_rel(0.25 * kMax, -0.125 * kMax, 2.0));
}

TEST(V0112Tolerance, InsideTheRangeTheTestsAreThePlainComparisons) {
    std::vector<double> values;
    for (int k = -16; k <= 16; ++k) values.push_back(0.25 * k);
    const std::vector<double> tolerances{0.0, 0.25, 0.5, 1.0, 3.0};
    const std::vector<double> rel{0.25, 0.5, 1.0, 2.0, 4.0};
    for (const double a : values) {
        for (const double b : values) {
            for (const double t : tolerances)
                EXPECT_EQ(difference_within_abs(a, b, t), a - b <= t) << a << " " << b << " " << t;
            for (const double r : rel)
                EXPECT_EQ(difference_within_rel(a, b, r), a - b <= r * std::fabs(b))
                    << a << " " << b << " " << r;
        }
    }
}
