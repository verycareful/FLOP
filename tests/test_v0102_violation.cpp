// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.0.2: the reported violation never carries a negative zero.
//
// max_constraint_violation is max(0, -min_i c_i(x)). A constraint that is
// exactly zero contributes -c_i = -0.0, and a maximum taken by a floating-
// point instruction can hand that sign back: under -ffast-math the
// compiler may assume the sign of a zero does not matter and pick the
// operand that carries it. The 0.1.0.0 smoke printed "-0.00e+00" for
// exactly that reason. The pin is on the bits, in both binaries.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "flop/detail/fp.hpp"
#include "flop/flop.hpp"
#include "v0101_problems.hpp"

namespace {

// A constraint that is exactly zero everywhere: satisfied, active, and the
// only value whose negation has a sign the maximum could keep.
void zero_constraint(std::span<const double>, std::span<double> out) {
    out[0] = 0.0;
}

TEST(V0102Violation, AnExactlyZeroConstraintReportsPositiveZero) {
    const std::vector<double> x0{0.4, 0.2};
    flop::cobyla::Options opts = v0101::options(0.3, 1e-9, 200);
    const flop::Result r = flop::cobyla::minimize(v0101::sphere, zero_constraint, 1, x0, opts);
    EXPECT_TRUE(v0101::same_bits(r.max_constraint_violation, 0.0)) << "bits carry a negative zero";
}

// The same promise when the cap ends the run at the best point seen, which
// is reported through a different path.
TEST(V0102Violation, TheCapReportsPositiveZeroToo) {
    const std::vector<double> x0{0.4, 0.2};
    flop::cobyla::Options opts = v0101::options(0.3, 1e-9, 5);
    const flop::Result r = flop::cobyla::minimize(v0101::sphere, zero_constraint, 1, x0, opts);
    ASSERT_EQ(r.status, flop::Status::MaxEvaluationsReached);
    EXPECT_TRUE(v0101::same_bits(r.max_constraint_violation, 0.0)) << "bits carry a negative zero";
}

// The helper behind the promise, on bit patterns written into memory: a
// negative zero becomes a positive one, and nothing else changes.
TEST(V0102Violation, DropNegativeZeroTouchesOnlyTheNegativeZero) {
    double x = 0.0;
    v0101::write_bits(&x, UINT64_C(0x8000000000000000));
    flop::detail::drop_negative_zero(x);
    EXPECT_TRUE(v0101::same_bits(x, 0.0));

    v0101::write_bits(&x, UINT64_C(0));
    flop::detail::drop_negative_zero(x);
    EXPECT_TRUE(v0101::same_bits(x, 0.0));

    const double y = -0.25;
    x = y;
    flop::detail::drop_negative_zero(x);
    EXPECT_TRUE(v0101::same_bits(x, y));
}

}  // namespace
