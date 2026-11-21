// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: flop/bobyqa.hpp stands alone.
//
// This translation unit includes nothing of FLOP but flop/bobyqa.hpp, and runs
// the method; if the header ever depends on something it does not include,
// this file stops compiling. It must not include flop/flop.hpp or the shared
// test headers, which include everything. That the facade's header does not
// pull BOBYQA in is held in test_v0121_facade_header.cpp.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "flop/bobyqa.hpp"

TEST(V0121Header, TheAlgorithmHeaderAloneRunsBobyqa) {
    flop::bobyqa::Options o;
    o.stopping.xtol_abs = 1e-9;
    o.stopping.max_evaluations = 10000;
    o.initial_step = 0.5;
    const std::vector<double> x0{1.0, -1.0, 2.0};
    const flop::Result r = flop::bobyqa::minimize(
        [](std::span<const double> x) {
            return (x[0] - 0.5) * (x[0] - 0.5) + x[1] * x[1] + (x[2] - 1.0) * (x[2] - 1.0);
        },
        x0, o);
    EXPECT_EQ(r.status, flop::Status::XtolReached);
    EXPECT_NEAR(r.x[0], 0.5, 1e-7);
    EXPECT_NEAR(r.x[1], 0.0, 1e-7);
    EXPECT_NEAR(r.x[2], 1.0, 1e-7);
}

TEST(V0121Header, TheBatchEntryPointNeedsNothingElseEither) {
    flop::bobyqa::Options o;
    o.stopping.xtol_abs = 1e-9;
    o.stopping.max_evaluations = 10000;
    o.initial_step = 0.5;
    const std::vector<double> x0{1.0, -1.0};
    const flop::Result r = flop::bobyqa::minimize_batch(
        [](std::span<const std::span<const double>> xs, std::span<double> out) {
            for (std::size_t i = 0; i < xs.size(); ++i)
                out[i] = (xs[i][0] - 0.5) * (xs[i][0] - 0.5) + xs[i][1] * xs[i][1];
        },
        x0, o);
    EXPECT_EQ(r.status, flop::Status::XtolReached);
    EXPECT_NEAR(r.x[0], 0.5, 1e-7);
    EXPECT_NEAR(r.x[1], 0.0, 1e-7);
}
