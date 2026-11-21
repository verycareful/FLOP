// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: flop/minimizer.hpp stands alone, BOBYQA included.
//
// The facade header does not include flop/bobyqa.hpp, so a caller of the facade
// compiles none of BOBYQA's templates. This translation unit includes nothing
// of FLOP but that header and asserts at compile time that BOBYQA's options
// type is not defined here: if the header ever pulls BOBYQA in, by any route,
// this file stops compiling. It must not include flop/flop.hpp or the shared
// test headers, which include everything.

#include <gtest/gtest.h>

#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

#include "flop/minimizer.hpp"

namespace flop::bobyqa {
struct Options;  // a redeclaration if the type were already defined
}  // namespace flop::bobyqa

namespace {

template <class T, class = void>
constexpr bool kComplete = false;
template <class T>
constexpr bool kComplete<T, std::void_t<decltype(sizeof(T))>> = true;

static_assert(!kComplete<flop::bobyqa::Options>,
              "flop/minimizer.hpp pulls in BOBYQA's definitions");

}  // namespace

TEST(V0121Facade, TheFacadeHeaderAloneRunsBobyqa) {
    flop::Minimizer m = flop::Minimizer::create("BOBYQA");
    m.set_interpolation_points(0);
    flop::Options o;
    o.stopping.xtol_abs = 1e-9;
    o.stopping.max_evaluations = 10000;
    o.initial_step = 0.5;
    const std::vector<double> x0{1.0, -1.0};
    const flop::Result r = m.minimize(
        [](std::span<const double> x) { return (x[0] - 0.5) * (x[0] - 0.5) + x[1] * x[1]; }, x0, o);
    EXPECT_EQ(r.status, flop::Status::XtolReached);
    EXPECT_NEAR(r.x[0], 0.5, 1e-7);
    EXPECT_NEAR(r.x[1], 0.0, 1e-7);
}
