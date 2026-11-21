// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: the range of coordinates BOBYQA evaluates.
//
// Nothing non-finite is ever handed to the objective. An initial_step for
// which x0 +- initial_step overflows is refused at entry, and a run that
// would evaluate a point beyond the largest double ends with
// std::runtime_error (docs/api/bobyqa.md), which in practice means an
// objective unbounded below.
//
// The tests that end a run on a thread of its own are red on purpose: they
// pin defects, and a run that does not end within the deadline fails its test
// instead of holding the suite.
//   - An unbounded objective, f = -x[0] from the origin with a first step of
//     1, ends under the strict model and does not end under -ffast-math.
//   - For a first step above about 1.3e154 (the square root of the largest
//     double) and an objective whose values stay denormal, a run does not end
//     under either model.
//   - With an initial_step above about 1.3e154, the objective is handed a NaN
//     coordinate at the first trust-region step. The test accepts either
//     honest outcome, a refusal at entry or a run in which every point is
//     finite.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "flop/detail/fp.hpp"
#include "flop/flop.hpp"
#include "v0121_problems.hpp"

namespace {

using v0121::Vec;

constexpr double kMax = std::numeric_limits<double>::max();

// Records the largest coordinate magnitude it was handed and how many
// coordinates were non-finite, read from memory by reference. Decreasing in
// x[0] without bound, at the scale 2^exponent.
struct Watch {
    int exponent = 0;
    double largest = 0.0;
    std::size_t non_finite = 0;
    std::size_t calls = 0;
    double operator()(std::span<const double> x) {
        ++calls;
        if (flop::detail::any_bad(x)) {
            ++non_finite;
            return 0.0;
        }
        for (const double& v : x) largest = std::max(largest, std::fabs(v));
        return -std::ldexp(x[0], exponent);
    }
};

// What a run on its own thread left behind. The thread owns it through a
// shared_ptr, so a run that never ends cannot leave the test reading memory
// that is gone.
struct Observed {
    Watch f;
    bool refused = false;       // std::invalid_argument at entry
    bool errored = false;       // std::runtime_error during the run
    bool returned = false;      // a Result came back
    bool result_finite = true;  // its point and value are finite
};

// Runs minimize (or minimize_batch) on f = -x[0] 2^exponent from the origin in
// n variables and gives it limit to end. Nullptr when it did not.
std::shared_ptr<Observed> observe(std::size_t n, double step, int exponent, bool batch,
                                  std::chrono::milliseconds limit) {
    auto state = std::make_shared<Observed>();
    state->f.exponent = exponent;
    const flop::bobyqa::Options o = v0121::options(step, 100000);
    const bool ended = v0121::finishes_within(limit, [state, n, o, batch] {
        const Vec x0(n, 0.0);
        auto fb = [state](std::span<const std::span<const double>> xs, std::span<double> out) {
            for (std::size_t i = 0; i < xs.size(); ++i) out[i] = state->f(xs[i]);
        };
        try {
            const flop::Result r = batch ? flop::bobyqa::minimize_batch(fb, x0, o)
                                         : flop::bobyqa::minimize(state->f, x0, o);
            state->returned = true;
            state->result_finite = !flop::detail::any_bad(r.x) && !flop::detail::fp_bad(r.f);
        } catch (const std::invalid_argument&) {
            state->refused = true;
        } catch (const std::runtime_error&) {
            state->errored = true;
        }
    });
    return ended ? state : nullptr;
}

constexpr std::chrono::milliseconds kDeadline{5000};

}  // namespace

TEST(V0121Range, AStepThatOverflowsX0IsRefused) {
    for (const double x0 : {0.5 * kMax, -0.5 * kMax}) {
        Watch f;
        flop::bobyqa::Options o = v0121::options(0.75 * kMax, 10);
        v0121::expect_rejected_everywhere(Vec{x0}, o, "x0 + initial_step overflows");
        EXPECT_THROW((void)flop::bobyqa::minimize(f, Vec{x0}, o), std::invalid_argument);
        EXPECT_EQ(f.calls, 0u) << "the objective was called";
    }
    // initial_step = max - |x0| exactly: both sums are finite, so it is
    // accepted; with the cap at one evaluation the run evaluates x0 alone.
    Watch f;
    flop::bobyqa::Options o = v0121::options(0.5 * kMax, 1);
    EXPECT_NO_THROW((void)flop::bobyqa::minimize(f, Vec{0.5 * kMax}, o));
    EXPECT_EQ(f.non_finite, 0u);
    EXPECT_EQ(f.calls, 1u);
}

TEST(V0121Range, AnObjectiveUnboundedBelowEndsWithoutANonFinitePoint) {
    // f = -x[0] falls without bound. The run may end with a status or with
    // std::runtime_error; what it may not do is fail to end, evaluate a point
    // that is not finite, or return a point or a value that is not.
    struct Case {
        std::size_t n;
        double step;
        bool batch;
    };
    for (const Case c :
         {Case{.n = 1, .step = 1.0, .batch = false}, Case{.n = 2, .step = 1e50, .batch = false},
          Case{.n = 3, .step = 1e100, .batch = false}, Case{.n = 2, .step = 1.0, .batch = true}}) {
        const auto seen = observe(c.n, c.step, 0, c.batch, kDeadline);
        ASSERT_NE(seen, nullptr) << "the run did not end: n " << c.n << " step " << c.step
                                 << (c.batch ? " batch" : "");
        EXPECT_EQ(seen->f.non_finite, 0u) << "n " << c.n << " step " << c.step;
        EXPECT_TRUE(seen->result_finite) << "n " << c.n << " step " << c.step;
        EXPECT_TRUE(seen->returned || seen->errored) << "n " << c.n << " step " << c.step;
    }
}

TEST(V0121Range, AnUnboundedObjectiveOfTinyScaleEndsToo) {
    // The same objective scaled by 2^-1023, so its values stay denormal while
    // the points grow, from a first step above sqrt(DBL_MAX).
    for (const std::size_t n : {std::size_t{1}, std::size_t{2}}) {
        const auto seen = observe(n, 1e200, -1023, false, kDeadline);
        ASSERT_NE(seen, nullptr) << "the run did not end: n " << n;
        EXPECT_EQ(seen->f.non_finite, 0u) << "n " << n;
        EXPECT_TRUE(seen->result_finite) << "n " << n;
    }
}

TEST(V0121Range, ALargeStepIsNeverTurnedIntoANonFinitePoint) {
    // Red on purpose, see the file header. A step above sqrt(DBL_MAX) passes
    // validation: x0 +- step is finite.
    for (const std::size_t n : {std::size_t{1}, std::size_t{2}}) {
        for (const double step : {1e155, 1e200, 1e290}) {
            const auto seen = observe(n, step, 0, false, kDeadline);
            ASSERT_NE(seen, nullptr) << "the run did not end: n " << n << " step " << step;
            if (seen->refused) continue;  // refused at entry: nothing was evaluated
            EXPECT_EQ(seen->f.non_finite, 0u) << "n " << n << " step " << step;
        }
    }
}
