// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: the range of coordinates BOBYQA evaluates.
//
// Nothing non-finite is ever handed to the objective. An initial_step for
// which x0 +- initial_step overflows is refused at entry, and so is one beyond
// DBL_MAX^(1/4) / (4 n^2), the range in which the arithmetic of the model
// stays finite (docs/api/bobyqa.md), unless shrink_initial_step brings the
// first radius inside it. A run that would evaluate a point beyond the largest
// double ends with std::runtime_error, and one whose model leaves the range
// of a double ends with RoundoffLimited; in practice both mean an objective
// unbounded below.
//
// The tests that end a run on a thread of its own are given a deadline: a run
// that does not end within it fails its test instead of holding the suite.

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
    bool refused = false;                             // std::invalid_argument at entry
    bool errored = false;                             // std::runtime_error during the run
    bool returned = false;                            // a Result came back
    flop::Status status = flop::Status::XtolReached;  // its status, when one did
    bool result_finite = true;                        // its point and value are finite
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
            state->status = r.status;
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
    // initial_step = max - |x0| exactly: both sums are finite, but the step is
    // far beyond the range limit, so it is refused all the same.
    Watch f;
    flop::bobyqa::Options o = v0121::options(0.5 * kMax, 1);
    v0121::expect_rejected_everywhere(Vec{0.5 * kMax}, o, "beyond the range limit");
    EXPECT_THROW((void)flop::bobyqa::minimize(f, Vec{0.5 * kMax}, o), std::invalid_argument);
    EXPECT_EQ(f.calls, 0u) << "the objective was called";
}

TEST(V0121Range, TheFirstRadiusIsRefusedBeyondTheRangeLimitAndAcceptedAtIt) {
    // The limit is DBL_MAX^(1/4) / (4 n^2). At it the run is accepted (with
    // the cap at one evaluation it evaluates x0 alone); one step of the
    // spacing beyond it, or twice it, is refused at entry in every channel.
    for (const std::size_t n :
         {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{5}, std::size_t{20}}) {
        const auto count = static_cast<double>(n);
        const double limit = std::sqrt(std::sqrt(kMax)) / (4.0 * count * count);
        const Vec x0(n, 0.0);
        Watch f;
        flop::bobyqa::Options o = v0121::options(limit, 1);
        EXPECT_NO_THROW((void)flop::bobyqa::minimize(f, x0, o)) << "n " << n;
        EXPECT_EQ(f.calls, 1u) << "n " << n;
        o.initial_step = std::nextafter(limit, kMax);
        v0121::expect_rejected_everywhere(x0, o, "one spacing beyond the range limit");
        o.initial_step = 2.0 * limit;
        v0121::expect_rejected_everywhere(x0, o, "twice the range limit");
    }
}

TEST(V0121Range, ShrinkInitialStepBringsAStepBeyondTheLimitIntoItOnlyThroughANarrowBox) {
    // The first radius is the smaller of initial_step and half the narrowest
    // box, so a step beyond the limit is accepted when that is within it.
    Watch f;
    flop::bobyqa::Options o = v0121::options(1e200, 1);
    o.shrink_initial_step = true;
    o.bounds = flop::Bounds::box(Vec{-1.0}, Vec{1.0});
    EXPECT_NO_THROW((void)flop::bobyqa::minimize(f, Vec{0.0}, o));
    EXPECT_EQ(f.calls, 1u);
    // A box as wide as the step leaves the radius beyond the limit.
    o.bounds = flop::Bounds::box(Vec{-1e200}, Vec{1e200});
    v0121::expect_rejected_everywhere(Vec{0.0}, o, "a first radius beyond the range limit");
    // Without shrink_initial_step the narrow box is refused as before.
    o.shrink_initial_step = false;
    o.bounds = flop::Bounds::box(Vec{-1.0}, Vec{1.0});
    v0121::expect_rejected_everywhere(Vec{0.0}, o, "a box narrower than 2 initial_step");
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
          Case{.n = 3, .step = 1e70, .batch = false}, Case{.n = 2, .step = 1.0, .batch = true}}) {
        const auto seen = observe(c.n, c.step, 0, c.batch, kDeadline);
        ASSERT_NE(seen, nullptr) << "the run did not end: n " << c.n << " step " << c.step
                                 << (c.batch ? " batch" : "");
        EXPECT_EQ(seen->f.non_finite, 0u) << "n " << c.n << " step " << c.step;
        EXPECT_TRUE(seen->result_finite) << "n " << c.n << " step " << c.step;
        EXPECT_TRUE(seen->returned || seen->errored) << "n " << c.n << " step " << c.step;
    }
}

TEST(V0121Range, AnObjectiveFallingWithoutBoundEndsBeforeTheModelsArithmeticLeavesItsRange) {
    // The matrices of the model hold terms of the size of the fourth power of
    // the distances between the interpolation points. With the trust region
    // doubling at every step on f = -x[0], they overflow near DBL_MAX^(1/4),
    // about 1.2e77, long before a coordinate does; the run ends there with
    // RoundoffLimited, as one whose predicted reduction is not positive, and
    // does so whether or not the compiler assumes that no value is NaN.
    const double range = 16.0 * std::pow(kMax, 0.25);
    for (const std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4}}) {
        for (const bool batch : {false, true}) {
            const auto seen = observe(n, 1.0, 0, batch, kDeadline);
            ASSERT_NE(seen, nullptr) << "the run did not end: n " << n;
            ASSERT_TRUE(seen->returned) << "n " << n;
            EXPECT_EQ(seen->status, flop::Status::RoundoffLimited) << "n " << n;
            EXPECT_EQ(seen->f.non_finite, 0u) << "n " << n;
            EXPECT_LT(seen->f.largest, range) << "n " << n;
        }
    }
}

TEST(V0121Range, AnUnboundedObjectiveOfTinyScaleEndsToo) {
    // The same objective scaled by 2^-1023, so its values stay denormal while
    // the points grow, from a first step inside the range limit and from one
    // far beyond it, which is refused at entry.
    for (const std::size_t n : {std::size_t{1}, std::size_t{2}}) {
        for (const double step : {1e70, 1e200}) {
            const auto seen = observe(n, step, -1023, false, kDeadline);
            ASSERT_NE(seen, nullptr) << "the run did not end: n " << n << " step " << step;
            EXPECT_EQ(seen->refused, step > 1e100) << "n " << n << " step " << step;
            EXPECT_EQ(seen->f.non_finite, 0u) << "n " << n << " step " << step;
            EXPECT_TRUE(seen->result_finite) << "n " << n << " step " << step;
        }
    }
}

TEST(V0121Range, ALargeStepIsNeverTurnedIntoANonFinitePoint) {
    // A step above sqrt(DBL_MAX) passes the check that x0 +- step is finite,
    // and is beyond the range limit: it is refused at entry, and nothing is
    // evaluated.
    for (const std::size_t n : {std::size_t{1}, std::size_t{2}}) {
        for (const double step : {1e155, 1e200, 1e290}) {
            const auto seen = observe(n, step, 0, false, kDeadline);
            ASSERT_NE(seen, nullptr) << "the run did not end: n " << n << " step " << step;
            EXPECT_TRUE(seen->refused) << "n " << n << " step " << step;
            EXPECT_EQ(seen->f.calls, 0u) << "n " << n << " step " << step;
            EXPECT_EQ(seen->f.non_finite, 0u) << "n " << n << " step " << step;
        }
    }
}
