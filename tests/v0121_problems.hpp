// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// What the BOBYQA suites share: the options a test starts from, an objective
// that counts its calls, and the check that every entry point refuses a
// malformed problem without evaluating anything. The trace, the batch
// wrapper and the bit helpers are the ones the Nelder-Mead suites use.

#pragma once

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "flop/flop.hpp"
#include "v0111_problems.hpp"

namespace v0121 {

using v0111::batched;
using v0111::kInfinity;
using v0111::kNegInfinity;
using v0111::kQuietNaN;
using v0111::max_abs_diff;
using v0111::recorder;
using v0111::same_bits;
using v0111::Trace;
using v0111::write_bits;

using Vec = std::vector<double>;

inline flop::bobyqa::Options options(double initial_step, std::size_t max_evaluations) {
    flop::bobyqa::Options o;
    o.initial_step = initial_step;
    o.stopping.max_evaluations = max_evaluations;
    return o;
}

// Runs body on its own thread and waits up to limit for it to return. A body
// that does not return is left running, detached, and false is returned: a
// run that never ends has to fail its test, not hold the suite. The body owns
// everything it touches (shared_ptr captures) and reports through state it
// owns; the caller reads that state only after a true.
template <class Body>
bool finishes_within(std::chrono::milliseconds limit, Body body) {
    auto done = std::make_shared<std::promise<void>>();
    std::future<void> finished = done->get_future();
    std::thread([done, work = std::move(body)]() mutable {
        work();
        done->set_value();
    }).detach();
    return finished.wait_for(limit) == std::future_status::ready;
}

// The same, stopping on the radius instead of the cap.
inline flop::bobyqa::Options options_xtol(double initial_step, double xtol_rel,
                                          std::size_t max_evaluations) {
    flop::bobyqa::Options o = options(initial_step, max_evaluations);
    o.stopping.xtol_rel = xtol_rel;
    return o;
}

struct Counting {
    std::size_t calls = 0;
    double operator()(std::span<const double> x) {
        ++calls;
        return v0111::sphere(x);
    }
};

// A facade configured with the BOBYQA settings of o.
inline flop::Minimizer facade(const flop::bobyqa::Options& o) {
    flop::Minimizer m = flop::Minimizer::create("BOBYQA");
    m.set_final_trust_radius(o.final_trust_radius)
        .set_interpolation_points(o.interpolation_points)
        .set_shrink_initial_step(o.shrink_initial_step)
        .set_newuoa_weight(o.newuoa_weight);
    return m;
}

// Calls every entry point that takes an objective, the two templates and the
// two facade overloads, with the same x0 and options, expects each to throw
// std::invalid_argument, and expects no evaluation from any of them.
inline void expect_rejected_everywhere(const Vec& x0, const flop::bobyqa::Options& o,
                                       const char* what) {
    Counting f;
    auto fb = [&f](std::span<const std::span<const double>> xs, std::span<double> out) {
        for (std::size_t i = 0; i < xs.size(); ++i) out[i] = f(xs[i]);
    };
    EXPECT_THROW((void)flop::bobyqa::minimize(f, x0, o), std::invalid_argument) << what;
    EXPECT_THROW((void)flop::bobyqa::minimize_batch(fb, x0, o), std::invalid_argument) << what;
    const flop::Minimizer m = facade(o);
    const flop::Minimizer::Objective fo = [&f](std::span<const double> x) { return f(x); };
    const flop::Minimizer::BatchObjective fbo = fb;
    const flop::Options& shared = o;
    EXPECT_THROW((void)m.minimize(fo, x0, shared), std::invalid_argument) << what;
    EXPECT_THROW((void)m.minimize(fbo, x0, shared), std::invalid_argument) << what;
    EXPECT_EQ(f.calls, 0u) << what;
}

}  // namespace v0121
