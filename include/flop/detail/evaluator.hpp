// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// evaluator - the scalar and the batch objective behind one interface
// =============================================================================
//
// A solver is a template on one of these two, so it asks for one point or
// for several the same way whatever the caller supplied. has_batch says
// whether the caller's objective takes several points in one call; a solver
// sends a batch only where it has independent points and has_batch is true,
// and otherwise loops. A scalar objective asked for a batch evaluates the
// points one at a time, in order, so the evaluation order is the same on
// both channels.

#pragma once

#include <cstddef>
#include <span>

#include "flop/concepts.hpp"

namespace flop::detail {

template <ScalarObjective F>
struct ScalarEvaluator {
    F& f;
    static constexpr bool has_batch = false;
    double operator()(std::span<const double> x) { return static_cast<double>(f(x)); }
    void batch(std::span<const std::span<const double>> xs, std::span<double> out) {
        for (std::size_t i = 0; i < xs.size(); ++i) out[i] = static_cast<double>(f(xs[i]));
    }
};

template <BatchObjective F>
struct BatchEvaluator {
    F& f;
    static constexpr bool has_batch = true;
    double operator()(std::span<const double> x) {
        double out = 0.0;
        std::span<const double> one[1] = {x};
        f(std::span<const std::span<const double>>(one, 1), std::span<double>(&out, 1));
        return out;
    }
    void batch(std::span<const std::span<const double>> xs, std::span<double> out) { f(xs, out); }
};

}  // namespace flop::detail
