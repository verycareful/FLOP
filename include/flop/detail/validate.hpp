// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// validate - every std::invalid_argument an entry point can throw, in one place
// =============================================================================
//
// Validation runs once, at entry, before any evaluation. Everything past it
// may assume a well-formed problem, and nothing past it throws for input.

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

#include "flop/detail/box.hpp"
#include "flop/detail/fp.hpp"
#include "flop/options.hpp"

namespace flop::detail {

inline void fail(const char* where, const std::string& what) {
    throw std::invalid_argument(std::string(where) + ": " + what);
}

inline void validate_x0(const char* where, std::span<const double> x0) {
    if (x0.empty()) fail(where, "x0 is empty");
    if (any_bad(x0)) fail(where, "x0 contains a non-finite value");
}

inline void validate_stopping(const char* where, const Stopping& s) {
    if (s.xtol_rel < 0.0 || s.xtol_abs < 0.0 || s.ftol_rel < 0.0 || s.ftol_abs < 0.0)
        fail(where, "a tolerance is negative");
    if (fp_bad(s.xtol_rel) || fp_bad(s.xtol_abs) || fp_bad(s.ftol_rel) || fp_bad(s.ftol_abs))
        fail(where, "a tolerance is non-finite");
    if (s.stop_value && fp_bad(*s.stop_value)) fail(where, "stop_value is non-finite");
    const bool any = s.max_evaluations > 0 || s.xtol_rel > 0.0 || s.xtol_abs > 0.0 ||
                     s.ftol_rel > 0.0 || s.ftol_abs > 0.0 || s.stop_value.has_value();
    if (!any)
        fail(where,
             "no stopping criterion is set (max_evaluations, xtol_rel, xtol_abs, ftol_rel, "
             "ftol_abs or stop_value)");
}

inline void validate_bounds(const char* where, const Bounds& b, std::span<const double> x0) {
    if (b.lower.size() != x0.size() || b.upper.size() != x0.size())
        fail(where, "bounds have " + std::to_string(b.lower.size()) + " lower and " +
                        std::to_string(b.upper.size()) + " upper entries for " +
                        std::to_string(x0.size()) + " variables");
    for (std::size_t i = 0; i < x0.size(); ++i) {
        const std::optional<double>& lo = b.lower[i];
        const std::optional<double>& hi = b.upper[i];
        if (lo.has_value() && fp_bad(*lo)) fail(where, "a lower bound is non-finite");
        if (hi.has_value() && fp_bad(*hi)) fail(where, "an upper bound is non-finite");
        if (lo.has_value() && hi.has_value() && *lo >= *hi)
            fail(where, "lower bound is not below upper bound at coordinate " + std::to_string(i) +
                            " (a fixed coordinate is not a variable; drop it from x)");
        if (lo.has_value() && x0[i] < *lo)
            fail(where, "x0 is below the lower bound at coordinate " + std::to_string(i));
        if (hi.has_value() && x0[i] > *hi)
            fail(where, "x0 is above the upper bound at coordinate " + std::to_string(i));
    }
}

// The initial simplex computes x0[i] + initial_step and x0[i] - initial_step,
// box or no box (box.hpp), so both must stay finite: initial_step <= max -
// |x0[i]|, which is computed without overflow since |x0[i]| <= max.
inline void validate_step_range(const char* where, double h, std::span<const double> x0) {
    constexpr double kMax = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < x0.size(); ++i)
        if (h > kMax - std::fabs(x0[i]))
            fail(where,
                 "x0 plus or minus initial_step overflows the largest double at coordinate " +
                     std::to_string(i));
}

inline void validate_options(const char* where, const Options& o, std::span<const double> x0) {
    validate_x0(where, x0);
    validate_stopping(where, o.stopping);
    if (fp_bad(o.initial_step) || !(o.initial_step > 0.0))
        fail(where, "initial_step must be positive and finite");
    validate_step_range(where, o.initial_step, x0);
    if (o.bounds) validate_bounds(where, *o.bounds, x0);
}

// Nelder-Mead evaluates no point with a coordinate larger than this in
// magnitude; nelder_mead_impl.hpp (check_range) says why the factor is
// n + 5.
constexpr double nelder_mead_range_limit(std::size_t n) noexcept {
    return std::numeric_limits<double>::max() / static_cast<double>(n + 5);
}

// x0 and the initial vertices are input, so a coordinate of either beyond
// Nelder-Mead's range makes the problem invalid rather than the run fail.
// Runs after validate_options, which has checked the bounds and that
// x0 +- initial_step is finite.
inline void validate_nelder_mead_range(const char* where, const Options& o,
                                       std::span<const double> x0) {
    const double limit = nelder_mead_range_limit(x0.size());
    const Bounds* b = o.bounds ? &*o.bounds : nullptr;
    for (std::size_t i = 0; i < x0.size(); ++i) {
        if (std::fabs(x0[i]) > limit)
            fail(where, "x0 is beyond DBL_MAX / (n + 5) at coordinate " + std::to_string(i) +
                            ", the range in which the method's arithmetic stays finite");
        if (std::fabs(axis_vertex(b, x0, i, o.initial_step)) > limit)
            fail(where, "the initial vertex along coordinate " + std::to_string(i) +
                            " is beyond DBL_MAX / (n + 5), the range in which the method's "
                            "arithmetic stays finite; initial_step is too large for x0");
    }
}

}  // namespace flop::detail
