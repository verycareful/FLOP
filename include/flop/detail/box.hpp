// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// box - the two box operations more than one algorithm needs
// =============================================================================
//
// A missing bound is an empty optional, so every test here asks has_value()
// first and nothing compares against an infinity.

#pragma once

#include <cstddef>
#include <optional>
#include <span>

#include "flop/bounds.hpp"

namespace flop::detail {

// The displacement of the initial vertex along coordinate i from x, for a
// requested step h > 0: +h when the box allows it, else -h, else whichever
// side has more room, shrunk to that room. Without bounds it is always +h.
// The result is never zero, because validation requires lower < upper
// wherever both bounds are present.
inline double axis_offset(const Bounds* bounds, std::span<const double> x, std::size_t i,
                          double h) noexcept {
    if (!bounds) return h;
    const std::optional<double>& hi = bounds->upper[i];
    const std::optional<double>& lo = bounds->lower[i];
    const double up = hi.has_value() ? *hi - x[i] : h;
    const double down = lo.has_value() ? x[i] - *lo : h;
    if (up >= h) return h;
    if (down >= h) return -h;
    return up >= down ? up : -down;
}

// Clamps every coordinate of x into the box: the Euclidean projection onto
// it, coordinate by coordinate. A coordinate at or inside its bounds keeps
// its bits.
inline void project_onto_box(const Bounds& bounds, std::span<double> x) noexcept {
    for (std::size_t i = 0; i < x.size(); ++i) {
        const std::optional<double>& lo = bounds.lower[i];
        const std::optional<double>& hi = bounds.upper[i];
        if (lo.has_value() && x[i] < *lo) x[i] = *lo;
        if (hi.has_value() && x[i] > *hi) x[i] = *hi;
    }
}

}  // namespace flop::detail
