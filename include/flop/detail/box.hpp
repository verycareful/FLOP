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

// Coordinate i of the initial vertex displaced from x, for a requested step
// h > 0: x[i] + h when that point is inside the box, else x[i] - h when that
// one is, else the bound on the side with more room, itself. Without bounds
// it is always x[i] + h. The vertex is a point, not an offset added to x[i]:
// x[i] + (bound - x[i]) can round past the bound (x[i] = -5, bound 0.2 gives
// 0.20000000000000018), and no offset can avoid that, since x[i] + offset
// lands on a grid that need not hold the bound.
//
// Requires |x[i]| + h <= DBL_MAX, so that neither trial point overflows;
// validation guarantees it for x0, and a restart checks it before calling.
// When neither trial point fits, both rooms are below h, so neither room
// overflows either. The result differs from x[i], because validation
// requires lower < upper wherever both bounds are present.
inline double axis_vertex(const Bounds* bounds, std::span<const double> x, std::size_t i,
                          double h) noexcept {
    const double up = x[i] + h;
    if (!bounds) return up;
    const std::optional<double>& hi = bounds->upper[i];
    const std::optional<double>& lo = bounds->lower[i];
    if (!hi.has_value() || up <= *hi) return up;
    const double down = x[i] - h;
    if (!lo.has_value() || down >= *lo) return down;
    return *hi - x[i] >= x[i] - *lo ? *hi : *lo;
}

// Clamps every coordinate of x into the box: the Euclidean projection onto
// it, coordinate by coordinate. A coordinate at or inside its bounds keeps
// its bits. True when some coordinate was outside and moved.
inline bool project_onto_box(const Bounds& bounds, std::span<double> x) noexcept {
    bool moved = false;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const std::optional<double>& lo = bounds.lower[i];
        const std::optional<double>& hi = bounds.upper[i];
        if (lo.has_value() && x[i] < *lo) {
            x[i] = *lo;
            moved = true;
        }
        if (hi.has_value() && x[i] > *hi) {
            x[i] = *hi;
            moved = true;
        }
    }
    return moved;
}

}  // namespace flop::detail
