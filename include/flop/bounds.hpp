// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// Bounds - box constraints on the variables
// =============================================================================

#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace flop {

// One optional bound per coordinate in each direction. "No bound" is an empty
// optional, never an infinity: FLOP carries no non-finite value anywhere, so
// nothing in it depends on how a compiler treats one.
struct Bounds {
    std::vector<std::optional<double>> lower;
    std::vector<std::optional<double>> upper;

    // Every coordinate bounded on both sides.
    static Bounds box(std::span<const double> lo, std::span<const double> hi) {
        Bounds b;
        b.lower.assign(lo.begin(), lo.end());
        b.upper.assign(hi.begin(), hi.end());
        return b;
    }

    // n coordinates, none bounded; set individual entries afterwards.
    static Bounds none(std::size_t n) {
        Bounds b;
        b.lower.assign(n, std::nullopt);
        b.upper.assign(n, std::nullopt);
        return b;
    }

    [[nodiscard]] std::size_t size() const noexcept { return lower.size(); }
};

}  // namespace flop
