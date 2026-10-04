// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// tolerance - f tolerance tests that never leave the finite range
// =============================================================================
//
// Both algorithms test a difference of two objective values, a - b, against
// ftol_abs or against ftol_rel * |b|: Nelder-Mead's spread f_{n+1} - f_1,
// COBYLA's fall f_before - f_after. Objective values may be anywhere in the
// finite range, and a - b of two finite values can exceed it (DBL_MAX minus
// -DBL_MAX), as can ftol_rel * |b| when ftol_rel > 1. Forming either is an
// infinity, undefined under -ffinite-math-only. These tests decide the same
// inequalities without forming any value outside the range.
//
// Where nothing can overflow and ftol_rel <= 1, each test is the plain one,
// a - b <= t or a - b <= r |b|, rounded exactly as written, so a run that
// never comes near the ends of the range takes the branches the plain test
// takes. ftol_rel > 1 is decided by a division instead of a product.
//
// Under -ffast-math the compiler may regroup any arithmetic expression, so
// splitting a - b into pieces that each stay finite (0.5 a - 0.5 b) is no
// protection: it is free to factor them back into 0.5 (a - b). Where the
// difference can leave the range, nothing here computes an expression in
// both a and b; each side of the comparison is formed from one of them.

#pragma once

#include <cmath>
#include <limits>

namespace flop::detail {

inline constexpr double kMaxFinite = std::numeric_limits<double>::max();

// The exact a - b is above the largest double. That needs b < 0 < a, and
// then max + b is computed without overflow.
inline bool difference_above_range(const double& a, const double& b) noexcept {
    return b < 0.0 && a > kMaxFinite + b;
}

// The exact a - b is below minus the largest double: a < 0 < b.
inline bool difference_below_range(const double& a, const double& b) noexcept {
    return b > 0.0 && a < b - kMaxFinite;
}

// a - b <= t, for t >= 0 finite. A difference above the range exceeds every
// finite t; one below it is under every t >= 0.
inline bool difference_within_abs(const double& a, const double& b, double t) noexcept {
    if (difference_above_range(a, b)) return false;
    if (difference_below_range(a, b)) return true;
    return a - b <= t;
}

// a - b <= r |b|, for r > 0 finite. With r <= 1 the right side is at most
// |b| and cannot overflow, and a difference above the range exceeds it.
// Above the range b < 0 < a, so the inequality is a + |b| <= r |b|, that is
// a <= (r - 1) |b|: with c = r - 1, a / c <= |b| when c >= 1 and
// a <= c |b| when c < 1, neither of which can overflow. In range with r > 1,
// both sides are divided by r. A difference below the range is under the
// nonnegative right side.
inline bool difference_within_rel(const double& a, const double& b, double r) noexcept {
    const double mag = std::fabs(b);
    if (difference_above_range(a, b)) {
        if (r <= 1.0) return false;
        const double c = r - 1.0;
        if (c >= 1.0) return a / c <= mag;
        return a <= c * mag;
    }
    if (difference_below_range(a, b)) return true;
    const double d = a - b;
    if (r <= 1.0) return d <= r * mag;
    return d / r <= mag;
}

}  // namespace flop::detail
