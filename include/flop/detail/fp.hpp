// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// fp - non-finite detection that survives -ffast-math
// =============================================================================
//
// Every guard in FLOP that asks "is this value finite" comes here. Under
// -ffast-math a compiler may assume no infinity or NaN exists, so
// std::isfinite folds to a constant and the guard becomes dead code. These
// helpers read the IEEE-754 exponent field of the object representation
// instead: integer work on the bits, no floating-point predicate, nothing a
// range assumption can remove.
//
// Two rules keep that true, and both are visible in the signatures:
//
//   1. Values are taken by reference, never by value. Under -ffinite-math-only
//      Clang marks by-value floating-point parameters and return values as
//      finite, and from Clang 22 on it folds an exponent-field test on such a
//      value to "finite" even after inlining. A reference is a pointer; a load
//      through it carries no such mark.
//
//   2. Nothing here produces a NaN or an infinity. FLOP never uses either as a
//      sentinel: "no best point yet" is a bool, "no bound" is an empty
//      std::optional, and a value the objective returned is checked the moment
//      it lands in memory.
//
// The contract is therefore: fp_bad reports whether the object representation
// in memory is a NaN or an infinity. In a translation unit compiled with
// -ffinite-math-only the caller has promised the compiler that no such value
// is ever computed, and FLOP can only check what reaches memory.

#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

namespace flop::detail {

// True if x is NaN or an infinity: the binary64 exponent field is all ones for
// those and nothing else.
constexpr bool fp_bad(const double& x) noexcept {
    static_assert(sizeof(double) == sizeof(std::uint64_t), "fp_bad assumes IEEE-754 binary64");
    const auto bits = std::bit_cast<std::uint64_t>(x);
    return ((bits >> 52) & UINT64_C(0x7FF)) == UINT64_C(0x7FF);
}

// True if any element of v is non-finite.
constexpr bool any_bad(std::span<const double> v) noexcept {
    for (const double& x : v) {
        if (fp_bad(x)) return true;
    }
    return false;
}

}  // namespace flop::detail
