// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// concepts - what an objective, a batch objective and a constraint look like
// =============================================================================
//
// FLOP's algorithms are class templates constrained by these concepts, so an
// optimizer that needs a capability the objective lacks fails to compile
// rather than at run time. Points go in as std::span<const double>: no
// allocation per evaluation, and the objective cannot mistake the vector for
// something it owns.

#pragma once

#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>

namespace flop {

// f(x) -> double. The objective to minimise.
template <class F>
concept ScalarObjective =
    std::invocable<F&, std::span<const double>> &&
    std::convertible_to<std::invoke_result_t<F&, std::span<const double>>, double>;

// f(xs, out): evaluates every point in xs and writes out[i] = f(xs[i]), in
// order. xs.size() == out.size() always. The caller decides how the points are
// evaluated (serially, with OpenMP, on a device); FLOP never spawns a thread.
// An algorithm uses this channel wherever it has independent points to
// evaluate, COBYLA for its initial simplex, and the scalar channel elsewhere.
template <class F>
concept BatchObjective =
    requires(F& f, std::span<const std::span<const double>> xs, std::span<double> out) {
        { f(xs, out) };
    };

// c(x, out): writes out[i] = c_i(x) for every constraint. A point is feasible
// when every c_i(x) >= 0, Powell's sign convention. out.size() is the
// constraint count the caller declared.
template <class C>
concept ConstraintFunction = requires(C& c, std::span<const double> x, std::span<double> out) {
    { c(x, out) };
};

}  // namespace flop
