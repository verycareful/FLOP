// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// nelder_mead - the Nelder-Mead simplex method (Nelder and Mead 1965, as
//               stated by Lagarias, Reeds, Wright and Wright 1998)
// =============================================================================
//
// Derivative-free minimisation of f(x), optionally inside box bounds, by a
// simplex of n + 1 points that reflects, expands, contracts and shrinks. Two
// entry points, one algorithm: a scalar or a batch objective. The batch
// objective is used where the method has independent points: the initial
// simplex (n + 1 points), a shrink (the vertices it moved), a rung of the
// poll ladder on a box (up to 2n points) and a restart's simplex (n points);
// every other evaluation is one point. The method has no nonlinear
// constraints, and there is no entry point that takes them.
//
// Every entry point validates its input and throws std::invalid_argument on a
// malformed problem; after that nothing throws for input. An objective that
// returns a non-finite value ends the run with std::runtime_error, where the
// compiler lets the value reach memory, and so does a point the method would
// evaluate beyond DBL_MAX / (n + 5), the range in which its arithmetic stays
// finite (an objective unbounded below).
//
// Example:
//
//     auto rosenbrock = [](std::span<const double> x) {
//         const double a = 1.0 - x[0], b = x[1] - x[0] * x[0];
//         return a * a + 100.0 * b * b;
//     };
//     flop::nelder_mead::Options opts;
//     opts.stopping.xtol_rel = 1e-8;
//     opts.stopping.max_evaluations = 2000;
//     opts.initial_step = 0.5;
//     const double x0[2] = {-1.2, 1.0};
//     flop::Result r = flop::nelder_mead::minimize(rosenbrock, x0, opts);
//     // r.x near (1, 1), flop::converged(r.status) true

#pragma once

#include <span>
#include <type_traits>

#include "flop/concepts.hpp"
#include "flop/detail/evaluator.hpp"
#include "flop/detail/nelder_mead_impl.hpp"
#include "flop/detail/validate.hpp"
#include "flop/result.hpp"

namespace flop::nelder_mead {

// Scalar objective; box bounds through opts.bounds.
template <ScalarObjective F>
[[nodiscard]] Result minimize(F&& f, std::span<const double> x0, const Options& opts) {
    detail::validate_options("flop::nelder_mead::minimize", opts, x0);
    detail::validate_nelder_mead_range("flop::nelder_mead::minimize", opts, x0);
    detail::nelder_mead::Solver<detail::ScalarEvaluator<std::remove_reference_t<F>>> solver({f}, x0,
                                                                                            opts);
    return solver.run();
}

// Batch objective f(xs, out): the initial simplex goes out as one call of
// n + 1 points, every shrink as one call of the vertices it moved, every
// rung of the ladder as one call of up to 2n points and every restart's
// simplex as one call of n, each when the evaluation cap allows all of them.
template <BatchObjective F>
[[nodiscard]] Result minimize_batch(F&& f, std::span<const double> x0, const Options& opts) {
    detail::validate_options("flop::nelder_mead::minimize_batch", opts, x0);
    detail::validate_nelder_mead_range("flop::nelder_mead::minimize_batch", opts, x0);
    detail::nelder_mead::Solver<detail::BatchEvaluator<std::remove_reference_t<F>>> solver({f}, x0,
                                                                                           opts);
    return solver.run();
}

}  // namespace flop::nelder_mead
