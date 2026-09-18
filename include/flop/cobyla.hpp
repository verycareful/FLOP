// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// cobyla - Constrained Optimization BY Linear Approximations (Powell 1994)
// =============================================================================
//
// Derivative-free minimisation of f(x) subject to c_i(x) >= 0 and box bounds,
// by linear interpolation on a simplex inside a trust region. Four entry
// points, one algorithm: with or without constraints, with a scalar or a
// batch objective. The batch objective is used for the initial simplex, whose
// n + 1 points are independent; every later evaluation is one point.
//
// Every entry point validates its input and throws std::invalid_argument on a
// malformed problem; after that nothing throws for input. An objective or
// constraint that returns a non-finite value ends the run with
// std::runtime_error, where the compiler lets the value reach memory.
//
// Example:
//
//     auto rosenbrock = [](std::span<const double> x) {
//         const double a = 1.0 - x[0], b = x[1] - x[0] * x[0];
//         return a * a + 100.0 * b * b;
//     };
//     flop::cobyla::Options opts;
//     opts.stopping.xtol_rel = 1e-8;
//     opts.stopping.max_evaluations = 2000;
//     opts.initial_step = 0.5;
//     const double x0[2] = {-1.2, 1.0};
//     flop::Result r = flop::cobyla::minimize(rosenbrock, x0, opts);
//     // r.x near (1, 1), flop::converged(r.status) true

#pragma once

#include <cstddef>
#include <span>

#include "flop/concepts.hpp"
#include "flop/detail/cobyla_impl.hpp"
#include "flop/detail/validate.hpp"
#include "flop/result.hpp"

namespace flop::cobyla {

// Unconstrained (box bounds through opts.bounds), scalar objective.
template <ScalarObjective F>
[[nodiscard]] Result minimize(F&& f, std::span<const double> x0, const Options& opts) {
    detail::validate_options("flop::cobyla::minimize", opts, x0);
    detail::cobyla::NoConstraints none;
    detail::cobyla::Solver<detail::cobyla::ScalarEvaluator<std::remove_reference_t<F>>,
                           detail::cobyla::NoConstraints>
        solver({f}, none, 0, x0, opts);
    return solver.run();
}

// With n_constraints inequality constraints c(x, out), feasible when every
// out[i] >= 0.
template <ScalarObjective F, ConstraintFunction C>
[[nodiscard]] Result minimize(F&& f, C&& c, std::size_t n_constraints, std::span<const double> x0,
                              const Options& opts) {
    detail::validate_options("flop::cobyla::minimize", opts, x0);
    detail::cobyla::Solver<detail::cobyla::ScalarEvaluator<std::remove_reference_t<F>>,
                           std::remove_reference_t<C>>
        solver({f}, c, n_constraints, x0, opts);
    return solver.run();
}

// Batch objective f(xs, out): the initial simplex goes out as one call of
// n + 1 points when the evaluation cap allows it.
template <BatchObjective F>
[[nodiscard]] Result minimize_batch(F&& f, std::span<const double> x0, const Options& opts) {
    detail::validate_options("flop::cobyla::minimize_batch", opts, x0);
    detail::cobyla::NoConstraints none;
    detail::cobyla::Solver<detail::cobyla::BatchEvaluator<std::remove_reference_t<F>>,
                           detail::cobyla::NoConstraints>
        solver({f}, none, 0, x0, opts);
    return solver.run();
}

template <BatchObjective F, ConstraintFunction C>
[[nodiscard]] Result minimize_batch(F&& f, C&& c, std::size_t n_constraints,
                                    std::span<const double> x0, const Options& opts) {
    detail::validate_options("flop::cobyla::minimize_batch", opts, x0);
    detail::cobyla::Solver<detail::cobyla::BatchEvaluator<std::remove_reference_t<F>>,
                           std::remove_reference_t<C>>
        solver({f}, c, n_constraints, x0, opts);
    return solver.run();
}

}  // namespace flop::cobyla
