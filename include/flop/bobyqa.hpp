// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// bobyqa - Powell's BOBYQA (DAMTP 2009/NA06), bound optimisation by quadratic
//          approximation
// =============================================================================
//
// Derivative-free minimisation of f(x), optionally inside box bounds, by a
// quadratic model that interpolates f at m points (2n + 1 by default) and is
// updated by least Frobenius norm changes, with a trust region whose lower
// bound rho falls from initial_step to the final radius. Two entry points,
// one algorithm: a scalar or a batch objective. The batch objective is used
// where the method has independent points: the starting points (2n + 1 of
// them, then the m - 2n - 1 that depend on their values) and the new points
// of a RESCUE; every other evaluation is one point. The method has no
// nonlinear constraints, and there is no entry point that takes them.
//
// Every coordinate bounded on both sides needs b_i - a_i >= 2 initial_step
// (the report's (2.1)); Options::shrink_initial_step lowers the step to fit
// instead. Every entry point validates its input and throws
// std::invalid_argument on a malformed problem; after that nothing throws
// for input. An objective that returns a non-finite value ends the run with
// std::runtime_error, where the compiler lets the value reach memory, and so
// does a point beyond the largest double (an objective unbounded below).
//
// Example:
//
//     auto rosenbrock = [](std::span<const double> x) {
//         const double a = 1.0 - x[0], b = x[1] - x[0] * x[0];
//         return a * a + 100.0 * b * b;
//     };
//     flop::bobyqa::Options opts;
//     opts.stopping.xtol_abs = 1e-8;
//     opts.stopping.max_evaluations = 2000;
//     opts.initial_step = 0.5;
//     const double x0[2] = {-1.2, 1.0};
//     flop::Result r = flop::bobyqa::minimize(rosenbrock, x0, opts);

#pragma once

#include <span>
#include <type_traits>

#include "flop/concepts.hpp"
#include "flop/detail/bobyqa_impl.hpp"
#include "flop/detail/evaluator.hpp"
#include "flop/detail/validate.hpp"
#include "flop/result.hpp"

namespace flop::bobyqa {

template <ScalarObjective F>
[[nodiscard]] Result minimize(F&& f, std::span<const double> x0, const Options& opts) {
    detail::validate_options("flop::bobyqa::minimize", opts, x0);
    detail::validate_bobyqa("flop::bobyqa::minimize", opts, x0);
    detail::bobyqa::Solver<detail::ScalarEvaluator<std::remove_reference_t<F>>> solver({f}, x0,
                                                                                       opts);
    return solver.run();
}

template <BatchObjective F>
[[nodiscard]] Result minimize_batch(F&& f, std::span<const double> x0, const Options& opts) {
    detail::validate_options("flop::bobyqa::minimize_batch", opts, x0);
    detail::validate_bobyqa("flop::bobyqa::minimize_batch", opts, x0);
    detail::bobyqa::Solver<detail::BatchEvaluator<std::remove_reference_t<F>>> solver({f}, x0,
                                                                                      opts);
    return solver.run();
}

}  // namespace flop::bobyqa
