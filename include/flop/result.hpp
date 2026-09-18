// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// Result and Evaluation - what a run returns, and what the trace sees
// =============================================================================

#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "flop/status.hpp"

namespace flop {

// One objective evaluation, handed to Options::on_evaluation as it happens.
// The spans point into the optimizer's working storage and are valid only for
// the duration of the call; copy what has to outlive it.
struct Evaluation {
    std::size_t index;  // 0 for the first evaluation
    std::span<const double> x;
    double f;
    std::span<const double> constraints;  // empty when the problem has none
};

struct Result {
    std::vector<double> x;        // the best point seen, feasible when one was found
    double f = 0.0;               // the objective there
    std::size_t evaluations = 0;  // objective calls made, batch points counted singly
    Status status = Status::RoundoffLimited;
    double final_trust_radius = 0.0;        // rho at exit; the paper's user would ask for it
    double max_constraint_violation = 0.0;  // max(0, -min_i c_i(x)); 0 without constraints
};

}  // namespace flop
