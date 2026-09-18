// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// Options - the settings every algorithm shares
// =============================================================================

#pragma once

#include <functional>
#include <optional>

#include "flop/bounds.hpp"
#include "flop/result.hpp"
#include "flop/stopping.hpp"

namespace flop {

struct Options {
    Stopping stopping;
    std::optional<Bounds> bounds;

    // The initial trust-region radius, Powell's rhobeg: the size of the first
    // step along each coordinate. 1.0 fits a problem scaled around unity, which
    // a problem in radians is. There is no rule deriving it from x0; a caller
    // whose coordinates differ in scale rescales the problem.
    double initial_step = 1.0;

    // Called once per objective evaluation, in evaluation order, with the
    // point, the value and the constraint values. Empty = off. A std::function
    // on purpose: one call per evaluation costs nothing next to an evaluation,
    // and this is what a trace, a history and a fidelity check hang off.
    std::function<void(const Evaluation&)> on_evaluation;
};

}  // namespace flop
