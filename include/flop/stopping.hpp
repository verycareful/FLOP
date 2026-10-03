// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// Stopping - when a run ends
// =============================================================================

#pragma once

#include <cstddef>
#include <optional>

namespace flop {

// At least one criterion must be set, or validation throws: a method with no
// stopping rule would run forever, and a default that hides that is a silent
// failure. Every tolerance is checked as written: xtol on the method's radius
// (COBYLA's trust radius, Nelder-Mead's simplex radius), ftol on the progress
// in f the method measures (COBYLA: the change across an accepted step;
// Nelder-Mead: the spread over the simplex), stop_value on f after every
// evaluation, max_evaluations before every evaluation.
struct Stopping {
    std::size_t max_evaluations = 0;  // 0 = no cap
    double xtol_rel = 0.0;  // relative to initial_step: stop at radius xtol_rel * initial_step
    double xtol_abs = 0.0;  // absolute floor on the radius
    double ftol_rel = 0.0;  // relative to |f|
    double ftol_abs = 0.0;
    std::optional<double> stop_value;  // stop as soon as f <= stop_value
};

}  // namespace flop
