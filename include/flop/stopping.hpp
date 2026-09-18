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
// failure. Every tolerance is checked as written: xtol on the trust region
// radius, ftol on the change in f across an accepted step, stop_value on f
// after every evaluation, max_evaluations before every evaluation.
struct Stopping {
    std::size_t max_evaluations = 0;  // 0 = no cap
    double xtol_rel = 0.0;  // relative to initial_step: rho stops at xtol_rel * initial_step
    double xtol_abs = 0.0;  // absolute floor on rho
    double ftol_rel = 0.0;  // relative to |f|
    double ftol_abs = 0.0;
    std::optional<double> stop_value;  // stop as soon as f <= stop_value
};

}  // namespace flop
