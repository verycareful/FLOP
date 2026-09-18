// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// Status - why a run stopped
// =============================================================================

#pragma once

#include <string_view>

namespace flop {

// Every run ends with exactly one of these. Reaching the evaluation cap is an
// outcome like any other and is reported as itself: a caller who wants to know
// whether the method finished asks converged(), and the cap answers false.
enum class Status {
    XtolReached,            // the trust region shrank below the x tolerance
    FtolReached,            // an accepted step changed f by less than the f tolerance
    StopValueReached,       // f fell to or below stop_value
    MaxEvaluationsReached,  // the evaluation cap; the best point so far is returned
    RoundoffLimited         // the model could not be improved at this precision
};

constexpr bool converged(Status s) noexcept {
    return s == Status::XtolReached || s == Status::FtolReached || s == Status::StopValueReached;
}

constexpr std::string_view to_string(Status s) noexcept {
    switch (s) {
        case Status::XtolReached:
            return "XtolReached";
        case Status::FtolReached:
            return "FtolReached";
        case Status::StopValueReached:
            return "StopValueReached";
        case Status::MaxEvaluationsReached:
            return "MaxEvaluationsReached";
        case Status::RoundoffLimited:
            return "RoundoffLimited";
    }
    return "Status(?)";
}

}  // namespace flop
