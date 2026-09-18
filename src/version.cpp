// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "flop/version.hpp"

#ifndef FLOP_VERSION_LABEL
#error "FLOP_VERSION_LABEL is set by CMakeLists.txt; build through CMake"
#endif

namespace flop {

std::string_view version() noexcept {
    return FLOP_VERSION_LABEL;
}

}  // namespace flop
