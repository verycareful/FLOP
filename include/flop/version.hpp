// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// version - the label a binary was built with
// =============================================================================

#pragma once

#include <string_view>

namespace flop {

// The four-component label from CMakeLists.txt, compiled into the library so
// a consumer can ask the binary rather than trust the headers it found.
std::string_view version() noexcept;

}  // namespace flop
