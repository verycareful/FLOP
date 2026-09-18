// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The suite's entry point. Prints the library's version label after the
// run, so a log from a stale binary is recognisable as such.

#include <gtest/gtest.h>

#include <iostream>

#include "flop/version.hpp"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    std::cout << "FLOP " << flop::version() << '\n';
    return rc;
}
