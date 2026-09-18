# Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# The suite is one set of sources built twice: flop_tests_strict under the
# strict floating-point model and flop_tests_fast under -ffast-math. Both
# register with CTest and both must pass. The template entry points are
# instantiated inside each test binary under that binary's model, so the two
# variants exercise the same algorithm under both models; the facade, linked
# from the library, runs under the model the library was built with in both.

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Test sources, one file per area, named test_v<version>_<area>.cpp. The
# list is empty until the first test release; main.cpp alone builds a binary
# that runs zero tests and passes.
set(FLOP_TEST_SOURCES
)

function(flop_add_test_variant name fp_flags)
    add_executable(${name} main.cpp ${FLOP_TEST_SOURCES})
    target_link_libraries(${name} PRIVATE flop::flop GTest::gtest)
    target_compile_options(${name} PRIVATE ${FLOP_WARNING_FLAGS} ${fp_flags})
    add_test(NAME ${name} COMMAND ${name})
endfunction()

flop_add_test_variant(flop_tests_strict "${FLOP_STRICT_FP_FLAGS}")
flop_add_test_variant(flop_tests_fast "${FLOP_FAST_FP_FLAGS}")
