# Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Warnings, optimisation and the two floating-point models, in one place.
#
# FLOP_WARNING_FLAGS    applied to every target FLOP builds
# FLOP_STRICT_FP_FLAGS  the strict model: IEEE semantics, no reassociation
# FLOP_FAST_FP_FLAGS    the fast model: -ffast-math and everything it implies
#
# The library's own sources are compiled under the model FLOP_FAST_MATH
# selects. The test suite is built twice, once under each model, and both
# variants must pass: every guard in FLOP is written to survive -ffast-math,
# and a guard that only survives strict mode is not a guard.
#
# -march is off by default because a binary built with -march=native does
# not run on another machine; FLOP_MARCH_NATIVE turns it on for a local
# build that wants the instruction set.

if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    set(FLOP_WARNING_FLAGS -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow)
    set(FLOP_STRICT_FP_FLAGS -fno-fast-math)
    set(FLOP_FAST_FP_FLAGS -ffast-math)
elseif(MSVC)
    set(FLOP_WARNING_FLAGS /W4)
    set(FLOP_STRICT_FP_FLAGS /fp:strict)
    set(FLOP_FAST_FP_FLAGS /fp:fast)
else()
    set(FLOP_WARNING_FLAGS "")
    set(FLOP_STRICT_FP_FLAGS "")
    set(FLOP_FAST_FP_FLAGS "")
    message(WARNING "Unrecognised compiler '${CMAKE_CXX_COMPILER_ID}': the strict and fast-math "
                    "variants will be built identically.")
endif()

if(FLOP_MARCH_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" FLOP_HAS_MARCH_NATIVE)
    if(FLOP_HAS_MARCH_NATIVE)
        add_compile_options(-march=native)
    else()
        message(STATUS "FLOP: -march=native unsupported here; using the compiler default")
    endif()
endif()
