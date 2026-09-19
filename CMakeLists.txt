# Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.25)

# The version here is the single source of truth. The label below, the
# README badge and CITATION.cff carry the same string; nothing else does.
project(FLOP VERSION 0.1.0.4 DESCRIPTION "Fast Library of Optimization Procedures" LANGUAGES CXX)
set(FLOP_VERSION_LABEL "0.1.0.4")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(FLOP_IS_TOP_LEVEL ON)
else()
    set(FLOP_IS_TOP_LEVEL OFF)
endif()

option(FLOP_BUILD_TESTS "Build the test suite (both floating-point variants)" ${FLOP_IS_TOP_LEVEL})
option(FLOP_BUILD_BENCHMARKS "Build the Google Benchmark targets" OFF)
option(FLOP_FAST_MATH "Compile the library's own sources with -ffast-math" OFF)
option(FLOP_MARCH_NATIVE "Compile with -march=native (non-distributable)" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FlopCompilerFlags)

# ===========================================================================
# The library: header-mostly, plus the facade and the version label compiled
# once so that every consumer of them runs the same code.
# ===========================================================================
add_library(flop STATIC
    src/minimizer.cpp
    src/version.cpp
)
add_library(flop::flop ALIAS flop)

target_include_directories(flop PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(flop PUBLIC cxx_std_23)
target_compile_definitions(flop PRIVATE FLOP_VERSION_LABEL="${FLOP_VERSION_LABEL}")
target_compile_options(flop PRIVATE ${FLOP_WARNING_FLAGS})
if(FLOP_FAST_MATH)
    target_compile_options(flop PRIVATE ${FLOP_FAST_FP_FLAGS})
else()
    target_compile_options(flop PRIVATE ${FLOP_STRICT_FP_FLAGS})
endif()

# ===========================================================================
# Install: headers, the archive, the licence beside them, and an export so a
# consumer can find_package(flop) or FetchContent it and link flop::flop.
# ===========================================================================
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS flop EXPORT flopTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/flop DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/flop)
install(EXPORT flopTargets NAMESPACE flop:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/flop)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/flopConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/flopConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/flopConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/flop
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/flopConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/flopConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/flop
)

# ===========================================================================
# Tests and benchmarks
# ===========================================================================
if(FLOP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(FLOP_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
