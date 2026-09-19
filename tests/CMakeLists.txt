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

# Test sources, one file per area, named test_v<version>_<area>.cpp after
# the release that ships them; v0101_problems.hpp holds the problems and
# optima the 0.1.0.1 files share. All of them compile under both models.
#
#   validation     every std::invalid_argument, and the mid-run runtime_error
#   unconstrained  standard problems with closed-form optima
#   constrained    Powell's 1994 test set, HS24, HS35
#   bounds         every evaluation inside the box; inactive box is a no-op
#   corpus         replay of recorded NLopt runs from tests/data/nlopt-corpus
#   batch          the batch channel's call shape and optimum
#   status         each Status on a problem built for it; the trace
#   facade         Minimizer against the template; the version label
#   fp             the bit-pattern finiteness check; determinism
#   step           the constrained trust-region subproblem against closed forms
#   violation      the reported violation never carries a negative zero
#   paper          Powell's Tables 1 and 2: problems (A) to (J) at his settings
#   growth         the trust radius that grows, against the paper's that does not
set(FLOP_TEST_SOURCES
    test_v0101_validation.cpp
    test_v0101_unconstrained.cpp
    test_v0101_constrained.cpp
    test_v0101_bounds.cpp
    test_v0101_corpus.cpp
    test_v0101_batch.cpp
    test_v0101_status.cpp
    test_v0101_facade.cpp
    test_v0101_fp.cpp
    test_v0102_step.cpp
    test_v0102_violation.cpp
    test_v0102_paper.cpp
    test_v0102_growth.cpp
)

# extra_defs: definitions specific to one variant. The fast variant defines
# FLOP_TEST_FAST_MATH so a test knows its template instantiations run under
# -ffast-math; every variant learns the library's own model through
# FLOP_TEST_LIBRARY_FAST_MATH, set when FLOP_FAST_MATH is on, so a test can
# tell whether the facade it links shares its model.
function(flop_add_test_variant name fp_flags extra_defs)
    add_executable(${name} main.cpp ${FLOP_TEST_SOURCES})
    target_link_libraries(${name} PRIVATE flop::flop GTest::gtest)
    target_compile_options(${name} PRIVATE ${FLOP_WARNING_FLAGS} ${fp_flags})
    target_compile_definitions(${name} PRIVATE
        FLOP_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data" ${extra_defs})
    if(FLOP_FAST_MATH)
        target_compile_definitions(${name} PRIVATE FLOP_TEST_LIBRARY_FAST_MATH)
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()

flop_add_test_variant(flop_tests_strict "${FLOP_STRICT_FP_FLAGS}" "")
flop_add_test_variant(flop_tests_fast "${FLOP_FAST_FP_FLAGS}" "FLOP_TEST_FAST_MATH")
