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
# optima the 0.1.0.1 files share, v0111_problems.hpp and the reference
# transcription v0111_reference_nm.hpp what the Nelder-Mead files share.
# All of them compile under both models.
#
# COBYLA:
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
#
# Nelder-Mead:
#   paper          Lagarias et al.'s rules in exact arithmetic; what the papers prove
#   coefficients   Gao and Han's set against (4.1); their Table 1 and Figs. 1-2
#   reference      the method against the reference transcription
#   bookkeeping    the running sum, the extents, the stop from the diameter bracket
#   status         each stopping rule, the cap, the precision floor, the result
#   bounds         projection, the initial simplex in a box, the poll ladder
#   batch          the batch channel's call shape against the scalar trajectory
#   validation     every std::invalid_argument, and the run-time errors
#   facade         NELDER_MEAD through Minimizer; header compiles alone, so it
#                  includes nothing of FLOP but flop/minimizer.hpp
#   fp             determinism, extreme values, no non-finite point
#   corpus         sphere to 128-d, More-Garbow-Hillstrom, the documented examples
#   shrink         a shrink that moves some vertices or none; the radius after one
#   ladder         when the ladder runs, its bottom rung, both channels through it
#
# BOBYQA (test_v0121_*.cpp; v0121_problems.hpp holds what they share):
#   validation     every std::invalid_argument, and the mid-run runtime_error
#   options        interpolation_points, shrink_initial_step, the callback
#   range          the coordinate range: refusals at entry, no non-finite point
#   tolerance      FtolReached at the ends of the finite range
#   start          section 2: the points (2.2) to (2.4), the swap, the first model (2.5)
#   inverse        H = W^-1: its start (2.11)-(2.15), update (4.9)-(4.14), shift (6.14)-(6.16),
#                  each against the explicit inverse of W; v0121_reference.hpp
#   model          the matrix (4.6), the move into M, the shift (6.17), the update of section 4
#   trsbox         section 3: (3.3), the line search, (3.4), (3.5), the arc (3.6), closed forms
#   altmov         section 3: the line step (3.11), the Cauchy step (3.13)-(3.15), vs an
#                  independent computation from the explicit Lagrange function
#   control        section 6 through runs whose trace shows the rule: (6.4) doubling and
#                  halving, the short step of (6.5), (6.6) ending at rho_end, (6.13)
#   rescue         section 5: (5.4) and (5.5), the iteration of (5.6) and (5.7) written out
#                  again, the inverse left behind, (5.3) and (5.12)
#   status         rho_end and the floor, StopValue, the exact cap and its prefix, the best point
#   bounds         every evaluation in the box, on a bound to the bit, faces and corners, a box
#                  never reached, the first radius against a narrow box
#   batch          the call shape of the starting points and RESCUE; scalar and batch identical
#   fp             determinism, scale covariance under powers of two, extremes, a success is right
#   facade         Minimizer against the template, every setter, refusals, reuse and move
#   header         flop/bobyqa.hpp alone; facade_header: flop/minimizer.hpp alone, no BOBYQA in it
#   corpus         the header and API examples, sphere, More-Garbow-Hillstrom, the report's
#                  section 7 problems against its Tables 1 to 3
#
# Both:
#   range          the coordinate range: refusals at entry, the error at run time
#   tolerance      the f tolerance tests at the ends of the finite range
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
    test_v0111_paper.cpp
    test_v0111_coefficients.cpp
    test_v0111_reference.cpp
    test_v0111_bookkeeping.cpp
    test_v0111_status.cpp
    test_v0111_bounds.cpp
    test_v0111_batch.cpp
    test_v0111_validation.cpp
    test_v0111_facade.cpp
    test_v0111_header.cpp
    test_v0111_fp.cpp
    test_v0111_corpus.cpp
    test_v0112_shrink.cpp
    test_v0112_ladder.cpp
    test_v0112_range.cpp
    test_v0112_tolerance.cpp
    test_v0113_active_set.cpp
    test_v0121_validation.cpp
    test_v0121_options.cpp
    test_v0121_range.cpp
    test_v0121_tolerance.cpp
    test_v0121_start.cpp
    test_v0121_inverse.cpp
    test_v0121_model.cpp
    test_v0121_trsbox.cpp
    test_v0121_altmov.cpp
    test_v0121_control.cpp
    test_v0121_rescue.cpp
    test_v0121_status.cpp
    test_v0121_bounds.cpp
    test_v0121_batch.cpp
    test_v0121_fp.cpp
    test_v0121_facade.cpp
    test_v0121_header.cpp
    test_v0121_facade_header.cpp
    test_v0121_corpus.cpp
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
