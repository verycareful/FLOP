// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.0.2: Powell's own results, Tables 1 and 2 of the paper (page 65).
//
// The paper runs problems (A) to (J) from x0 = (1, ..., 1) with rho_beg =
// 1/2 and rho_end = 1e-3 (Table 1) and 1e-4 (Table 2), and reports the
// number of function evaluations, the final F, the final greatest violation
// and the final distance to a solution. Those numbers are the one yardstick
// for FLOP's iteration rules that owes nothing to another implementation:
// they were produced by the algorithm the paper describes. FLOP is held to
// the same accuracy and to an evaluation count within a factor of two of
// Powell's, the factor at which his arithmetic (single precision on a
// Sparc), the rounding of the printed values and the path noise of two
// floating-point models stop deciding the outcome.
//
// Problems (A) to (I) are the shared problems 1 to 9; (J) is HS108, whose
// solution is not unique, so its distance is not compared.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>
#include <span>
#include <string>
#include <vector>

#include "flop/flop.hpp"
#include "v0101_problems.hpp"

namespace {

// A gross rule error shows on these tables as a factor of three to ten (a
// flattened simplex crawling a valley was five). Below two the comparison
// is path noise: Powell's numbers are one single-precision path, each
// binary of this suite is another, and the growth decisions are threshold
// comparisons that rounding can flip. The finer comparison is statistical,
// against NLopt over twenty starts, and is not a test.
constexpr double kEvaluationFactor = 2.0;
constexpr double kAccuracyFactor = 2.0;
// One factor on the variables; F, quadratic in them near a nondegenerate
// minimum, gets its square, or the F bar would be the stricter of the two
// by a factor of root two for no reason of its own.
constexpr double kObjectiveFactor = kAccuracyFactor * kAccuracyFactor;
// The paper's runs were in single precision (page 64), whose resolution on
// constraint values of order one is about 6e-8, so a violation printed as
// 0 means below that; the floor stands in for the printed zero.
constexpr double kSinglePrecisionFloor = 1e-7;

// One row of a table: Powell's count, final F, final violation, final
// distance, and the half-unit of the last printed digit of F, so that a
// printed "-0.5000" is read as -0.5 within 5e-5.
struct Row {
    std::size_t evaluations;
    double f;
    double f_half_unit;
    double violation;
    double distance;
};

struct Problem {
    char letter;
    double (*f)(std::span<const double>);
    void (*c)(std::span<const double>, std::span<double>);
    std::size_t m;
    std::size_t n;
    std::vector<double> x_opt;  // empty when the solution is not unique
    double f_opt;
    Row table1;  // rho_end = 1e-3
    Row table2;  // rho_end = 1e-4
};

Row row(std::size_t evaluations, double f, double f_half_unit, double violation, double distance) {
    return {.evaluations = evaluations,
            .f = f,
            .f_half_unit = f_half_unit,
            .violation = violation,
            .distance = distance};
}

// A value as the paper prints it, to four decimals.
double printed4(double x) {
    return std::round(x * 1e4) / 1e4;
}

std::vector<Problem> problems() {
    std::vector<v0101::Constrained> shared = v0101::powell_1994_problems();
    std::vector<Problem> out;
    auto add = [&](char letter, const v0101::Constrained& p, const Row& t1, const Row& t2) {
        out.push_back({.letter = letter,
                       .f = p.f,
                       .c = p.c,
                       .m = p.m,
                       .n = p.x0.size(),
                       .x_opt = p.x_opt,
                       .f_opt = p.f_opt,
                       .table1 = t1,
                       .table2 = t2});
    };
    // Tables 1 and 2, transcribed. A final violation printed as 0 is 0.
    add('A', shared[0], row(37, 1.8e-5, 5e-7, 0.0, 3.3e-3), row(65, 1.2e-7, 5e-9, 0.0, 2.8e-4));
    add('B', shared[1], row(37, -0.5, 5e-5, 2.0e-6, 1.3e-3), row(44, -0.5, 5e-5, 6.0e-8, 6.1e-5));
    add('C', shared[2], row(45, -0.0786, 5e-5, 4.7e-6, 1.4e-3),
        row(60, -0.0786, 5e-5, 0.0, 9.2e-6));
    add('D', shared[3], row(100, 3.1e-5, 5e-7, 0.0, 1.3e-2), row(173, 6.4e-7, 5e-9, 0.0, 1.7e-3));
    add('E', shared[4], row(347, 4.0e-3, 5e-5, 0.0, 1.4e-1), row(698, 9.5e-5, 5e-7, 0.0, 2.2e-2));
    add('F', shared[5], row(30, printed4(-std::numbers::sqrt2), 5e-5, 3.0e-6, 1.2e-3),
        row(41, printed4(-std::numbers::sqrt2), 5e-5, 1.5e-7, 4.6e-5));
    add('G', shared[6], row(29, -3.0, 5e-5, 1.3e-4, 5.9e-3), row(33, -3.0, 5e-5, 0.0, 2.4e-5));
    add('H', shared[7], row(74, -44.0, 5e-5, 2.9e-6, 1.4e-3), row(87, -44.0, 5e-5, 2.2e-6, 1.2e-3));
    add('I', shared[8], row(198, 680.6303, 5e-5, 5.7e-5, 5.9e-3),
        row(212, 680.6303, 5e-5, 0.0, 5.3e-3));
    out.push_back({.letter = 'J',
                   .f = v0101::hs108_f,
                   .c = v0101::hs108_c,
                   .m = v0101::kHs108Constraints,
                   .n = 9,
                   .x_opt = {},
                   .f_opt = -0.5 * std::numbers::sqrt3,
                   .table1 = row(143, -0.8660, 5e-5, 1.0e-6, 8.9e-4),
                   .table2 = row(173, -0.8660, 5e-5, 1.2e-7, 9.5e-5)});
    return out;
}

double distance(std::span<const double> x, std::span<const double> x_opt) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) s += (x[i] - x_opt[i]) * (x[i] - x_opt[i]);
    return std::sqrt(s);
}

// Runs one problem at Powell's settings and holds it to a row. What the run
// produced is printed beside the table so a tightening of the factor can be
// read off the capture.
void hold_to(const Problem& p, double rho_end, const Row& row, int table) {
    const std::vector<double> x0(p.n, 1.0);
    flop::cobyla::Options opts;
    opts.initial_step = 0.5;
    opts.final_trust_radius = rho_end;
    opts.stopping.max_evaluations = 5000;  // a runaway, not a stopping rule
    const flop::Result r = p.m ? flop::cobyla::minimize(p.f, p.c, p.m, x0, opts)
                               : flop::cobyla::minimize(p.f, x0, opts);
    const double f_error = std::fabs(r.f - p.f_opt);
    const double f_allowed = std::max(std::fabs(row.f - p.f_opt), row.f_half_unit);
    const bool have_x = !p.x_opt.empty();
    const double dist = have_x ? distance(r.x, p.x_opt) : 0.0;

    std::cout << "[ table " << table << "  ] (" << p.letter << ") evaluations " << r.evaluations
              << " (Powell " << row.evaluations << "), |F - F_opt| " << f_error << " (Powell "
              << f_allowed << "), violation " << r.max_constraint_violation << " (Powell "
              << row.violation << ")";
    if (have_x) std::cout << ", distance " << dist << " (Powell " << row.distance << ")";
    std::cout << '\n';

    EXPECT_EQ(r.status, flop::Status::XtolReached) << p.letter;
    EXPECT_LE(static_cast<double>(r.evaluations),
              kEvaluationFactor * static_cast<double>(row.evaluations))
        << p.letter;
    EXPECT_LE(f_error, kObjectiveFactor * f_allowed) << p.letter;
    EXPECT_LE(r.max_constraint_violation,
              kAccuracyFactor * std::max(row.violation, kSinglePrecisionFloor))
        << p.letter;
    // Page 66: the accuracy in the variables is controlled approximately by
    // rho_end, so a printed distance below it is a fortunate path, not a
    // bar; rho_end is the floor.
    if (have_x) {
        EXPECT_LE(dist, std::max(kAccuracyFactor * row.distance, rho_end)) << p.letter;
    }
}

class V0102Paper : public ::testing::TestWithParam<Problem> {};

}  // namespace

TEST_P(V0102Paper, Table1AtRhoEnd1e3) {
    hold_to(GetParam(), 1e-3, GetParam().table1, 1);
}

TEST_P(V0102Paper, Table2AtRhoEnd1e4) {
    hold_to(GetParam(), 1e-4, GetParam().table2, 2);
}

INSTANTIATE_TEST_SUITE_P(Problems, V0102Paper, ::testing::ValuesIn(problems()),
                         [](const ::testing::TestParamInfo<Problem>& tpi) {
                             return std::string(1, tpi.param.letter);
                         });
