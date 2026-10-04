// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.1.1: the two coefficient sets.
//
// Gao and Han, Comput. Optim. Appl. 51(1), 2012, section 4.1, (4.1), state
// the adaptive set for n >= 2: alpha = 1, beta = 1 + 2/n, gamma = 0.75 -
// 1/(2n), delta = 1 - 1/n, and note that at n = 2 it is the standard set
// (Lagarias et al., (2.2)). At n = 1 their delta would be 0, outside the
// condition 0 < delta < 1 both papers impose ((2.1)); FLOP uses the standard
// set there, as documented. The expected values are computed here from the
// papers' formulas, not read from FLOP.
//
// The documented reason for the adaptive set is Gao and Han's Table 1: on
// their problem (4.3) the standard method stops far from the minimiser once
// n reaches 20 when sigma = 1e-4, and the adaptive one does not. FLOP is run
// on that problem from the paper's start with the paper's initial simplex
// ((3.1): FMINSEARCH's x0 + 0.05 x0_i e_i, which is x0 + 0.05 e_i at x0 =
// (1, ..., 1)) and tolerance ((4.2): TolX = 1e-4), and held to the claim.
// The paper's evaluation counts are printed beside FLOP's and not asserted:
// FMINSEARCH stops when TolX and TolFun both hold, FLOP on the radius alone.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

#include "flop/flop.hpp"
#include "v0111_problems.hpp"
#include "v0111_reference_nm.hpp"

namespace {

namespace nm = flop::detail::nelder_mead;
using Vec = std::vector<double>;

bool is_power_of_two(std::size_t n) {
    return (n & (n - 1)) == 0;
}

#if defined(FLOP_TEST_FAST_MATH)
constexpr bool kFastMath = true;
#else
constexpr bool kFastMath = false;
#endif

// Two values a division apart: identical in the strict binary (both are the
// correctly rounded value of the same quotient) and at every power of two
// (where the quotient is exact), within a few units in the last place under
// -ffast-math otherwise, which may divide through a reciprocal.
void expect_formula(double got, double want, std::size_t n) {
    if (kFastMath && !is_power_of_two(n)) {
        EXPECT_NEAR(got, want, 4.0 * std::numeric_limits<double>::epsilon() * std::fabs(want))
            << "n " << n;
        return;
    }
    EXPECT_TRUE(v0111::same_bits(got, want)) << "n " << n << ": " << got << " vs " << want;
}

}  // namespace

TEST(V0111Coefficients, TheAdaptiveSetIsGaoAndHansFormula4_1) {
    for (std::size_t n = 2; n <= 4096; ++n) {
        const nm::Coefficients c = nm::coefficients(n, true);
        const v0111::Coefficients want = v0111::gao_han(n);
        expect_formula(c.rho, want.rho, n);
        expect_formula(c.chi, want.chi, n);
        expect_formula(c.gamma, want.gamma, n);
        expect_formula(c.sigma, want.sigma, n);
    }
}

TEST(V0111Coefficients, EveryAdaptiveSetSatisfiesTheConditions2_1) {
    // Lagarias et al., (2.1): rho > 0, chi > 1, chi > rho, 0 < gamma < 1,
    // 0 < sigma < 1; and rho gamma < 1, the further restriction of Lemma 3.6
    // (p. 123) under which f_n and f_{n+1} share a limit.
    for (std::size_t n = 2; n <= 4096; ++n) {
        const nm::Coefficients c = nm::coefficients(n, true);
        EXPECT_GT(c.rho, 0.0) << n;
        EXPECT_GT(c.chi, 1.0) << n;
        EXPECT_GT(c.chi, c.rho) << n;
        EXPECT_GT(c.gamma, 0.0) << n;
        EXPECT_LT(c.gamma, 1.0) << n;
        EXPECT_GT(c.sigma, 0.0) << n;
        EXPECT_LT(c.sigma, 1.0) << n;
        EXPECT_LT(c.rho * c.gamma, 1.0) << n;
    }
}

TEST(V0111Coefficients, TheStandardSetIs2_2) {
    const v0111::Coefficients want = v0111::standard();
    for (const std::size_t n :
         {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{17}, std::size_t{1000}}) {
        const nm::Coefficients c = nm::coefficients(n, false);
        EXPECT_TRUE(v0111::same_bits(c.rho, want.rho)) << n;
        EXPECT_TRUE(v0111::same_bits(c.chi, want.chi)) << n;
        EXPECT_TRUE(v0111::same_bits(c.gamma, want.gamma)) << n;
        EXPECT_TRUE(v0111::same_bits(c.sigma, want.sigma)) << n;
    }
}

TEST(V0111Coefficients, OneDimensionUsesTheStandardSetWhateverTheOption) {
    const nm::Coefficients c = nm::coefficients(1, true);
    const v0111::Coefficients want = v0111::standard();
    EXPECT_TRUE(v0111::same_bits(c.chi, want.chi));
    EXPECT_TRUE(v0111::same_bits(c.gamma, want.gamma));
    EXPECT_TRUE(v0111::same_bits(c.sigma, want.sigma));
    // Observed, with the option on: from x_1 = 0, x_2 = 1 the expansion is at
    // -rho chi = -2 (formula (4.1) at n = 1 would give beta = 3, so -3) and
    // the outside contraction at -rho gamma = -1/2 (not -1/4).
    const v0111::Simplex one{{0.0}, {1.0}};
    {
        v0111::Table t;
        t.set(one[0], 1.0).set(one[1], 2.0).set({-1.0}, 0.5).set({-2.0}, 0.25);
        const v0111::Traced run = v0111::traced(t, one[0], v0111::options(1.0, 4, true));
        ASSERT_EQ(run.trace.size(), 4u);
        EXPECT_TRUE(v0111::same_bits(run.trace[3].x, v0111::trial_point(one, want.rho * want.chi)));
    }
    {
        v0111::Table t;
        t.set(one[0], 1.0).set(one[1], 2.0).set({-1.0}, 1.5).set({-0.5}, 1.0);
        const v0111::Traced run = v0111::traced(t, one[0], v0111::options(1.0, 4, true));
        ASSERT_EQ(run.trace.size(), 4u);
        EXPECT_TRUE(
            v0111::same_bits(run.trace[3].x, v0111::trial_point(one, want.rho * want.gamma)));
    }
    // And a whole run is the same with the option on and off.
    auto f = [](std::span<const double> x) { return std::cosh(x[0] - 0.7); };
    flop::nelder_mead::Options on = v0111::options(0.5, 500, true);
    on.stopping.xtol_abs = 1e-12;
    flop::nelder_mead::Options off = on;
    off.adaptive_coefficients = false;
    const v0111::Traced a = v0111::traced(f, Vec{3.0}, on);
    const v0111::Traced b = v0111::traced(f, Vec{3.0}, off);
    ASSERT_EQ(a.trace.size(), b.trace.size());
    for (std::size_t k = 0; k < a.trace.size(); ++k)
        EXPECT_TRUE(v0111::same_bits(a.trace[k].x, b.trace[k].x)) << k;
}

TEST(V0111Coefficients, AtTwoDimensionsTheTwoSetsAreTheSame) {
    // Gao and Han, section 4.1: "when n = 2, ANMS is identical to SNMS".
    const nm::Coefficients a = nm::coefficients(2, true);
    const nm::Coefficients s = nm::coefficients(2, false);
    EXPECT_TRUE(v0111::same_bits(a.rho, s.rho));
    EXPECT_TRUE(v0111::same_bits(a.chi, s.chi));
    EXPECT_TRUE(v0111::same_bits(a.gamma, s.gamma));
    EXPECT_TRUE(v0111::same_bits(a.sigma, s.sigma));
    for (auto* f : {&v0111::extended_rosenbrock, &v0111::beale, &v0111::sphere}) {
        flop::nelder_mead::Options on = v0111::options(0.5, 3000, true);
        on.stopping.xtol_abs = 1e-10;
        flop::nelder_mead::Options off = on;
        off.adaptive_coefficients = false;
        const v0111::Traced x = v0111::traced(f, Vec{-1.2, 1.0}, on);
        const v0111::Traced y = v0111::traced(f, Vec{-1.2, 1.0}, off);
        ASSERT_EQ(x.trace.size(), y.trace.size());
        for (std::size_t k = 0; k < x.trace.size(); ++k)
            EXPECT_TRUE(v0111::same_bits(x.trace[k].x, y.trace[k].x)) << k;
        EXPECT_EQ(x.r.status, y.r.status);
    }
}

TEST(V0111Coefficients, AboveTwoDimensionsTheOptionChangesTheTrajectory) {
    for (const std::size_t n : {std::size_t{3}, std::size_t{4}, std::size_t{8}}) {
        flop::nelder_mead::Options on = v0111::options(1.0, 400, true);
        flop::nelder_mead::Options off = on;
        off.adaptive_coefficients = false;
        const v0111::Traced a = v0111::traced(v0111::sphere, Vec(n, 0.0), on);
        const v0111::Traced b = v0111::traced(v0111::sphere, Vec(n, 0.0), off);
        bool differ = a.trace.size() != b.trace.size();
        for (std::size_t k = 0; !differ && k < a.trace.size(); ++k)
            differ = !v0111::same_bits(a.trace[k].x, b.trace[k].x);
        EXPECT_TRUE(differ) << "n " << n;
        // And each run uses its own set: the reference with the documented
        // coefficients replays each trace.
        for (const auto* p : {&a, &b}) {
            const flop::nelder_mead::Options& o = p == &a ? on : off;
            const v0111::AuditReport rep = v0111::audit(Vec(n, 0.0), o, p->trace, p->r, false);
            EXPECT_TRUE(rep.points_agree && rep.same_length) << rep.detail;
        }
    }
}

namespace {

// One row of Gao and Han's Table 1: the problem, the dimension, and the
// paper's nfeval and final f for SNMS and ANMS.
struct Table1Row {
    double eps;
    double sigma;
    std::size_t n;
    std::size_t snms_nfeval;
    double snms_f;
    std::size_t anms_nfeval;
    double anms_f;
};

// Table 1, the rows at n = 10, 20 and 30, transcribed.
const Table1Row table1[] = {
    {.eps = 0.0,
     .sigma = 0.0,
     .n = 10,
     .snms_nfeval = 1228,
     .snms_f = 1.4968e-8,
     .anms_nfeval = 898,
     .anms_f = 5.9143e-9},
    {.eps = 0.0,
     .sigma = 0.0,
     .n = 20,
     .snms_nfeval = 12614,
     .snms_f = 1.0429e-7,
     .anms_nfeval = 2259,
     .anms_f = 1.1343e-8},
    {.eps = 0.0,
     .sigma = 0.0,
     .n = 30,
     .snms_nfeval = 38161,
     .snms_f = 7.9366e-7,
     .anms_nfeval = 4072,
     .anms_f = 1.5503e-8},
    {.eps = 0.05,
     .sigma = 0.0,
     .n = 10,
     .snms_nfeval = 1123,
     .snms_f = 1.1166e-7,
     .anms_nfeval = 910,
     .anms_f = 9.0552e-9},
    {.eps = 0.05,
     .sigma = 0.0,
     .n = 20,
     .snms_nfeval = 9454,
     .snms_f = 2.7389e-7,
     .anms_nfeval = 2548,
     .anms_f = 1.8433e-8},
    {.eps = 0.05,
     .sigma = 0.0,
     .n = 30,
     .snms_nfeval = 55603,
     .snms_f = 5.3107e-3,
     .anms_nfeval = 5067,
     .anms_f = 2.6663e-8},
    {.eps = 0.0,
     .sigma = 1e-4,
     .n = 10,
     .snms_nfeval = 1587,
     .snms_f = 2.0101e-8,
     .anms_nfeval = 1088,
     .anms_f = 1.4603e-8},
    {.eps = 0.0,
     .sigma = 1e-4,
     .n = 20,
     .snms_nfeval = 24313,
     .snms_f = 2.2788,
     .anms_nfeval = 4134,
     .anms_f = 2.8482e-8},
    {.eps = 0.0,
     .sigma = 1e-4,
     .n = 30,
     .snms_nfeval = 43575,
     .snms_f = 4.5166e2,
     .anms_nfeval = 13148,
     .anms_f = 4.0639e-8},
    {.eps = 0.05,
     .sigma = 1e-4,
     .n = 10,
     .snms_nfeval = 1787,
     .snms_f = 3.1878e-8,
     .anms_nfeval = 994,
     .anms_f = 6.0454e-9},
    {.eps = 0.05,
     .sigma = 1e-4,
     .n = 20,
     .snms_nfeval = 20824,
     .snms_f = 1.2984e1,
     .anms_nfeval = 3788,
     .anms_f = 1.5294e-8},
    {.eps = 0.05,
     .sigma = 1e-4,
     .n = 30,
     .snms_nfeval = 39557,
     .snms_f = 1.8108e2,
     .anms_nfeval = 10251,
     .anms_f = 4.0331e-8},
};

// The rows where the paper reports the standard method failing: sigma =
// 1e-4 at n >= 20 ("terminates prematurely"), and (0.05, 0) at large n
// ("fails to find a good approximate solution").
bool snms_fails(const Table1Row& row) {
    return row.n >= 20 && (row.sigma > 0.0 || (row.eps > 0.0 && row.n >= 30));
}

// The line between success and failure, from the table itself: the
// geometric mean of the worst ANMS final f and the best failing SNMS final
// f over the rows above. Every success in the table is below it and every
// failure above, by a factor of about eighty on each side.
double table1_bar() {
    double worst_success = 0.0;
    double best_failure = std::numeric_limits<double>::max();
    for (const Table1Row& row : table1) {
        worst_success = std::max(worst_success, row.anms_f);
        if (snms_fails(row))
            best_failure = std::min(best_failure, row.snms_f);
        else
            worst_success = std::max(worst_success, row.snms_f);
    }
    return std::sqrt(worst_success * best_failure);
}

}  // namespace

TEST(V0111Coefficients, GaoAndHanTable1TheStandardSetStallsAboveTenDimensions) {
    const double bar = table1_bar();
    for (const Table1Row& row : table1) {
        const v0111::GaoHan43 f{.eps = row.eps, .sigma = row.sigma};
        const Vec x0(row.n, 1.0);
        flop::Result res[2];
        for (const bool adaptive : {false, true}) {
            flop::nelder_mead::Options o = v0111::options(0.05, 1000000, adaptive);
            o.stopping.xtol_abs = 1e-4;
            res[adaptive ? 1 : 0] = flop::nelder_mead::minimize(f, x0, o);
        }
        std::cout << "[ table 1  ] (" << row.eps << ", " << row.sigma << ") n " << row.n
                  << "  SNMS " << res[0].evaluations << " evaluations, f " << res[0].f << " (paper "
                  << row.snms_nfeval << ", " << row.snms_f << ")  ANMS " << res[1].evaluations
                  << ", f " << res[1].f << " (paper " << row.anms_nfeval << ", " << row.anms_f
                  << ")\n";
        EXPECT_EQ(res[1].status, flop::Status::XtolReached);
        EXPECT_LE(res[1].f, bar) << "ANMS (" << row.eps << ", " << row.sigma << ") n " << row.n;
        if (snms_fails(row))
            EXPECT_GT(res[0].f, bar) << "SNMS (" << row.eps << ", " << row.sigma << ") n " << row.n;
        else
            EXPECT_LE(res[0].f, bar) << "SNMS (" << row.eps << ", " << row.sigma << ") n " << row.n;
    }
}

TEST(V0111Coefficients, GaoAndHanFigures1And2TheAdaptiveSetReflectsLess) {
    // Section 3 and 4.1, Figs. 1 and 2, on x'x from (1, ..., 1) with the
    // setup above: the standard method's share of reflection steps grows
    // toward one with n, and the adaptive one's is "at most 0.45 for all
    // dimensions". A reflection step is an iteration whose accepted point is
    // x_r (tau = rho), whether it ended in step 2 or step 3.
    auto quadratic = [](std::span<const double> x) {
        double s = 0.0;
        for (const double v : x) s += v * v;
        return s;
    };
    for (const std::size_t n : {std::size_t{10}, std::size_t{20}, std::size_t{40}}) {
        double share[2] = {0.0, 0.0};
        for (const bool adaptive : {false, true}) {
            flop::nelder_mead::Options o = v0111::options(0.05, 1000000, adaptive);
            o.stopping.xtol_abs = 1e-4;
            const v0111::Traced run = v0111::traced(quadratic, Vec(n, 1.0), o);
            v0111::ReferenceRun ref;
            const v0111::AuditReport rep =
                v0111::audit(Vec(n, 1.0), o, run.trace, run.r, false, &ref, true);
            ASSERT_TRUE(rep.points_agree && rep.same_length) << rep.detail;
            std::size_t reflections = 0;
            const double rho = v0111::documented(n, adaptive).rho;
            for (const v0111::Step& s : ref.steps)
                if (s.termination != v0111::Termination::Shrink && s.tau == rho) ++reflections;
            share[adaptive ? 1 : 0] =
                static_cast<double>(reflections) / static_cast<double>(ref.steps.size());
        }
        std::cout << "[ figs 1-2 ] n " << n << ": reflection share SNMS " << share[0] << ", ANMS "
                  << share[1] << '\n';
        EXPECT_LE(share[1], 0.45) << n;
        EXPECT_GT(share[0], share[1]) << n;
    }
}
