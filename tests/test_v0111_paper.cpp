// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.1.1: Nelder-Mead against the papers, rule by rule.
//
// Lagarias, Reeds, Wright and Wright, SIAM J. Optim. 9(1), 1998, section
// 2.1, pp. 115-117, states one iteration as five steps with four acceptance
// tests and two tie-breaking rules. The first half of this suite holds FLOP
// to each of them with scripted runs in exact arithmetic (v0111_problems.hpp
// says why that is exact): the test picks f at every point the paper says
// the method evaluates, on both sides of each inequality and AT its
// boundary, since which inequalities are strict is exactly what an
// implementation can get wrong, and checks the points FLOP evaluates next
// against the paper's formulas to the bit. Every expected point is computed
// in the test from (2.4) to (2.7) and step 5, never typed.
//
// The second half holds FLOP's runs to what the papers prove about the
// method: the structure (2.9) of a nonshrink step and the change index
// (2.8), p. 118's bound on how long the worst value can stand, Lemma 3.3,
// the evaluation counts of section 3.1, the volume of Lemma 3.1, the
// one-dimensional properties of section 4 (p. 124's diameter (4.1), p.
// 125's "never terminates in step 2", Lemma 4.3's proximity bound), Lemma
// 3.5 (no shrink on a strictly convex function), and Gao and Han's Theorem
// 2.1 (Comput. Optim. Appl. 51(1), 2012, (2.2)). Those are read off the
// reference transcription (v0111_reference_nm.hpp) replaying FLOP's own
// trace, so the simplex each property is checked on is FLOP's.
//
// At n = 2 the two coefficient sets are the same (Gao and Han, section 4.1),
// so the scripted n = 2 runs hold for both.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "flop/flop.hpp"
#include "v0111_problems.hpp"
#include "v0111_reference_nm.hpp"

namespace {

using v0111::Simplex;
using v0111::Table;
using v0111::Termination;
using v0111::trial_point;
using Vec = std::vector<double>;

using v0111::Traced;

// Runs Nelder-Mead on a table from x0 with edge h and a cap.
Traced run_table(Table& t, const Vec& x0, double h, std::size_t cap, bool adaptive = true) {
    return v0111::traced(t, x0, v0111::options(h, cap, adaptive));
}

// The trace holds exactly the expected points, to the bit, in order, and
// every point stayed in exact arithmetic.
void expect_points(const v0111::Trace& trace, const std::vector<Vec>& expected) {
    ASSERT_EQ(trace.size(), expected.size());
    for (std::size_t k = 0; k < expected.size(); ++k)
        EXPECT_TRUE(v0111::same_bits(trace[k].x, expected[k])) << "evaluation " << k;
    EXPECT_TRUE(v0111::all_exact(trace)) << "the run left exact arithmetic";
}

// The n = 2 scripted start: x0 = (0, 0), h = 1, so the initial simplex is
// s0 = (0, 0), s1 = (1, 0), s2 = (0, 1), ranked s0, s1, s2 by f = 1, 2, 3.
Vec s0() {
    return {0.0, 0.0};
}
Vec s1() {
    return {1.0, 0.0};
}
Vec s2() {
    return {0.0, 1.0};
}
Simplex start() {
    return {s0(), s1(), s2()};
}

Table start_table() {
    Table t;
    t.set(s0(), 1.0).set(s1(), 2.0).set(s2(), 3.0);
    return t;
}

v0111::Coefficients std2() {
    return v0111::standard();
}
Vec xr() {
    return trial_point(start(), std2().rho);
}  // (1, -1)
Vec xe() {
    return trial_point(start(), std2().rho * std2().chi);
}  // (1.5, -2)
Vec xc() {
    return trial_point(start(), std2().rho * std2().gamma);
}  // (0.75, -0.5)
Vec xcc() {
    return trial_point(start(), -std2().gamma);
}  // (0.25, 0.5)

// The point the next iteration reflects, given the simplex the paper says
// the accepted point produced.
Vec next_reflection(const Simplex& ranked) {
    return trial_point(ranked, std2().rho);
}

}  // namespace

// ============================================================================
// Step 2, reflect: accept x_r when f(x_1) <= f(x_r) < f(x_n) (p. 116).
// ============================================================================

TEST(V0111Paper, AReflectionTyingTheBestIsAccepted) {
    // f_r == f_1 is inside the interval, and not below f_1, so no expansion.
    Table t = start_table();
    t.set(xr(), 1.0);
    const Traced run = run_table(t, s0(), 1.0, 5);
    // x_r ties x_1 and ranks after it: (s0, x_r, s1), so s1 is reflected.
    expect_points(run.trace, {s0(), s1(), s2(), xr(), next_reflection({s0(), xr(), s1()})});
}

TEST(V0111Paper, AReflectionJustBelowTheNextWorstIsAccepted) {
    Table t = start_table();
    t.set(xr(), std::nextafter(2.0, 0.0));
    const Traced run = run_table(t, s0(), 1.0, 5);
    expect_points(run.trace, {s0(), s1(), s2(), xr(), next_reflection({s0(), xr(), s1()})});
}

TEST(V0111Paper, AReflectionEqualToTheNextWorstContractsOutside) {
    // f_r == f_n is outside the reflection interval and inside the outside
    // contraction's, f_n <= f_r < f_{n+1} (2.6).
    Table t = start_table();
    t.set(xr(), 2.0).set(xc(), 1.5);
    const Traced run = run_table(t, s0(), 1.0, 5);
    expect_points(run.trace, {s0(), s1(), s2(), xr(), xc()});
}

// ============================================================================
// Step 3, expand: if f_r < f_1, accept x_e when f_e < f_r, else x_r (p. 116).
// ============================================================================

TEST(V0111Paper, AnExpansionStrictlyBelowTheReflectionIsAccepted) {
    Table t = start_table();
    t.set(xr(), 0.5).set(xe(), 0.25);
    const Traced run = run_table(t, s0(), 1.0, 6);
    expect_points(run.trace, {s0(), s1(), s2(), xr(), xe(), next_reflection({xe(), s0(), s1()})});
}

TEST(V0111Paper, AnExpansionEqualToTheReflectionKeepsTheReflection) {
    Table t = start_table();
    t.set(xr(), 0.5).set(xe(), 0.5);
    const Traced run = run_table(t, s0(), 1.0, 6);
    expect_points(run.trace, {s0(), s1(), s2(), xr(), xe(), next_reflection({xr(), s0(), s1()})});
}

TEST(V0111Paper, TheExpansionIsComparedWithTheReflectionNotWithTheBest) {
    // f_r < f_e < f_1: Nelder and Mead's 1965 rule (accept x_e if f_e < f_1)
    // would take x_e; the paper's rule (section 3.1, property 4, p. 120)
    // takes the better of the two, x_r.
    Table t = start_table();
    t.set(xr(), 0.5).set(xe(), 0.75);
    const Traced run = run_table(t, s0(), 1.0, 6);
    expect_points(run.trace, {s0(), s1(), s2(), xr(), xe(), next_reflection({xr(), s0(), s1()})});
}

TEST(V0111Paper, AnExpansionWorseThanTheBestKeepsTheReflection) {
    Table t = start_table();
    t.set(xr(), 0.5).set(xe(), 5.0);
    const Traced run = run_table(t, s0(), 1.0, 6);
    expect_points(run.trace, {s0(), s1(), s2(), xr(), xe(), next_reflection({xr(), s0(), s1()})});
}

// ============================================================================
// Step 4a, outside: f_n <= f_r < f_{n+1}; accept x_c when f_c <= f_r (2.6).
// ============================================================================

TEST(V0111Paper, AnOutsideContractionBelowOrAtTheReflectionIsAccepted) {
    for (const double fc : {2.25, 2.5}) {
        Table t = start_table();
        t.set(xr(), 2.5).set(xc(), fc);
        const Traced run = run_table(t, s0(), 1.0, 6);
        // x_c is worse than every kept vertex, so it is reflected next.
        expect_points(run.trace,
                      {s0(), s1(), s2(), xr(), xc(), next_reflection({s0(), s1(), xc()})});
    }
}

TEST(V0111Paper, AnOutsideContractionAboveTheReflectionShrinks) {
    Table t = start_table();
    t.set(xr(), 2.5).set(xc(), 2.75);
    const Traced run = run_table(t, s0(), 1.0, 7);
    const double sigma = std2().sigma;
    expect_points(run.trace, {s0(), s1(), s2(), xr(), xc(), v0111::shrink_point(s0(), s1(), sigma),
                              v0111::shrink_point(s0(), s2(), sigma)});
}

// ============================================================================
// Step 4b, inside: f_r >= f_{n+1}; accept x_cc when f_cc < f_{n+1} (2.7).
// ============================================================================

TEST(V0111Paper, AReflectionEqualToTheWorstContractsInside) {
    for (const double fr : {3.0, 4.0}) {
        Table t = start_table();
        t.set(xr(), fr).set(xcc(), 2.5);
        const Traced run = run_table(t, s0(), 1.0, 6);
        expect_points(run.trace,
                      {s0(), s1(), s2(), xr(), xcc(), next_reflection({s0(), s1(), xcc()})});
    }
}

TEST(V0111Paper, AnInsideContractionEqualToOrAboveTheWorstShrinks) {
    for (const double fcc : {3.0, 3.5}) {
        Table t = start_table();
        t.set(xr(), 4.0).set(xcc(), fcc);
        const Traced run = run_table(t, s0(), 1.0, 7);
        const double sigma = std2().sigma;
        expect_points(run.trace,
                      {s0(), s1(), s2(), xr(), xcc(), v0111::shrink_point(s0(), s1(), sigma),
                       v0111::shrink_point(s0(), s2(), sigma)});
    }
}

// ============================================================================
// The four trial points and the shrink, at both coefficient sets.
// ============================================================================

TEST(V0111Paper, EveryTrialPointIsOnTheLineAtThePapersCoefficient) {
    // (2.12) and (2.13): x_r, x_e, x_c, x_cc are xbar + tau (xbar - x_{n+1})
    // at tau = rho, rho chi, rho gamma, -gamma; a shrink moves x_i to x_1 +
    // sigma (x_i - x_1), in rank order. x0 = 0, h = 1, f(s_i) = 1 + i, so
    // x_{n+1} = s_n and x_n = s_{n-1} with f = n.
    for (const std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
        for (const bool adaptive : {false, true}) {
            const v0111::Coefficients c = v0111::documented(n, adaptive);
            const Vec x0(n, 0.0);
            const Simplex ranked = v0111::axis_simplex(x0, 1.0);
            auto table = [&]() {
                Table t;
                for (std::size_t i = 0; i <= n; ++i) t.set(ranked[i], 1.0 + static_cast<double>(i));
                return t;
            };
            const Vec r = trial_point(ranked, c.rho);
            const auto fn = static_cast<double>(n);
            std::vector<Vec> init(ranked.begin(), ranked.end());
            {  // expansion
                Table t = table();
                t.set(r, 0.0).set(trial_point(ranked, c.rho * c.chi), 0.0);
                const Traced run = run_table(t, x0, 1.0, n + 3, adaptive);
                std::vector<Vec> want = init;
                want.push_back(r);
                want.push_back(trial_point(ranked, c.rho * c.chi));
                expect_points(run.trace, want);
            }
            {  // outside contraction: f_r = f_n
                Table t = table();
                t.set(r, fn).set(trial_point(ranked, c.rho * c.gamma), 0.0);
                const Traced run = run_table(t, x0, 1.0, n + 3, adaptive);
                std::vector<Vec> want = init;
                want.push_back(r);
                want.push_back(trial_point(ranked, c.rho * c.gamma));
                expect_points(run.trace, want);
            }
            {  // inside contraction, then a shrink
                Table t = table();
                t.set(r, 100.0).set(trial_point(ranked, -c.gamma), 100.0);
                const Traced run = run_table(t, x0, 1.0, n + 3 + n, adaptive);
                std::vector<Vec> want = init;
                want.push_back(r);
                want.push_back(trial_point(ranked, -c.gamma));
                for (std::size_t i = 1; i <= n; ++i)
                    want.push_back(v0111::shrink_point(ranked[0], ranked[i], c.sigma));
                expect_points(run.trace, want);
            }
        }
    }
}

// ============================================================================
// The nonshrink ordering rule (p. 116), read as p. 118's example reads it.
// ============================================================================

TEST(V0111Paper, ThePage118ExampleVerbatim) {
    // n = 4, values (1, 2, 2, 3, 3), f(v) = 2: the next simplex has values
    // (1, 2, 2, 2, 3) with x_4 = v, k* = 4. The construction order breaks
    // the initial ties (s1 before s2, s3 before s4), so v = x_r of s4.
    // Three reflections then show the order: the next one reflects s3 (so v
    // is not last), and after two more accepted points whose values place
    // them between, v is reflected before s2 and s2 before s1.
    for (const bool adaptive : {false, true}) {
        const Vec x0(4, 0.0);
        const Simplex init = v0111::axis_simplex(x0, 1.0);
        Table t;
        const double values[5] = {1.0, 2.0, 2.0, 3.0, 3.0};
        for (std::size_t i = 0; i < 5; ++i) t.set(init[i], values[i]);
        const double rho = v0111::documented(4, adaptive).rho;
        const Vec v = trial_point(init, rho);
        t.set(v, 2.0);
        // The paper's simplex after the step: (s0, s1, s2, v, s3).
        const Vec p1 = trial_point({init[0], init[1], init[2], v, init[3]}, rho);
        t.set(p1, 1.5);
        const Vec p2 = trial_point({init[0], p1, init[1], init[2], v}, rho);
        t.set(p2, 1.75);
        const Vec p3 = trial_point({init[0], p1, p2, init[1], init[2]}, rho);
        const Traced run = run_table(t, x0, 1.0, 9, adaptive);
        expect_points(run.trace, {init[0], init[1], init[2], init[3], init[4], v, p1, p2, p3});
    }
}

TEST(V0111Paper, AnAcceptedPointTyingTheBestRanksAfterIt) {
    // x_r ties x_1, so the paper's simplex is (s0, x_r, s1) with s0 still
    // first. Forcing a shrink next shows which vertex is x_1: the shrink
    // moves the others toward it.
    Table t = start_table();
    t.set(xr(), 1.0);
    const Simplex after{s0(), xr(), s1()};
    const Vec r2 = trial_point(after, std2().rho);
    const Vec cc2 = trial_point(after, -std2().gamma);
    t.set(r2, 10.0).set(cc2, 10.0);
    const Traced run = run_table(t, s0(), 1.0, 8);
    expect_points(run.trace,
                  {s0(), s1(), s2(), xr(), r2, cc2, v0111::shrink_point(s0(), xr(), std2().sigma),
                   v0111::shrink_point(s0(), s1(), std2().sigma)});
}

TEST(V0111Paper, AnAcceptedPointTyingTheNextWorstRanksAfterIt) {
    // An outside contraction accepted at f_c == f_n: (s0, s1, x_c), so x_c
    // is the next vertex reflected, not s1.
    Table t = start_table();
    t.set(xr(), 2.5).set(xc(), 2.0);
    const Traced run = run_table(t, s0(), 1.0, 6);
    expect_points(run.trace, {s0(), s1(), s2(), xr(), xc(), next_reflection({s0(), s1(), xc()})});
}

TEST(V0111Paper, AnAcceptedPointTakesTheFirstRankWhoseValueExceedsIt) {
    // Strictly best (an accepted expansion) goes first, strictly between
    // goes between, and the strictly worst of the kept goes last; the next
    // reflection shows which vertex is last.
    {
        Table t = start_table();
        t.set(xr(), 0.5).set(xe(), 0.25);
        const Traced run = run_table(t, s0(), 1.0, 6);
        expect_points(run.trace,
                      {s0(), s1(), s2(), xr(), xe(), next_reflection({xe(), s0(), s1()})});
    }
    {
        Table t = start_table();
        t.set(xr(), 1.5);
        const Traced run = run_table(t, s0(), 1.0, 5);
        expect_points(run.trace, {s0(), s1(), s2(), xr(), next_reflection({s0(), xr(), s1()})});
    }
    {
        Table t = start_table();
        t.set(xr(), 2.5).set(xc(), 2.25);
        const Traced run = run_table(t, s0(), 1.0, 6);
        expect_points(run.trace,
                      {s0(), s1(), s2(), xr(), xc(), next_reflection({s0(), s1(), xc()})});
    }
}

// ============================================================================
// The shrink ordering rule (pp. 116-117) and the initial ordering.
// ============================================================================

namespace {

// Shrinks the start simplex (x_r and x_cc both at or above f_3), gives the
// two new points the values a and b, then shrinks again; the second shrink's
// points show which vertex the paper's rule made x_1.
struct DoubleShrink {
    Vec v1 = v0111::shrink_point(s0(), s1(), std2().sigma);  // (0.5, 0)
    Vec v2 = v0111::shrink_point(s0(), s2(), std2().sigma);  // (0, 0.5)
};

}  // namespace

TEST(V0111Paper, AfterAShrinkTheBestStaysFirstWhenANewPointTiesIt) {
    const DoubleShrink d;
    Table t = start_table();
    t.set(xr(), 4.0).set(xcc(), 3.5).set(d.v1, 1.0).set(d.v2, 5.0);
    const Simplex after{s0(), d.v1, d.v2};  // v1 ties x_1 = s0; s0 stays first
    const Vec r2 = trial_point(after, std2().rho);
    const Vec cc2 = trial_point(after, -std2().gamma);
    t.set(r2, 10.0).set(cc2, 10.0);
    const Traced run = run_table(t, s0(), 1.0, 11);
    expect_points(run.trace, {s0(), s1(), s2(), xr(), xcc(), d.v1, d.v2, r2, cc2,
                              v0111::shrink_point(s0(), d.v1, std2().sigma),
                              v0111::shrink_point(s0(), d.v2, std2().sigma)});
}

TEST(V0111Paper, AfterAShrinkAStrictlyBetterNewPointBecomesTheBest) {
    const DoubleShrink d;
    Table t = start_table();
    t.set(xr(), 4.0).set(xcc(), 3.5).set(d.v1, 0.5).set(d.v2, 5.0);
    const Simplex after{d.v1, s0(), d.v2};
    const Vec r2 = trial_point(after, std2().rho);
    const Vec cc2 = trial_point(after, -std2().gamma);
    t.set(r2, 10.0).set(cc2, 10.0);
    const Traced run = run_table(t, s0(), 1.0, 11);
    expect_points(run.trace, {s0(), s1(), s2(), xr(), xcc(), d.v1, d.v2, r2, cc2,
                              v0111::shrink_point(d.v1, s0(), std2().sigma),
                              v0111::shrink_point(d.v1, d.v2, std2().sigma)});
}

TEST(V0111Paper, AfterAShrinkTiedNewPointsKeepTheOrderTheirOriginalsHad) {
    // FLOP's documented choice of "whatever rule is used to define the
    // original ordering" (p. 117). Ranked s0, s2, s1 before the shrink (f =
    // 1, 2, 3), so the shrink evaluates s2's image first, and with both new
    // points tied s1's image stays last and is reflected next. A rule that
    // fell back on storage order would reflect s2's image.
    Table t;
    t.set(s0(), 1.0).set(s1(), 3.0).set(s2(), 2.0);
    const Simplex ranked{s0(), s2(), s1()};
    const Vec r = trial_point(ranked, std2().rho);
    const Vec cc = trial_point(ranked, -std2().gamma);
    const Vec w2 = v0111::shrink_point(s0(), s2(), std2().sigma);
    const Vec w1 = v0111::shrink_point(s0(), s1(), std2().sigma);
    t.set(r, 4.0).set(cc, 3.5).set(w2, 2.0).set(w1, 2.0);
    const Traced run = run_table(t, s0(), 1.0, 8);
    expect_points(run.trace, {s0(), s1(), s2(), r, cc, w2, w1, next_reflection({s0(), w2, w1})});
}

TEST(V0111Paper, TheInitialSimplexKeepsConstructionOrderOnTies) {
    // FLOP's documented rule for the ordering the paper leaves to the user.
    {
        // s1 and s2 tie: s2 was built last and is reflected.
        Table t;
        t.set(s0(), 1.0).set(s1(), 2.0).set(s2(), 2.0);
        const Traced run = run_table(t, s0(), 1.0, 4);
        expect_points(run.trace, {s0(), s1(), s2(), next_reflection({s0(), s1(), s2()})});
    }
    {
        // All three tie: x0 is x_1, which a forced shrink shows.
        Table t;
        t.set(s0(), 1.0).set(s1(), 1.0).set(s2(), 1.0).set(xr(), 5.0).set(xcc(), 5.0);
        const Traced run = run_table(t, s0(), 1.0, 7);
        expect_points(run.trace,
                      {s0(), s1(), s2(), xr(), xcc(), v0111::shrink_point(s0(), s1(), std2().sigma),
                       v0111::shrink_point(s0(), s2(), std2().sigma)});
    }
}

// ============================================================================
// One dimension: x_n is x_1 (section 4.1, p. 124).
// ============================================================================

TEST(V0111Paper, InOneDimensionNoIterationTerminatesInStepTwo) {
    // p. 125: "a Nelder-Mead iteration can never terminate in step 2 ...
    // a reflection step will be taken only if f_r < f_1 and f_e >= f_r".
    // x_1 = 0 (f = 1), x_2 = 1 (f = 2), x_r = -1, x_e = -2, x_c = -1/2.
    const Simplex one{{0.0}, {1.0}};
    const Vec r = trial_point(one, std2().rho);
    const Vec e = trial_point(one, std2().rho * std2().chi);
    const Vec c = trial_point(one, std2().rho * std2().gamma);
    for (const double fr : {1.0, 1.5}) {
        // f_1 <= f_r < f_2: a reflection interval read with x_n = x_2 would
        // accept; the paper's (x_n = x_1, interval empty) contracts.
        Table t;
        t.set(one[0], 1.0).set(one[1], 2.0).set(r, fr).set(c, 0.5);
        const Traced run = run_table(t, one[0], 1.0, 4);
        expect_points(run.trace, {one[0], one[1], r, c});
    }
    {
        Table t;
        t.set(one[0], 1.0).set(one[1], 2.0).set(r, 0.5).set(e, 0.75);
        const Traced run = run_table(t, one[0], 1.0, 5);
        // x_r is kept through step 3: the next simplex is (x_r, x_1), and the
        // next reflection is of x_1 through x_r.
        expect_points(run.trace, {one[0], one[1], r, e, trial_point({r, one[0]}, std2().rho)});
    }
}

// ============================================================================
// What the papers prove, on FLOP's own runs.
// ============================================================================

namespace {

struct Audited {
    flop::Result r;
    v0111::Trace trace;
    v0111::ReferenceRun ref;
};

// Runs FLOP with a trace and replays it through the reference with every
// iteration recorded. The replay must account for every evaluation and end
// the same way. The final radius is the status suite's business.
template <class F>
Audited audited(F&& f, const Vec& x0, flop::nelder_mead::Options o) {
    Audited a;
    o.on_evaluation = v0111::recorder(a.trace);
    a.r = flop::nelder_mead::minimize(f, x0, o);
    o.on_evaluation = nullptr;
    const v0111::AuditReport rep = v0111::audit(x0, o, a.trace, a.r, false, &a.ref, true);
    EXPECT_TRUE(rep.points_agree) << rep.detail;
    EXPECT_TRUE(rep.same_length) << rep.detail;
    EXPECT_TRUE(rep.same_outcome) << rep.detail;
    return a;
}

double staircase(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        s += std::floor(4.0 * std::fabs(x[i] - v0111::sphere_centre(i) - 0.11));
    return s;
}

// A strictly convex quadratic with every pair of coordinates coupled.
double coupled(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double d = x[i] - v0111::sphere_centre(i);
        s += d * d;
        if (i + 1 < x.size()) {
            const double e = x[i] - x[i + 1];
            s += 10.0 * e * e;
        }
    }
    return s;
}

// The runs every property below is checked on: smooth and convex, smooth
// and not, and piecewise constant (ties and shrinks), across dimensions and
// both coefficient sets.
std::vector<Audited> property_runs() {
    std::vector<Audited> out;
    for (const std::size_t n :
         {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4}, std::size_t{8}}) {
        for (const bool adaptive : {false, true}) {
            flop::nelder_mead::Options o = v0111::options(1.0, 4000, adaptive);
            o.stopping.xtol_abs = 1e-9;
            out.push_back(audited(v0111::sphere, Vec(n, 0.0), o));
            out.push_back(audited(coupled, Vec(n, 0.0), o));
            flop::nelder_mead::Options s = v0111::options(1.0, 400, adaptive);
            out.push_back(audited(staircase, Vec(n, 0.0), s));
        }
    }
    flop::nelder_mead::Options o = v0111::options(0.5, 4000);
    o.stopping.xtol_abs = 1e-10;
    out.push_back(audited(v0111::extended_rosenbrock, Vec{-1.2, 1.0}, o));
    out.push_back(audited(v0111::beale, Vec{1.0, 1.0}, o));
    return out;
}

bool nonshrink(const v0111::Step& s) {
    return s.termination == Termination::Reflect || s.termination == Termination::Expand ||
           s.termination == Termination::Contract;
}

// k*, (2.8): the smallest index of a vertex that differs, 0-based.
std::size_t change_index(const v0111::Step& s) {
    for (std::size_t i = 0; i < s.before.size(); ++i)
        if (!v0111::same_bits(s.before[i].x, s.after[i].x)) return i;
    return s.before.size();
}

}  // namespace

TEST(V0111Paper, ANonshrinkStepHasTheStructureOf2_9) {
    // p. 117-118, (2.9): below k* nothing changes, at k* the value strictly
    // falls, above k* every vertex moves down one rank; so the value vector
    // strictly decreases lexicographically.
    std::size_t checked = 0;
    for (const Audited& a : property_runs()) {
        for (const v0111::Step& s : a.ref.steps) {
            if (!nonshrink(s)) continue;
            const std::size_t k = change_index(s);
            ASSERT_LT(k, s.before.size());
            for (std::size_t j = 0; j < k; ++j)
                EXPECT_TRUE(v0111::same_bits(s.after[j].f, s.before[j].f));
            EXPECT_LT(s.after[k].f, s.before[k].f);
            for (std::size_t j = k + 1; j < s.before.size(); ++j) {
                EXPECT_TRUE(v0111::same_bits(s.after[j].x, s.before[j - 1].x));
                EXPECT_TRUE(v0111::same_bits(s.after[j].f, s.before[j - 1].f));
            }
            std::vector<double> fb, fa;
            for (std::size_t j = 0; j < s.before.size(); ++j) {
                fb.push_back(s.before[j].f);
                fa.push_back(s.after[j].f);
            }
            EXPECT_TRUE(std::ranges::lexicographical_compare(fa, fb));
            ++checked;
        }
    }
    EXPECT_GT(checked, 1000u);
}

TEST(V0111Paper, TheChangeIndexIsWherePage117SaysItIs) {
    // p. 117: termination in step 2 gives 1 < k* <= n, step 3 gives k* = 1,
    // step 4 gives 1 <= k* <= n + 1, step 5 gives k* = 1 or 2 (1-based).
    std::size_t per_step[4] = {0, 0, 0, 0};
    for (const Audited& a : property_runs()) {
        for (const v0111::Step& s : a.ref.steps) {
            const std::size_t n = s.before.size() - 1;
            const std::size_t k = change_index(s) + 1;  // 1-based, as printed
            switch (s.termination) {
                case Termination::Reflect:
                    EXPECT_GT(k, 1u);
                    EXPECT_LE(k, n);
                    ++per_step[0];
                    break;
                case Termination::Expand:
                    EXPECT_EQ(k, 1u);
                    ++per_step[1];
                    break;
                case Termination::Contract:
                    EXPECT_GE(k, 1u);
                    EXPECT_LE(k, n + 1);
                    ++per_step[2];
                    break;
                case Termination::Shrink:
                    EXPECT_TRUE(k == 1 || k == 2) << k;
                    ++per_step[3];
                    break;
                case Termination::Restart:
                    break;
            }
        }
    }
    for (const std::size_t c : per_step) EXPECT_GT(c, 0u);
}

TEST(V0111Paper, TheWorstValueFallsWithinNPlusOneNonshrinkSteps) {
    // p. 118: "the worst function value must strictly decrease after at
    // most n + 1 consecutive nonshrink iterations".
    std::size_t windows = 0;
    for (const Audited& a : property_runs()) {
        const std::vector<v0111::Step>& st = a.ref.steps;
        for (std::size_t k = 0; k < st.size(); ++k) {
            const std::size_t n = st[k].before.size() - 1;
            if (k + n >= st.size()) break;
            bool all_nonshrink = true;
            for (std::size_t j = k; j <= k + n; ++j)
                all_nonshrink = all_nonshrink && nonshrink(st[j]);
            if (!all_nonshrink) continue;
            EXPECT_LT(st[k + n].after[n].f, st[k].before[n].f) << "steps " << k << " to " << k + n;
            ++windows;
        }
    }
    EXPECT_GT(windows, 1000u);
}

TEST(V0111Paper, NoVertexValueRisesAtANonshrinkStep) {
    // Lemma 3.3 (2), p. 121: f_i^(k+1) <= f_i^(k) for every i, strictly for
    // at least one.
    for (const Audited& a : property_runs()) {
        for (const v0111::Step& s : a.ref.steps) {
            if (!nonshrink(s)) continue;
            bool strict = false;
            for (std::size_t i = 0; i < s.before.size(); ++i) {
                EXPECT_LE(s.after[i].f, s.before[i].f);
                strict = strict || s.after[i].f < s.before[i].f;
            }
            EXPECT_TRUE(strict);
        }
    }
}

TEST(V0111Paper, AnIterationCostsWhatSection3_1Says) {
    // Section 3.1, property 1, p. 119: one evaluation when the iteration ends
    // in step 2, two in step 3 or 4, n + 2 with a shrink.
    for (const Audited& a : property_runs()) {
        for (const v0111::Step& s : a.ref.steps) {
            const std::size_t n = s.before.size() - 1;
            switch (s.termination) {
                case Termination::Reflect:
                    EXPECT_EQ(s.evaluations, 1u);
                    break;
                case Termination::Expand:
                case Termination::Contract:
                    EXPECT_EQ(s.evaluations, 2u);
                    break;
                case Termination::Shrink:
                    EXPECT_EQ(s.evaluations, n + 2);
                    break;
                case Termination::Restart:
                    break;
            }
        }
    }
}

namespace {

// |det M| for M = [x_1 - x_{n+1}, ..., x_n - x_{n+1}], (2.10)-(2.11) without
// the n!, by Gaussian elimination with partial pivoting in long double.
long double volume(const v0111::Ordered& s) {
    const std::size_t n = s.size() - 1;
    std::vector<std::vector<long double>> m(n, std::vector<long double>(n));
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i)
            m[i][j] = static_cast<long double>(s[j].x[i]) - static_cast<long double>(s[n].x[i]);
    long double det = 1.0L;
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t p = c;
        for (std::size_t r = c + 1; r < n; ++r)
            if (std::fabs(m[r][c]) > std::fabs(m[p][c])) p = r;
        if (m[p][c] == 0.0L) return 0.0L;
        std::swap(m[p], m[c]);
        det *= m[c][c];
        for (std::size_t r = c + 1; r < n; ++r) {
            const long double q = m[r][c] / m[c][c];
            for (std::size_t k = c; k < n; ++k) m[r][k] -= q * m[c][k];
        }
    }
    return std::fabs(det);
}

}  // namespace

TEST(V0111Paper, TheVolumeScalesByTauOrSigmaToTheN) {
    // Lemma 3.1, p. 120: vol(Delta_{k+1}) = |tau| vol(Delta_k) after a
    // nonshrink step of type tau, sigma^n vol(Delta_k) after a shrink.
    // Checked on the steps that are exact (all their points inside the exact
    // prefix of the trace), so the ratio is the geometry's, not rounding's.
    std::size_t checked = 0;
    for (const std::size_t n : {std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
        for (const bool adaptive : {false, true}) {
            flop::nelder_mead::Options o = v0111::options(1.0, 600, adaptive);
            for (auto* f : {&v0111::sphere, &staircase}) {
                const Audited a = audited(f, Vec(n, 0.0), o);
                const std::size_t exact = v0111::exact_prefix(a.trace);
                const double sigma = v0111::documented(n, adaptive).sigma;
                for (const v0111::Step& s : a.ref.steps) {
                    if (s.first_evaluation + s.evaluations > exact) break;
                    if (s.termination == Termination::Restart) continue;
                    const long double ratio = volume(s.after) / volume(s.before);
                    const long double want =
                        s.termination == Termination::Shrink
                            ? std::pow(static_cast<long double>(sigma), static_cast<long double>(n))
                            : std::fabs(static_cast<long double>(s.tau));
                    EXPECT_NEAR(static_cast<double>(ratio / want), 1.0, 1e-12) << "n " << n;
                    ++checked;
                }
            }
        }
    }
    EXPECT_GT(checked, 50u);
}

TEST(V0111Paper, InOneDimensionTheDiameterScalesByTau) {
    // (4.1), p. 124: diam(Delta_{k+1}) = |tau_k| diam(Delta_k), on the exact
    // steps; and no iteration terminates in step 2 over whole runs (p. 125).
    std::size_t checked = 0;
    for (const bool adaptive : {false, true}) {
        for (auto* f : {&v0111::sphere, &staircase, &coupled}) {
            const Audited a = audited(f, Vec{0.0}, v0111::options(1.0, 400, adaptive));
            const std::size_t exact = v0111::exact_prefix(a.trace);
            for (const v0111::Step& s : a.ref.steps) {
                EXPECT_NE(s.termination, Termination::Reflect);
                if (!nonshrink(s) || s.first_evaluation + s.evaluations > exact) continue;
                const double before = std::fabs(s.before[0].x[0] - s.before[1].x[0]);
                const double after = std::fabs(s.after[0].x[0] - s.after[1].x[0]);
                EXPECT_TRUE(v0111::same_bits(after, std::fabs(s.tau) * before));
                ++checked;
            }
        }
    }
    EXPECT_GT(checked, 20u);
}

TEST(V0111Paper, NoShrinkOnAStrictlyConvexFunction) {
    // Lemma 3.5, p. 122. Run to a radius far above the precision floor, so
    // the values the comparisons see are the function's, not rounding's.
    for (const std::size_t n :
         {std::size_t{2}, std::size_t{3}, std::size_t{5}, std::size_t{8}, std::size_t{16}}) {
        for (const bool adaptive : {false, true}) {
            flop::nelder_mead::Options o = v0111::options(1.0, 200000, adaptive);
            o.stopping.xtol_abs = 1e-7;
            for (auto* f : {&v0111::sphere, &coupled}) {
                const Audited a = audited(f, Vec(n, 0.0), o);
                EXPECT_EQ(a.ref.shrinks, 0u) << "n " << n;
                EXPECT_EQ(a.r.status, flop::Status::XtolReached) << "n " << n;
            }
        }
    }
}

TEST(V0111Paper, GaoHanSufficientDescent) {
    // Gao and Han, Theorem 2.1, (2.2): for n >= 2, a uniformly convex f and
    // the standard coefficients, an expansion or a contraction lowers
    // F = sum of the vertex values by at least (n - 1)/(2 n^2) rho(D/2), D
    // the diameter. For f = |x - c|^2, f(tx + (1 - t)y) = t f(x) + (1 - t) f(y)
    // - t(1 - t)|x - y|^2 exactly, so rho(s) = s^2 satisfies (2.1).
    std::size_t checked = 0;
    for (const std::size_t n :
         {std::size_t{2}, std::size_t{3}, std::size_t{4}, std::size_t{6}, std::size_t{8}}) {
        flop::nelder_mead::Options o = v0111::options(1.0, 100000, false);
        o.stopping.xtol_abs = 1e-6;
        const Audited a = audited(v0111::sphere, Vec(n, 0.0), o);
        const v0111::Coefficients c = v0111::standard();
        const auto dn = static_cast<double>(n);
        for (const v0111::Step& s : a.ref.steps) {
            const bool expansion = s.termination == Termination::Expand && s.tau == c.rho * c.chi;
            if (!expansion && s.termination != Termination::Contract) continue;
            double d = 0.0;
            for (const v0111::Vertex& p : s.before)
                for (const v0111::Vertex& q : s.before) {
                    double e = 0.0;
                    for (std::size_t i = 0; i < n; ++i) e += (p.x[i] - q.x[i]) * (p.x[i] - q.x[i]);
                    d = std::max(d, std::sqrt(e));
                }
            double fb = 0.0, fa = 0.0, mag = 0.0;
            for (std::size_t i = 0; i <= n; ++i) {
                fb += s.before[i].f;
                fa += s.after[i].f;
                mag += std::fabs(s.before[i].f);
            }
            const double bound = -(dn - 1.0) / (2.0 * dn * dn) * (0.5 * d) * (0.5 * d);
            const double rounding = 64.0 * std::numeric_limits<double>::epsilon() * mag;
            EXPECT_LE(fa - fb, bound + rounding) << "n " << n;
            ++checked;
        }
    }
    EXPECT_GT(checked, 100u);
}

TEST(V0111Paper, InOneDimensionTheMinimiserStaysWithinTheProximityBound) {
    // Theorem 4.1 and Lemma 4.3, pp. 125-126: in one dimension, on a strictly
    // convex f with bounded level sets and rho chi >= 1, the method converges
    // to the minimiser, and once it is bracketed it stays within
    // int(x_2, x_1 + N (x_1 - x_2)], N = max(1/(rho gamma), rho/gamma, rho chi,
    // chi - 1) (4.3). So at the stop |x_1 - x_min| <= N |x_1 - x_2|, and
    // |x_1 - x_2| is the radius.
    const v0111::Coefficients c = v0111::standard();
    const double big_n =
        std::max({1.0 / (c.rho * c.gamma), c.rho / c.gamma, c.rho * c.chi, c.chi - 1.0});
    const double centre = std::numbers::pi;
    auto quadratic = [&](std::span<const double> x) { return (x[0] - centre) * (x[0] - centre); };
    auto cosh_f = [&](std::span<const double> x) { return std::cosh(x[0] - centre); };
    auto quartic = [&](std::span<const double> x) {
        const double d = x[0] - centre;
        return d * d * d * d + d * d;
    };
    for (const double x0 : {-40.0, 0.0, 3.0, 25.5}) {
        for (const double h : {0.1, 1.0, 3.0}) {
            flop::nelder_mead::Options o = v0111::options(h, 100000);
            o.stopping.xtol_abs = 1e-7;
            const Vec from{x0};
            for (const flop::Result& r : {flop::nelder_mead::minimize(quadratic, from, o),
                                          flop::nelder_mead::minimize(cosh_f, from, o),
                                          flop::nelder_mead::minimize(quartic, from, o)}) {
                EXPECT_EQ(r.status, flop::Status::XtolReached);
                const double slack = 4.0 * std::numeric_limits<double>::epsilon() * centre;
                EXPECT_LE(std::fabs(r.x[0] - centre), big_n * r.final_radius + slack)
                    << "x0 " << x0 << " h " << h;
            }
        }
    }
}
