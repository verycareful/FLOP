// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.1.1: how a Nelder-Mead run ends.
//
// No paper fixes a stopping rule; FLOP's are documented
// (docs/algorithms/nelder-mead.md, "Stopping rules"): the radius r (the
// largest coordinate distance from a vertex to the best) against xtol_abs
// and xtol_rel * initial_step, then the spread f(x_{n+1}) - f(x_1) against
// ftol_abs and ftol_rel * |f(x_1)|, then r against the precision floor
// epsilon * max(initial_step, max |x_1|), all at the top of an iteration;
// stop_value after every evaluation; the cap before every evaluation. Each
// is pinned on a run built for it, and the reference transcription confirms
// the stop came at the same iteration with the same radius.
//
// Two defects are pinned red here, for 0.1.1.2:
//   - Result::final_radius after a cap part way through a shrink, or
//     through the first simplex inside a box narrower than initial_step,
//     describes vertices that were placed and never evaluated. It is the
//     radius of the last simplex whose every vertex was evaluated, and
//     initial_step while the first one is incomplete.
//   - The precision floor can sit below the smallest radius the arithmetic
//     reaches (a 4-d sphere near 1e6 stalls at two units in the last place
//     of 1e6, above a floor of 1.9; the 4-d staircase does the same near
//     1.1), so a run with nothing but a cap shrinks until the cap, and one
//     with only a tolerance never ends.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "flop/flop.hpp"
#include "v0111_problems.hpp"
#include "v0111_reference_nm.hpp"

namespace {

using Vec = std::vector<double>;
using v0111::Table;
using v0111::Traced;

double staircase(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        s += std::floor(4.0 * std::fabs(x[i] - v0111::sphere_centre(i) - 0.11));
    return s;
}

// Every field of the result against the reference replaying the trace.
void expect_reference_agrees(const Vec& x0, const flop::nelder_mead::Options& o,
                             const Traced& run) {
    const v0111::AuditReport rep = v0111::audit(x0, o, run.trace, run.r, false);
    EXPECT_TRUE(rep.points_agree) << rep.detail;
    EXPECT_TRUE(rep.same_length) << rep.detail;
    EXPECT_TRUE(rep.same_outcome) << rep.detail;
    EXPECT_TRUE(rep.same_radius) << rep.detail;
}

// The first of the smallest values in a trace prefix: the documented result.
const v0111::Point& best_of(const v0111::Trace& t, std::size_t count) {
    std::size_t b = 0;
    for (std::size_t k = 1; k < count; ++k)
        if (t[k].f < t[b].f) b = k;
    return t[b];
}

Vec s0() {
    return {0.0, 0.0};
}
Vec s1() {
    return {1.0, 0.0};
}
Vec s2() {
    return {0.0, 1.0};
}
v0111::Simplex start() {
    return {s0(), s1(), s2()};
}
v0111::Coefficients std2() {
    return v0111::standard();
}
Vec xr() {
    return v0111::trial_point(start(), std2().rho);
}
Vec xe() {
    return v0111::trial_point(start(), std2().rho * std2().chi);
}
Vec xc() {
    return v0111::trial_point(start(), std2().rho * std2().gamma);
}
Vec xcc() {
    return v0111::trial_point(start(), -std2().gamma);
}
Vec v1() {
    return v0111::shrink_point(s0(), s1(), std2().sigma);
}

Table start_table() {
    Table t;
    t.set(s0(), 1.0).set(s1(), 2.0).set(s2(), 3.0);
    return t;
}

}  // namespace

// ============================================================================
// Tolerances
// ============================================================================

TEST(V0111Status, XtolStopsAtTheRadiusOnEitherTolerance) {
    const Vec x0(3, 0.0);
    {
        flop::nelder_mead::Options o = v0111::options(1.0, 100000);
        o.stopping.xtol_abs = 1e-6;
        const Traced run = v0111::traced(v0111::sphere, x0, o);
        EXPECT_EQ(run.r.status, flop::Status::XtolReached);
        EXPECT_LE(run.r.final_radius, 1e-6);
        expect_reference_agrees(x0, o, run);
    }
    {
        // xtol_rel is relative to initial_step: 1e-6 * 0.5.
        flop::nelder_mead::Options o = v0111::options(0.5, 100000);
        o.stopping.xtol_rel = 1e-6;
        const Traced run = v0111::traced(v0111::sphere, x0, o);
        EXPECT_EQ(run.r.status, flop::Status::XtolReached);
        EXPECT_LE(run.r.final_radius, 0.5e-6);
        expect_reference_agrees(x0, o, run);
    }
    {
        // Both set: whichever is reached first, so the larger.
        flop::nelder_mead::Options o = v0111::options(0.5, 100000);
        o.stopping.xtol_rel = 1e-9;
        o.stopping.xtol_abs = 1e-5;
        const Traced run = v0111::traced(v0111::sphere, x0, o);
        EXPECT_EQ(run.r.status, flop::Status::XtolReached);
        EXPECT_LE(run.r.final_radius, 1e-5);
        EXPECT_GT(run.r.final_radius, 0.5e-9);
        expect_reference_agrees(x0, o, run);
    }
}

TEST(V0111Status, FtolStopsOnTheSpreadAbsoluteOrRelative) {
    const Vec x0(3, 0.0);
    {
        flop::nelder_mead::Options o = v0111::options(1.0, 100000);
        o.stopping.ftol_abs = 1e-10;
        const Traced run = v0111::traced(v0111::sphere, x0, o);
        EXPECT_EQ(run.r.status, flop::Status::FtolReached);
        expect_reference_agrees(x0, o, run);
    }
    {
        // A negative best value: the relative tolerance acts on |f_1|.
        auto shifted = [](std::span<const double> x) { return v0111::sphere(x) - 5.0; };
        flop::nelder_mead::Options o = v0111::options(1.0, 100000);
        o.stopping.ftol_rel = 1e-12;
        const Traced run = v0111::traced(shifted, x0, o);
        EXPECT_EQ(run.r.status, flop::Status::FtolReached);
        EXPECT_LT(run.r.f, -5.0 + 1e-10);
        expect_reference_agrees(x0, o, run);
    }
}

TEST(V0111Status, ARelativeFtolAtABestValueOfZeroNeedsAZeroSpread) {
    // f = max(0, |x| - 1) in one dimension from 0 with h = 4: f_1 = 0 from
    // the start, so ftol_rel * |f_1| = 0 and only a spread of exactly zero
    // stops. Simplex {0, 4} (spread 3), then two inside contractions:
    // {0, 2} (spread 1), {0, 1} (spread 0), which stops.
    auto f = [](std::span<const double> x) { return std::max(0.0, std::fabs(x[0]) - 1.0); };
    flop::nelder_mead::Options o = v0111::options(4.0, 100);
    o.stopping.ftol_rel = 0.5;
    const Traced run = v0111::traced(f, Vec{0.0}, o);
    EXPECT_EQ(run.r.status, flop::Status::FtolReached);
    const v0111::Simplex one{{0.0}, {4.0}};
    const Vec r1 = v0111::trial_point(one, std2().rho);
    const Vec c1 = v0111::trial_point(one, -std2().gamma);
    const v0111::Simplex two{{0.0}, c1};
    const Vec r2 = v0111::trial_point(two, std2().rho);
    const Vec c2 = v0111::trial_point(two, -std2().gamma);
    ASSERT_EQ(run.trace.size(), 6u);
    const std::vector<Vec> want{one[0], one[1], r1, c1, r2, c2};
    for (std::size_t k = 0; k < want.size(); ++k)
        EXPECT_TRUE(v0111::same_bits(run.trace[k].x, want[k])) << k;
    EXPECT_TRUE(v0111::same_bits(run.r.x, Vec{0.0}));  // the first of the two zeros
    EXPECT_EQ(run.r.final_radius, 1.0);
    expect_reference_agrees(Vec{0.0}, o, run);
}

TEST(V0111Status, XtolIsTestedBeforeFtol) {
    // Both hold at the first stopping test: the initial simplex has radius
    // 1 <= 10 and spread below 1e9.
    flop::nelder_mead::Options o = v0111::options(1.0, 100);
    o.stopping.xtol_abs = 10.0;
    o.stopping.ftol_abs = 1e9;
    const Traced run = v0111::traced(v0111::sphere, Vec(2, 0.0), o);
    EXPECT_EQ(run.r.status, flop::Status::XtolReached);
    EXPECT_EQ(run.r.evaluations, 3u);
}

// ============================================================================
// stop_value, at every kind of evaluation
// ============================================================================

TEST(V0111Status, StopValueEndsTheRunAtTheEvaluationThatMeetsIt) {
    struct Case {
        const char* where;
        Table table;
        Vec point;
        std::size_t index;
    };
    std::vector<Case> cases;
    {
        Table t;
        t.set(s0(), 0.5);
        cases.push_back({.where = "x0", .table = t, .point = s0(), .index = 0});
    }
    {
        Table t;
        t.set(s0(), 1.0).set(s1(), 2.0).set(s2(), 0.25);
        cases.push_back(
            {.where = "the last initial vertex", .table = t, .point = s2(), .index = 2});
    }
    {
        Table t = start_table();
        t.set(xr(), 0.25);
        cases.push_back({.where = "a reflection", .table = t, .point = xr(), .index = 3});
    }
    {
        Table t = start_table();
        t.set(xr(), 0.75).set(xe(), 0.25);
        cases.push_back({.where = "an expansion", .table = t, .point = xe(), .index = 4});
    }
    {
        Table t = start_table();
        t.set(xr(), 2.5).set(xc(), 0.25);
        cases.push_back({.where = "an outside contraction", .table = t, .point = xc(), .index = 4});
    }
    {
        Table t = start_table();
        t.set(xr(), 4.0).set(xcc(), 0.25);
        cases.push_back({.where = "an inside contraction", .table = t, .point = xcc(), .index = 4});
    }
    {
        Table t = start_table();
        t.set(xr(), 4.0).set(xcc(), 3.5).set(v1(), 0.25);
        cases.push_back({.where = "a shrink", .table = t, .point = v1(), .index = 5});
    }
    for (Case& c : cases) {
        flop::nelder_mead::Options o = v0111::options(1.0, 100);
        o.stopping.stop_value = 0.5;
        const Traced run = v0111::traced(c.table, s0(), o);
        EXPECT_EQ(run.r.status, flop::Status::StopValueReached) << c.where;
        EXPECT_EQ(run.r.evaluations, c.index + 1) << c.where;
        EXPECT_EQ(run.trace.size(), c.index + 1) << c.where;
        EXPECT_TRUE(v0111::same_bits(run.r.x, c.point)) << c.where;
        EXPECT_TRUE(flop::converged(run.r.status));
    }
}

// ============================================================================
// The cap
// ============================================================================

TEST(V0111Status, EveryCapGivesAPrefixOfTheUncappedRun) {
    // Determinism and the cap together: a run capped at k evaluates exactly
    // the first k points of the uncapped run, never k + 1, and returns the
    // best of them, with the radius of the last simplex it evaluated whole.
    // The staircase shrinks constantly, so the cap lands inside shrinks too,
    // where that is the simplex before the shrink.
    for (auto* f : {&v0111::sphere, &staircase}) {
        const Vec x0(3, 0.0);
        flop::nelder_mead::Options base = v0111::options(1.0, 100000);
        base.stopping.xtol_abs = 1e-6;
        const Traced full = v0111::traced(f, x0, base);
        ASSERT_TRUE(flop::converged(full.r.status) ||
                    full.r.status == flop::Status::RoundoffLimited);
        const std::size_t total = full.trace.size();
        ASSERT_GT(total, 50u);
        for (std::size_t cap = 1; cap <= total + 1; ++cap) {
            flop::nelder_mead::Options o = base;
            o.stopping.max_evaluations = cap;
            const Traced run = v0111::traced(f, x0, o);
            const std::size_t want = std::min(cap, total);
            ASSERT_EQ(run.trace.size(), want) << "cap " << cap;
            EXPECT_EQ(run.r.evaluations, want) << "cap " << cap;
            for (std::size_t k = 0; k < want; ++k)
                ASSERT_TRUE(v0111::same_bits(run.trace[k].x, full.trace[k].x)) << "cap " << cap;
            EXPECT_EQ(run.r.status,
                      cap < total ? flop::Status::MaxEvaluationsReached : full.r.status)
                << "cap " << cap;
            const v0111::Point& b = best_of(run.trace, want);
            EXPECT_TRUE(v0111::same_bits(run.r.x, b.x)) << "cap " << cap;
            EXPECT_TRUE(v0111::same_bits(run.r.f, b.f)) << "cap " << cap;
            const v0111::AuditReport rep = v0111::audit(x0, o, run.trace, run.r, false);
            EXPECT_TRUE(rep.same_outcome) << "cap " << cap << rep.detail;
            EXPECT_TRUE(rep.same_radius) << "cap " << cap << rep.detail;
        }
    }
}

TEST(V0111Status, TheCapIsExactInsideEveryKindOfStep) {
    // The cap lands on the second point of an expansion, of either
    // contraction, and inside a shrink: evaluations == cap, the refused point
    // is not evaluated, the status is the cap's.
    struct Case {
        const char* where;
        Table table;
        std::size_t cap;
    };
    std::vector<Case> cases;
    {
        Table t = start_table();
        t.set(xr(), 0.5);
        cases.push_back({.where = "before the expansion point", .table = t, .cap = 4});
    }
    {
        Table t = start_table();
        t.set(xr(), 2.5);
        cases.push_back({.where = "before the outside contraction point", .table = t, .cap = 4});
    }
    {
        Table t = start_table();
        t.set(xr(), 4.0);
        cases.push_back({.where = "before the inside contraction point", .table = t, .cap = 4});
    }
    {
        Table t = start_table();
        t.set(xr(), 4.0).set(xcc(), 3.5);
        cases.push_back({.where = "before the first shrink point", .table = t, .cap = 5});
    }
    {
        Table t = start_table();
        t.set(xr(), 4.0).set(xcc(), 3.5).set(v1(), 2.5);
        cases.push_back({.where = "between the shrink points", .table = t, .cap = 6});
    }
    for (const std::size_t cap : {std::size_t{1}, std::size_t{2}, std::size_t{3}})
        cases.push_back({.where = "inside or at the end of the initial simplex",
                         .table = start_table(),
                         .cap = cap});
    for (Case& c : cases) {
        const Traced run = v0111::traced(c.table, s0(), v0111::options(1.0, c.cap));
        EXPECT_EQ(run.r.status, flop::Status::MaxEvaluationsReached) << c.where;
        EXPECT_EQ(run.r.evaluations, c.cap) << c.where;
        EXPECT_EQ(run.trace.size(), c.cap) << c.where;
        EXPECT_FALSE(flop::converged(run.r.status));
    }
}

TEST(V0111Status, FinalRadiusOnACapInsideAShrinkIsTheLastEvaluatedSimplex) {
    // The shrink moves s1 and s2 halfway to s0; the cap admits the first new
    // point and refuses the second. The last
    // simplex whose every vertex was evaluated is {s0, s1, s2}, radius 1.
    Table t = start_table();
    t.set(xr(), 4.0).set(xcc(), 3.5).set(v1(), 2.5);
    const Traced run = v0111::traced(t, s0(), v0111::options(1.0, 6));
    ASSERT_EQ(run.r.status, flop::Status::MaxEvaluationsReached);
    EXPECT_EQ(run.r.final_radius, 1.0);
}

TEST(V0111Status, FinalRadiusWhileTheFirstSimplexIsIncompleteIsTheInitialStep) {
    {
        for (const std::size_t cap : {std::size_t{1}, std::size_t{2}}) {
            const Traced run = v0111::traced(v0111::sphere, Vec(3, 0.0), v0111::options(0.75, cap));
            EXPECT_EQ(run.r.final_radius, 0.75) << "cap " << cap;
        }
    }
    {
        // Inside a box narrower than initial_step the placed vertices sit
        // 0.25 away, but none of them was evaluated.
        flop::nelder_mead::Options o = v0111::options(1.0, 1);
        const Vec lo(2, 0.0), hi(2, 0.25);
        o.bounds = flop::Bounds::box(lo, hi);
        const Traced run = v0111::traced(v0111::sphere, Vec(2, 0.0), o);
        EXPECT_EQ(run.r.final_radius, 1.0);
    }
}

// ============================================================================
// The precision floor
// ============================================================================

TEST(V0111Status, WithOnlyACapTheRunEndsAtThePrecisionOfTheArithmetic) {
    // The floor epsilon * max(initial_step, max |x_1|), at three scales:
    // around 1, around 1e6 (the floor follows the coordinates), and around
    // 1e-6 with a step to match (the floor follows the step). A run that
    // cannot reach the floor ends at a shrink that moves no vertex instead:
    // a coordinate stays put only when sigma times its distance from the
    // best vertex rounds back to that distance, which needs the distance to
    // be at most 1 / (2 (1 - sigma)) units in the last place, n / 2 for Gao
    // and Han's sigma = 1 - 1/n and 1 for the standard 1/2. A unit in the last
    // place is at most epsilon times the coordinate, and one more unit covers
    // the rounding of the shrunk point, so the radius is at most
    // (max(1, n / 2) + 1) epsilon * scale either way.
    struct Case {
        double offset;
        double scale;
        double step;
        std::size_t n;
    };
    for (const Case c : {Case{.offset = 0.0, .scale = 1.0, .step = 1.0, .n = 2},
                         Case{.offset = 0.0, .scale = 1.0, .step = 1.0, .n = 4},
                         Case{.offset = 1e6, .scale = 1.0, .step = 1.0, .n = 2},
                         Case{.offset = 0.0, .scale = 1e-6, .step = 1e-6, .n = 2},
                         Case{.offset = 0.0, .scale = 1e-6, .step = 1e-6, .n = 4}}) {
        auto f = [&](std::span<const double> x) {
            double s = 0.0;
            for (std::size_t i = 0; i < x.size(); ++i) {
                const double d = x[i] - c.offset - c.scale * v0111::sphere_centre(i);
                s += d * d;
            }
            return s;
        };
        const Vec x0(c.n, c.offset);
        const flop::nelder_mead::Options o = v0111::options(c.step, 100000);
        const Traced run = v0111::traced(f, x0, o);
        EXPECT_EQ(run.r.status, flop::Status::RoundoffLimited) << c.offset << " " << c.n;
        double scale = c.step;
        for (const double v : run.r.x) scale = std::max(scale, std::fabs(v));
        const double units = std::max(1.0, 0.5 * static_cast<double>(c.n)) + 1.0;
        EXPECT_LE(run.r.final_radius, units * std::numeric_limits<double>::epsilon() * scale)
            << c.offset << " " << c.n;
        EXPECT_FALSE(flop::converged(run.r.status));
        expect_reference_agrees(x0, o, run);
    }
}

TEST(V0111Status, ARunBelowTheFloorsReachEndsAtAShrinkThatMovesNothing) {
    // The floor epsilon * max(h, max |x_1|) lies between one and two units in
    // the last place of the largest coordinate, and a simplex can stall at
    // two: a shrink by sigma = 3/4 (Gao and Han at n = 4) of a vertex two
    // units away rounds back onto it. That shrink would repeat forever; it
    // is a RoundoffLimited verdict instead, well before the cap. Around 1e6
    // (floor 1.9 units) and around 1.1 on the 4-d staircase (floor 1.12
    // units), both with nothing but a cap.
    const double offset = 1e6;
    auto far = [&](std::span<const double> x) {
        double s = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double d = x[i] - offset - v0111::sphere_centre(i);
            s += d * d;
        }
        return s;
    };
    const Traced a = v0111::traced(far, Vec(4, offset), v0111::options(1.0, 100000));
    EXPECT_EQ(a.r.status, flop::Status::RoundoffLimited);
    EXPECT_LT(a.r.evaluations, 100000u);
    const Traced b = v0111::traced(staircase, Vec(4, 0.0), v0111::options(1.0, 100000));
    EXPECT_EQ(b.r.status, flop::Status::RoundoffLimited);
    EXPECT_LT(b.r.evaluations, 100000u);
}

// ============================================================================
// What every result carries
// ============================================================================

TEST(V0111Status, EveryResultIsTheBestPointEvaluatedWithItsOwnValue) {
    // Across every status: converged() is true exactly for the three
    // converged ones, the result is the first evaluated point of least value
    // and its value is the one that evaluation returned, and the violation
    // is zero to the bit.
    std::vector<Traced> runs;
    const Vec x0(3, 0.0);
    flop::nelder_mead::Options o = v0111::options(1.0, 100000);
    o.stopping.xtol_abs = 1e-6;
    runs.push_back(v0111::traced(v0111::sphere, x0, o));
    o = v0111::options(1.0, 100000);
    o.stopping.ftol_abs = 1e-9;
    runs.push_back(v0111::traced(v0111::sphere, x0, o));
    o = v0111::options(1.0, 100000);
    o.stopping.stop_value = 0.1;
    runs.push_back(v0111::traced(v0111::sphere, x0, o));
    runs.push_back(v0111::traced(v0111::sphere, x0, v0111::options(1.0, 37)));
    runs.push_back(v0111::traced(v0111::sphere, x0, v0111::options(1.0, 100000)));
    runs.push_back(v0111::traced(staircase, x0, v0111::options(1.0, 500)));
    bool seen[5] = {false, false, false, false, false};
    for (const Traced& run : runs) {
        seen[static_cast<std::size_t>(run.r.status)] = true;
        const bool c = run.r.status == flop::Status::XtolReached ||
                       run.r.status == flop::Status::FtolReached ||
                       run.r.status == flop::Status::StopValueReached;
        EXPECT_EQ(flop::converged(run.r.status), c);
        const v0111::Point& b = best_of(run.trace, run.trace.size());
        EXPECT_TRUE(v0111::same_bits(run.r.x, b.x));
        EXPECT_TRUE(v0111::same_bits(run.r.f, b.f));
        EXPECT_TRUE(v0111::same_bits(run.r.max_constraint_violation, 0.0));
        EXPECT_EQ(run.r.evaluations, run.trace.size());
        for (std::size_t k = 0; k < run.trace.size(); ++k) EXPECT_EQ(run.trace[k].index, k);
    }
    for (const bool s : seen) EXPECT_TRUE(s);
}
