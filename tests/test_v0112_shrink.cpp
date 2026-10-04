// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.1.2: the shrink in rounded arithmetic, and the radius it leaves.
//
// A shrink moves every vertex but the best to x_1 + sigma (x_i - x_1)
// (Lagarias et al., section 2, step 5). Rounded, a vertex a few units in the
// last place from x_1 can land back on itself. FLOP evaluates only the
// vertices that moved, an unmoved one keeping its value, and a shrink that
// moves none is the precision of the arithmetic reached: RoundoffLimited,
// with no evaluation. Before this, such a shrink repeated until the cap.
//
// A panel of runs taken to the precision of the arithmetic (the sphere
// centred near 1e6 and the staircase, n = 4, 5, 6, 8, Gao and Han's
// coefficients) supplies shrinks of each kind. For n >= 4 a vertex stalls
// up to about n / 2 units in the last place from x_1, above the floor's one
// or two, so such runs end at a shrink that moves nothing. Which run shows
// which event depends on rounding, and -ffast-math rounds differently, so
// each test checks its property on every run where the event occurs and
// requires that it occurs somewhere in the panel. The reference
// transcription replays every run and must agree point by point.
//
// Result::final_radius is the radius of the last simplex whose every vertex
// was evaluated. A batch shrink that meets stop_value has evaluated all its
// points, so its simplex is complete; the scalar channel stops at the point
// that met it, inside the shrink.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "flop/flop.hpp"
#include "v0111_problems.hpp"
#include "v0111_reference_nm.hpp"

namespace {

using Vec = std::vector<double>;
using v0111::Table;
using v0111::Traced;

constexpr double kOffset = 1e6;

double far_sphere(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double d = x[i] - kOffset - v0111::sphere_centre(i);
        s += d * d;
    }
    return s;
}

double staircase(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        s += std::floor(4.0 * std::fabs(x[i] - v0111::sphere_centre(i) - 0.11));
    return s;
}

struct PanelRun {
    double (*f)(std::span<const double>);
    Vec x0;
    Traced run;
    v0111::ReferenceRun ref;
    v0111::AuditReport rep;
};

std::vector<PanelRun> panel() {
    std::vector<PanelRun> out;
    for (const std::size_t n : {std::size_t{4}, std::size_t{5}, std::size_t{6}, std::size_t{8}}) {
        for (auto* f : {&far_sphere, &staircase}) {
            const Vec x0(n, f == &far_sphere ? kOffset : 0.0);
            const flop::nelder_mead::Options o = v0111::options(1.0, 100000, true);
            PanelRun r{.f = f, .x0 = x0, .run = v0111::traced(f, x0, o), .ref = {}, .rep = {}};
            r.rep = v0111::audit(x0, o, r.run.trace, r.run.r, false, &r.ref, true);
            out.push_back(std::move(r));
        }
    }
    return out;
}

// Evaluations the reference accounts for: the initial simplex and every
// completed iteration.
std::size_t counted(const PanelRun& r) {
    std::size_t c = r.x0.size() + 1;
    for (const v0111::Step& s : r.ref.steps) c += s.evaluations;
    return c;
}

}  // namespace

TEST(V0112Shrink, AShrinkThatMovesNoVertexEndsTheRunWithoutAnEvaluation) {
    std::size_t still = 0;
    for (const PanelRun& r : panel()) {
        const std::size_t n = r.x0.size();
        EXPECT_TRUE(r.rep.points_agree && r.rep.same_length && r.rep.same_outcome &&
                    r.rep.same_radius)
            << "n " << n << r.rep.detail;
        EXPECT_LT(r.run.r.evaluations, 100000u) << "n " << n;
        EXPECT_LE(r.ref.still_shrinks, 1u) << "n " << n;
        if (r.ref.still_shrinks == 0) continue;
        ++still;
        // It is the run's last act: after the last completed iteration come
        // only that iteration's reflection and contraction, and nothing is
        // evaluated for the shrink.
        EXPECT_EQ(r.run.r.status, flop::Status::RoundoffLimited) << "n " << n;
        EXPECT_EQ(r.run.r.evaluations, counted(r) + 2) << "n " << n;
    }
    EXPECT_GE(still, 1u) << "no run in the panel ended at a shrink that moved nothing";
}

TEST(V0112Shrink, AVertexTheShrinkLeavesInPlaceKeepsItsValue) {
    std::size_t partial = 0;
    for (const PanelRun& r : panel()) {
        const std::size_t n = r.x0.size();
        ASSERT_TRUE(r.rep.points_agree) << "n " << n << r.rep.detail;
        for (const v0111::Step& s : r.ref.steps) {
            if (s.termination != v0111::Termination::Shrink) continue;
            const std::size_t moved = s.evaluations - 2;  // after the reflection and contraction
            if (moved == n) continue;
            ++partial;
            // Every vertex after the shrink with the bits of one before it
            // carries that vertex's value. Those include x_1 and the n - moved
            // that stayed, and can include a moved vertex that landed exactly
            // where another one was, a unit or two away at this precision; its
            // value was evaluated afresh, and equals the old one because the
            // objective is a function of the point.
            std::size_t kept = 0;
            for (const v0111::Vertex& a : s.after) {
                for (const v0111::Vertex& b : s.before) {
                    if (!v0111::same_bits(a.x, b.x)) continue;
                    EXPECT_TRUE(v0111::same_bits(a.f, b.f)) << "n " << n;
                    ++kept;
                    break;
                }
            }
            EXPECT_GE(kept, 1 + (n - moved)) << "n " << n;
        }
    }
    EXPECT_GE(partial, 1u) << "no run in the panel had a partial shrink";
}

TEST(V0112Shrink, AShrinkIsOneCallOfTheVerticesItMoved) {
    std::size_t partial = 0;
    for (const PanelRun& r : panel()) {
        const std::size_t n = r.x0.size();
        ASSERT_TRUE(r.rep.points_agree) << "n " << n << r.rep.detail;
        auto g = v0111::batched(r.f);
        const Traced batch = v0111::traced_batch(g, r.x0, v0111::options(1.0, 100000, true));
        ASSERT_EQ(batch.trace.size(), r.run.trace.size()) << "n " << n;
        for (std::size_t k = 0; k < batch.trace.size(); ++k)
            ASSERT_TRUE(v0111::same_bits(batch.trace[k].x, r.run.trace[k].x))
                << "n " << n << ", " << k;
        std::vector<std::size_t> want{n + 1};
        for (const v0111::Step& s : r.ref.steps) {
            if (s.termination == v0111::Termination::Shrink) {
                want.insert(want.end(), {1, 1, s.evaluations - 2});
                if (s.evaluations - 2 < n) ++partial;
            } else {
                want.insert(want.end(), s.evaluations, 1);
            }
        }
        want.insert(want.end(), r.run.trace.size() - counted(r), 1);
        EXPECT_EQ(g.sizes, want) << "n " << n;
    }
    EXPECT_GE(partial, 1u) << "no run in the panel had a partial shrink";
}

namespace {

// n = 2 from s0 = (0, 0), h = 1: s1 = (1, 0) at 2, s2 = (0, 1) at 3. The
// reflection of s2 through (0.5, 0) is (1, -1), at 4, no better than s2, so
// the inside contraction (0.25, 0.5), at 3.5, also no better: a shrink
// toward s0, of s1 to (0.5, 0) and of s2 to (0, 0.5), in that order. The
// simplex before it has radius 1.
Table shrink_table(double f_first, double f_second) {
    Table t;
    t.set({0.0, 0.0}, 1.0).set({1.0, 0.0}, 2.0).set({0.0, 1.0}, 3.0);
    t.set({1.0, -1.0}, 4.0).set({0.25, 0.5}, 3.5);
    t.set({0.5, 0.0}, f_first).set({0.0, 0.5}, f_second);
    return t;
}

}  // namespace

TEST(V0112Shrink, FinalRadiusAfterAShrinkCutByStopValueDependsOnWhetherItCompleted) {
    flop::nelder_mead::Options o = v0111::options(1.0, 100);
    o.stopping.stop_value = 0.5;
    {
        // Scalar: (0.5, 0) meets stop_value and (0, 0.5) is never evaluated;
        // the last complete simplex is the one before the shrink, radius 1.
        Table t = shrink_table(0.1, 2.5);
        const Traced run = v0111::traced(t, Vec{0.0, 0.0}, o);
        EXPECT_EQ(run.r.status, flop::Status::StopValueReached);
        EXPECT_EQ(run.r.evaluations, 6u);
        EXPECT_EQ(run.r.final_radius, 1.0);
    }
    {
        // Batch: the shrink is one call of both points, so the shrunk simplex
        // is complete; its best is (0.5, 0), and both other vertices are 0.5
        // from it in the largest coordinate.
        Table t = shrink_table(0.1, 2.5);
        auto f = v0111::batched([&t](std::span<const double> x) { return t(x); });
        const Traced run = v0111::traced_batch(f, Vec{0.0, 0.0}, o);
        EXPECT_EQ(run.r.status, flop::Status::StopValueReached);
        EXPECT_EQ(run.r.evaluations, 7u);
        EXPECT_EQ(f.sizes, (std::vector<std::size_t>{3, 1, 1, 2}));
        EXPECT_EQ(run.r.final_radius, 0.5);
    }
    {
        // Scalar, with stop_value met at the shrink's last point: every point
        // was evaluated, so the shrunk simplex is complete here too.
        Table t = shrink_table(2.5, 0.1);
        const Traced run = v0111::traced(t, Vec{0.0, 0.0}, o);
        EXPECT_EQ(run.r.evaluations, 7u);
        EXPECT_EQ(run.r.final_radius, 0.5);
    }
}
