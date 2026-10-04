// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.1.2: the poll ladder on a box.
//
// Projection onto the box can flatten the simplex against a face or
// collapse it onto a corner, and a stop there need not be near a minimiser.
// Once the box has acted on the run (a projected trial point, or an initial
// vertex the box moved off x + h), every stop verdict is preceded by the
// ladder: the best vertex +- d e_i clipped to the box, for d = initial_step,
// d / 2, ... down to the stop's scale (the larger x tolerance for
// XtolReached, else the radius or the precision floor), each rung evaluated
// whole, and a restart of size d from the first point of least value when
// it is better. The bounds suite pins the rungs on scripted runs; this one
// pins when the ladder runs, its bottom, and the two channels through it.

#include <gtest/gtest.h>

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

void expect_points(const v0111::Trace& t, const std::vector<Vec>& want) {
    ASSERT_EQ(t.size(), want.size());
    for (std::size_t k = 0; k < want.size(); ++k)
        EXPECT_TRUE(v0111::same_bits(t[k].x, want[k])) << "evaluation " << k;
}

// The corner scenario of the bounds suite: n = 2 on [0, 1]^2 from (0, 0)
// with h = 1, driven onto the corner in five projected steps; a ladder rung
// at d is then (d, 0), (0, d). Every rung point not listed is worse.
Table corner_table() {
    Table t;
    t.set({0.0, 0.0}, 1.0).set({1.0, 0.0}, 3.0).set({0.0, 1.0}, 2.0).set({0.0, 0.75}, 1.5);
    t.set({0.5, 0.0}, 5.0).set({0.0, 0.5}, 5.0);
    return t;
}

std::vector<Vec> corner_path() {
    return {{0.0, 0.0},  {1.0, 0.0}, {0.0, 1.0}, {0.0, 1.0},
            {0.0, 0.75}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
}

flop::nelder_mead::Options corner_options() {
    flop::nelder_mead::Options o = v0111::options(1.0, 100);
    o.bounds = flop::Bounds::box(Vec{0.0, 0.0}, Vec{1.0, 1.0});
    return o;
}

}  // namespace

TEST(V0112Ladder, AnInitialVertexMovedByTheBoxIsEnoughForTheLadder) {
    // f on [0, 1] from x0 = 1, the upper bound, h = 0.5: x0 + h is outside,
    // so the vertex is 0.5 and the box has acted, though no trial point is
    // ever projected. Values 1 at 1 and 0.5 at 0.5; the reflection 0, at 2,
    // is worse than both, the inside contraction 0.75, at 0.75, better than
    // the worst: the simplex {0.5, 0.75}, radius 0.25 = xtol_abs, stops.
    // The ladder then polls 0.5 +- 0.5 (1 and 0) and 0.5 +- 0.25 (0.75 and
    // 0.25), all worse, and the stop stands.
    Table t;
    t.set({1.0}, 1.0).set({0.5}, 0.5).set({0.0}, 2.0).set({0.75}, 0.75).set({0.25}, 5.0);
    flop::nelder_mead::Options o = v0111::options(0.5, 100);
    o.bounds = flop::Bounds::box(Vec{0.0}, Vec{1.0});
    o.stopping.xtol_abs = 0.25;
    const Traced run = v0111::traced(t, Vec{1.0}, o);
    expect_points(run.trace, {{1.0}, {0.5}, {0.0}, {0.75}, {1.0}, {0.0}, {0.75}, {0.25}});
    EXPECT_EQ(run.r.status, flop::Status::XtolReached);
    EXPECT_TRUE(v0111::same_bits(run.r.x, Vec{0.5}));
    EXPECT_EQ(t.misses(), 0u);
}

TEST(V0112Ladder, TheBottomRungIsTheLargerOfTheTwoXTolerances) {
    // The corner scenario stops at radius 0, so every x tolerance holds; the
    // ladder's bottom is the larger of the two set.
    struct Case {
        double xtol_abs;
        double xtol_rel;
        std::vector<Vec> rungs;
    };
    const std::vector<Case> cases{
        {.xtol_abs = 0.5,
         .xtol_rel = 0.0,
         .rungs = {{1.0, 0.0}, {0.0, 1.0}, {0.5, 0.0}, {0.0, 0.5}}},
        // xtol_rel * initial_step = 0.5 is the larger: two rungs.
        {.xtol_abs = 0.25,
         .xtol_rel = 0.5,
         .rungs = {{1.0, 0.0}, {0.0, 1.0}, {0.5, 0.0}, {0.0, 0.5}}},
        {.xtol_abs = 0.5,
         .xtol_rel = 0.25,
         .rungs = {{1.0, 0.0}, {0.0, 1.0}, {0.5, 0.0}, {0.0, 0.5}}},
        // 0.25 alone: a third rung, (0.25, 0) and (0, 0.25), not in the table.
        {.xtol_abs = 0.25,
         .xtol_rel = 0.0,
         .rungs = {{1.0, 0.0}, {0.0, 1.0}, {0.5, 0.0}, {0.0, 0.5}, {0.25, 0.0}, {0.0, 0.25}}},
    };
    for (const Case& c : cases) {
        flop::nelder_mead::Options o = corner_options();
        o.stopping.xtol_abs = c.xtol_abs;
        o.stopping.xtol_rel = c.xtol_rel;
        Table t = corner_table();
        const Traced run = v0111::traced(t, Vec{0.0, 0.0}, o);
        std::vector<Vec> want = corner_path();
        want.insert(want.end(), c.rungs.begin(), c.rungs.end());
        expect_points(run.trace, want);
        EXPECT_EQ(run.r.status, flop::Status::XtolReached);
    }
}

TEST(V0112Ladder, BothChannelsEvaluateTheSamePointsThroughRungsAndRestarts) {
    // Convex problems with their minimum off the faces, from a corner: the
    // simplex is flattened, the ladder restarts the method, and the batch
    // channel sends each rung and each restart's simplex as one call. The
    // scalar run must agree with the reference point by point.
    struct Case {
        Vec centre;
        Vec x0;
    };
    for (const Case& c :
         {Case{.centre = {0.25}, .x0 = {0.0}}, Case{.centre = {0.125, 0.25}, .x0 = {0.0, 0.0}},
          Case{.centre = {0.125, 0.25, 0.375}, .x0 = {0.0, 0.0, 0.0}}}) {
        const std::size_t n = c.x0.size();
        auto f = [&c](std::span<const double> x) {
            double s = 0.0;
            for (std::size_t i = 0; i < x.size(); ++i)
                s += (x[i] - c.centre[i]) * (x[i] - c.centre[i]);
            return s;
        };
        flop::nelder_mead::Options o = v0111::options(1.0, 20000);
        o.bounds = flop::Bounds::box(Vec(n, 0.0), Vec(n, 8.0));
        o.stopping.xtol_abs = 1e-9;
        const Traced scalar = v0111::traced(f, c.x0, o);
        v0111::ReferenceRun ref;
        const v0111::AuditReport rep =
            v0111::audit(c.x0, o, scalar.trace, scalar.r, false, &ref, true);
        EXPECT_TRUE(rep.points_agree && rep.same_length && rep.same_outcome && rep.same_radius)
            << "n " << n << rep.detail;
        EXPECT_GE(ref.restarts, 1u) << "n " << n;
        EXPECT_LE(v0111::max_abs_diff(scalar.r.x, c.centre), 10.0 * o.stopping.xtol_abs)
            << "n " << n;
        auto g = v0111::batched(f);
        const Traced batch = v0111::traced_batch(g, c.x0, o);
        ASSERT_EQ(batch.trace.size(), scalar.trace.size()) << "n " << n;
        for (std::size_t k = 0; k < batch.trace.size(); ++k)
            ASSERT_TRUE(v0111::same_bits(batch.trace[k].x, scalar.trace[k].x))
                << "n " << n << ", " << k;
        EXPECT_EQ(batch.r.status, scalar.r.status);
        EXPECT_TRUE(v0111::same_bits(batch.r.final_radius, scalar.r.final_radius));
        if (n > 1) {
            bool rung_or_restart = false;
            for (std::size_t i = 1; i < g.sizes.size(); ++i) rung_or_restart |= g.sizes[i] > 1;
            EXPECT_TRUE(rung_or_restart) << "n " << n;
        }
    }
}
