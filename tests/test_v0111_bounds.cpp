// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.1.1: Nelder-Mead on a box.
//
// No paper defines the method on a box; FLOP's handling is documented
// (docs/algorithms/nelder-mead.md, "Box bounds" and "The initial simplex"):
// every trial point is projected onto the box; the initial vertex along
// coordinate i is x0 + h e_i, or x0 - h e_i where the box forbids +h, or
// moved to the bound on the side with more room where it forbids both; and
// before a tolerance stop, every coordinate in which all vertices sit on one
// bound is probed once, the best vertex moved off the bound by the radius
// (less where the other bound is closer), with a restart from the probe when
// it is better. Each of these is pinned to the bit on scripted runs.
//
// Pinned red for 0.1.1.2:
//   - the initial vertex moved to a bound is computed as x0 + (bound - x0),
//     which rounds past the bound for some x0 (x0 = -5, bound 0.2 gives
//     0.20000000000000018), so the method evaluates outside the box; the
//     helper is shared, and COBYLA does the same;
//   - the face test does not run before a RoundoffLimited stop, so a run
//     with only a cap ends flat on a face it could leave;
//   - a convex problem with its minimum off the face can end XtolReached a
//     unit in the last place from the bound, at a point that is not the
//     minimiser: the projected reflection triggers an outside contraction
//     that is projected too, and the simplex collapses next to the face
//     (after a restart from the face test, which then never fires again,
//     because the best vertex is no longer on the bound); and a minimum on
//     a corner of the box in five dimensions ends fifty tolerances away.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include "flop/detail/box.hpp"
#include "flop/flop.hpp"
#include "v0111_problems.hpp"
#include "v0111_reference_nm.hpp"

namespace {

using Vec = std::vector<double>;
using v0111::Table;
using v0111::Traced;

constexpr double kEps = std::numeric_limits<double>::epsilon();

bool inside(const flop::Bounds& b, std::span<const double> x) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        const std::optional<double>& lo = b.lower[i];
        const std::optional<double>& hi = b.upper[i];
        if (lo.has_value() && x[i] < *lo) return false;
        if (hi.has_value() && x[i] > *hi) return false;
    }
    return true;
}

void expect_all_inside(const flop::Bounds& b, const v0111::Trace& t, const char* what) {
    for (const v0111::Point& p : t) {
        EXPECT_TRUE(inside(b, p.x)) << what << ", evaluation " << p.index << " at " << p.x[0];
    }
}

void expect_points(const v0111::Trace& t, const std::vector<Vec>& want) {
    ASSERT_EQ(t.size(), want.size());
    for (std::size_t k = 0; k < want.size(); ++k)
        EXPECT_TRUE(v0111::same_bits(t[k].x, want[k])) << "evaluation " << k;
}

flop::nelder_mead::Options boxed(double h, std::size_t cap, const flop::Bounds& b) {
    flop::nelder_mead::Options o = v0111::options(h, cap);
    o.bounds = b;
    return o;
}

flop::Bounds box(const Vec& lo, const Vec& hi) {
    return flop::Bounds::box(lo, hi);
}

// The corner scenario, n = 2 on [0, 1]^2 from the corner (0, 0) with h = 1:
// the simplex is driven onto (0, 0) in both coordinates in five steps, all
// in exact arithmetic (every trial point is projected onto the box):
//   (0, 1) [x_r of s1, projected; equals s2], (0, 0.75) [x_c, projected],
//   (0, 0) [x_r, projected], (0, 0) [x_r], (0, 0) [x_c], radius 0.
// Then the face test probes coordinate 0, then coordinate 1, each by the
// precision floor epsilon * max(h, 0) = epsilon.
Vec c00() {
    return {0.0, 0.0};
}
Vec c10() {
    return {1.0, 0.0};
}
Vec c01() {
    return {0.0, 1.0};
}
Vec c0q() {
    return {0.0, 0.75};
}
Vec probe0() {
    return {kEps, 0.0};
}
Vec probe1() {
    return {0.0, kEps};
}

Table corner_table(double f_probe0, double f_probe1) {
    Table t;
    t.set(c00(), 1.0).set(c10(), 3.0).set(c01(), 2.0).set(c0q(), 1.5);
    t.set(probe0(), f_probe0).set(probe1(), f_probe1);
    return t;
}

std::vector<Vec> corner_path() {
    return {c00(), c10(), c01(), c01(), c0q(), c00(), c00(), c00()};
}

flop::Bounds unit_box() {
    return box(Vec{0.0, 0.0}, Vec{1.0, 1.0});
}

flop::nelder_mead::Options corner_options(std::size_t cap) {
    flop::nelder_mead::Options o = boxed(1.0, cap, unit_box());
    o.stopping.xtol_abs = 0.5;
    return o;
}

}  // namespace

// ============================================================================
// Projection
// ============================================================================

TEST(V0111Bounds, EveryEvaluationIsInsideTheBox) {
    const Vec lo3(3, -1.0), hi3(3, 0.25);
    const Vec wide_lo(3, -2.0), wide_hi(3, 2.0);
    struct Case {
        const char* what;
        double (*f)(std::span<const double>);
        Vec x0;
        flop::Bounds b;
        double h;
    };
    const std::vector<Case> cases{
        {.what = "minimum outside the box",
         .f = v0111::sphere,
         .x0 = Vec(3, 0.0),
         .b = box(lo3, hi3),
         .h = 0.5},
        {.what = "minimum inside the box",
         .f = v0111::sphere,
         .x0 = Vec(3, 0.0),
         .b = box(wide_lo, wide_hi),
         .h = 1.0},
        {.what = "start in a corner",
         .f = v0111::sphere,
         .x0 = Vec(3, -1.0),
         .b = box(lo3, hi3),
         .h = 1.0},
        {.what = "rosenbrock",
         .f = v0111::extended_rosenbrock,
         .x0 = Vec{-1.2, 1.0},
         .b = box(Vec{-2.0, -0.5}, Vec{0.5, 2.0}),
         .h = 0.5},
        {.what = "step larger than the box",
         .f = v0111::sphere,
         .x0 = Vec(3, 0.0),
         .b = box(lo3, hi3),
         .h = 4.0},
    };
    for (const Case& c : cases) {
        for (const bool adaptive : {false, true}) {
            flop::nelder_mead::Options o = boxed(c.h, 20000, c.b);
            o.adaptive_coefficients = adaptive;
            o.stopping.xtol_abs = 1e-10;
            const Traced run = v0111::traced(c.f, c.x0, o);
            expect_all_inside(c.b, run.trace, c.what);
            EXPECT_TRUE(inside(c.b, run.r.x)) << c.what;
        }
    }
    // The face tests and restarts below, too.
    const Traced corner = v0111::traced(corner_table(5.0, 0.5), c00(), corner_options(12));
    expect_all_inside(unit_box(), corner.trace, "the corner scenario");
}

TEST(V0111Bounds, AnOptimumOutsideTheBoxLandsOnItsFace) {
    // The sphere's centre 0.3 (i + 1) is beyond the upper bound 0.25 in every
    // coordinate, so the minimiser is the corner (0.25, ..., 0.25). The
    // projection puts vertices exactly on it.
    for (const std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{3}}) {
        for (const bool adaptive : {false, true}) {
            const Vec lo(n, -1.0), hi(n, 0.25);
            flop::nelder_mead::Options o = boxed(0.5, 100000, box(lo, hi));
            o.adaptive_coefficients = adaptive;
            o.stopping.xtol_abs = 1e-9;
            const Traced run = v0111::traced(v0111::sphere, Vec(n, 0.0), o);
            EXPECT_EQ(run.r.status, flop::Status::XtolReached) << n;
            EXPECT_TRUE(v0111::same_bits(run.r.x, hi)) << n;
        }
    }
}

TEST(V0111Bounds, AMinimumOnACornerInFiveDimensionsIsReached) {
    // Pinned red for 0.1.1.2. The same problem at n = 5 stops XtolReached at
    // xtol 1e-9 with the best vertex about 5e-8 from the corner: the simplex
    // collapsed next to the faces rather than onto them. Ten tolerances is
    // the stagnation bar the corpus uses.
    const std::size_t n = 5;
    const Vec lo(n, -1.0), hi(n, 0.25);
    for (const bool adaptive : {false, true}) {
        flop::nelder_mead::Options o = boxed(0.5, 100000, box(lo, hi));
        o.adaptive_coefficients = adaptive;
        o.stopping.xtol_abs = 1e-9;
        const Traced run = v0111::traced(v0111::sphere, Vec(n, 0.0), o);
        EXPECT_LE(v0111::max_abs_diff(run.r.x, hi), 10.0 * o.stopping.xtol_abs)
            << (adaptive ? "adaptive" : "standard");
    }
}

TEST(V0111Bounds, AnInactiveBoxGivesTheUnboundedTrajectoryToTheBit) {
    // The projection keeps the bits of a coordinate inside its bounds.
    struct Case {
        double (*f)(std::span<const double>);
        Vec x0;
        double h;
    };
    for (const Case& c :
         {Case{.f = v0111::extended_rosenbrock, .x0 = {-1.2, 1.0}, .h = 0.5},
          Case{.f = v0111::sphere, .x0 = Vec(4, 0.0), .h = 1.0},
          Case{.f = v0111::powell_singular, .x0 = {3.0, -1.0, 0.0, 1.0}, .h = 1.0}}) {
        flop::nelder_mead::Options free = v0111::options(c.h, 3000);
        free.stopping.xtol_abs = 1e-10;
        flop::nelder_mead::Options held = free;
        const Vec lo(c.x0.size(), -1e3), hi(c.x0.size(), 1e3);
        held.bounds = box(lo, hi);
        const Traced a = v0111::traced(c.f, c.x0, free);
        const Traced b = v0111::traced(c.f, c.x0, held);
        ASSERT_EQ(a.trace.size(), b.trace.size());
        for (std::size_t k = 0; k < a.trace.size(); ++k)
            ASSERT_TRUE(v0111::same_bits(a.trace[k].x, b.trace[k].x)) << k;
        EXPECT_EQ(a.r.status, b.r.status);
        EXPECT_TRUE(v0111::same_bits(a.r.final_radius, b.r.final_radius));
    }
}

// ============================================================================
// The initial simplex in a box
// ============================================================================

TEST(V0111Bounds, TheInitialSimplexFollowsTheDocumentedRule) {
    struct Case {
        const char* what;
        Vec x0;
        flop::Bounds b;
        double h;
        std::vector<Vec> vertices;
    };
    flop::Bounds lower_only = flop::Bounds::none(1);
    lower_only.lower[0] = 0.0;
    flop::Bounds upper_only = flop::Bounds::none(1);
    upper_only.upper[0] = 0.0;
    const std::vector<Case> cases{
        {.what = "room on both sides",
         .x0 = {0.5, 0.5},
         .b = box({0.0, 0.0}, {2.0, 2.0}),
         .h = 0.5,
         .vertices = {{0.5, 0.5}, {1.0, 0.5}, {0.5, 1.0}}},
        {.what = "on the upper bound",
         .x0 = {1.0},
         .b = box({-5.0}, {1.0}),
         .h = 0.5,
         .vertices = {{1.0}, {0.5}}},
        {.what = "on the lower bound",
         .x0 = {-5.0},
         .b = box({-5.0}, {1.0}),
         .h = 0.5,
         .vertices = {{-5.0}, {-4.5}}},
        {.what = "in a corner",
         .x0 = {0.0, 1.0},
         .b = box({0.0, 0.0}, {1.0, 1.0}),
         .h = 0.5,
         .vertices = {{0.0, 1.0}, {0.5, 1.0}, {0.0, 0.5}}},
        // Both rooms below h: to the bound on the side with more room.
        {.what = "narrower than the step",
         .x0 = {0.0, 0.0},
         .b = box({0.0, -0.5}, {0.25, 0.25}),
         .h = 1.0,
         .vertices = {{0.0, 0.0}, {0.25, 0.0}, {0.0, -0.5}}},
        {.what = "lower bound only",
         .x0 = {0.0},
         .b = lower_only,
         .h = 0.5,
         .vertices = {{0.0}, {0.5}}},
        {.what = "upper bound only",
         .x0 = {0.0},
         .b = upper_only,
         .h = 0.5,
         .vertices = {{0.0}, {-0.5}}},
    };
    for (const Case& c : cases) {
        const Traced run = v0111::traced(v0111::sphere, c.x0, boxed(c.h, c.vertices.size(), c.b));
        ASSERT_EQ(run.trace.size(), c.vertices.size()) << c.what;
        for (std::size_t k = 0; k < c.vertices.size(); ++k)
            EXPECT_TRUE(v0111::same_bits(run.trace[k].x, c.vertices[k]))
                << c.what << ", vertex " << k;
    }
}

TEST(V0111Bounds, TheOffsetHelperFollowsItsRule) {
    // flop::detail::axis_offset, shared by both algorithms: +h when the box
    // allows it, else -h, else the larger room, signed; +h without a box.
    const Vec x{0.0};
    EXPECT_EQ(flop::detail::axis_offset(nullptr, x, 0, 0.5), 0.5);
    const flop::Bounds wide = box({-1.0}, {1.0});
    EXPECT_EQ(flop::detail::axis_offset(&wide, x, 0, 0.5), 0.5);
    const flop::Bounds at_hi = box({-1.0}, {0.0});
    EXPECT_EQ(flop::detail::axis_offset(&at_hi, x, 0, 0.5), -0.5);
    const flop::Bounds narrow_up = box({-0.125}, {0.25});
    EXPECT_EQ(flop::detail::axis_offset(&narrow_up, x, 0, 0.5), 0.25);
    const flop::Bounds narrow_down = box({-0.25}, {0.125});
    EXPECT_EQ(flop::detail::axis_offset(&narrow_down, x, 0, 0.5), -0.25);
    const flop::Bounds equal_rooms = box({-0.25}, {0.25});
    EXPECT_EQ(flop::detail::axis_offset(&equal_rooms, x, 0, 0.5), 0.25);
}

namespace {

// Boxes narrower than the step where x0 + (bound - x0) rounds past the bound.
struct Overshoot {
    double x0;
    double lo;
    double hi;
    double h;
};
const Overshoot overshoots[] = {
    {.x0 = -5.0,
     .lo = -5.5,
     .hi = 0.2,
     .h = 10.0},  // to the upper bound: -5 + (0.2 + 5) = 0.20000000000000018
    {.x0 = -0.7,
     .lo = -3.1,
     .hi = -0.5,
     .h = 10.0},  // to the lower bound: -0.7 - 2.4 = -3.1000000000000005
};

}  // namespace

TEST(V0111Bounds, AnInitialVertexMovedToABoundLandsOnIt) {
    // Pinned red for 0.1.1.2 (in the strict binaries; under -ffast-math the
    // compiler may fold x0 + (bound - x0) to the bound).
    for (const Overshoot& c : overshoots) {
        const flop::Bounds b = box({c.lo}, {c.hi});
        const Vec x0{c.x0};
        const double v = c.x0 + flop::detail::axis_offset(&b, x0, 0, c.h);
        EXPECT_TRUE(inside(b, std::span<const double>(&v, 1))) << c.x0 << " -> " << v;
        const Traced run = v0111::traced(v0111::sphere, x0, boxed(c.h, 50, b));
        expect_all_inside(b, run.trace, "nelder_mead");
        v0111::Trace cobyla_trace;
        flop::cobyla::Options co;
        co.initial_step = c.h;
        co.stopping.max_evaluations = 50;
        co.bounds = b;
        co.on_evaluation = v0111::recorder(cobyla_trace);
        (void)flop::cobyla::minimize(v0111::sphere, x0, co);
        expect_all_inside(b, cobyla_trace, "cobyla");
    }
}

// ============================================================================
// The face test
// ============================================================================

TEST(V0111Bounds, AFaceHoldingTheMinimumKeepsItsStopAfterOneProbe) {
    // f = (x + 1)^2 on [0, 8] from 0, h = 1: the reflection and the outside
    // contraction both project onto 0, the simplex is {0, 0}, radius 0. The
    // probe is 0 + max(radius, epsilon * max(h, |x_1|)) = epsilon, worse, so
    // XtolReached stands after five evaluations.
    auto f = [](std::span<const double> x) { return (x[0] + 1.0) * (x[0] + 1.0); };
    flop::nelder_mead::Options o = boxed(1.0, 100, box({0.0}, {8.0}));
    o.stopping.xtol_abs = 1e-9;
    const Traced run = v0111::traced(f, Vec{0.0}, o);
    EXPECT_EQ(run.r.status, flop::Status::XtolReached);
    expect_points(run.trace, {{0.0}, {1.0}, {0.0}, {0.0}, {kEps}});
    EXPECT_TRUE(v0111::same_bits(run.r.x, Vec{0.0}));
    EXPECT_EQ(run.r.final_radius, 0.0);
}

namespace {

// n = 2, x in [0, b], y free, from (0, 0) with h = 2. The x step is
// shrunk to the room, s1 = (b, 0); s2 = (0, 2). Ranked s0, s2, s1 (f = 1,
// 2, 3): x_r of s1 projects onto (0, 2) = s2 (f = 2 = f_n), so an outside
// contraction, projected onto (0, 1.5), accepted at f = 0.875. Every vertex
// now has x = 0 and the spread is 2 - 0.875 = 1.125, the f tolerance: the
// face test probes the best vertex (0, 1.5) by the radius, 1.5, in x.
Vec g0() {
    return {0.0, 0.0};
}
Vec g2() {
    return {0.0, 2.0};
}
Vec gc() {
    return {0.0, 1.5};
}

Table face_table(const Vec& s1, const Vec& probe, double f_probe) {
    Table t;
    t.set(g0(), 1.0).set(s1, 3.0).set(g2(), 2.0).set(gc(), 0.875).set(probe, f_probe);
    return t;
}

flop::nelder_mead::Options face_options(double b, std::size_t cap) {
    flop::Bounds bounds = flop::Bounds::none(2);
    bounds.lower[0] = 0.0;
    bounds.upper[0] = b;
    flop::nelder_mead::Options o = boxed(2.0, cap, bounds);
    o.stopping.ftol_abs = 1.125;
    return o;
}

}  // namespace

TEST(V0111Bounds, TheProbeIsDisplacedByTheRadiusOrLessWhereTheOtherBoundIsCloser) {
    {
        // x in [0, 8]: the probe is the full radius off the face.
        const Vec s1{2.0, 0.0};
        const Vec probe{1.5, 1.5};
        Table t = face_table(s1, probe, 4.0);
        const Traced run = v0111::traced(t, g0(), face_options(8.0, 100));
        EXPECT_EQ(run.r.status, flop::Status::FtolReached);
        expect_points(run.trace, {g0(), s1, g2(), g2(), gc(), probe});
    }
    {
        // x in [0, 0.25]: the other bound is closer than the radius.
        const Vec s1{0.25, 0.0};
        const Vec probe{0.25, 1.5};
        Table t = face_table(s1, probe, 4.0);
        const Traced run = v0111::traced(t, g0(), face_options(0.25, 100));
        EXPECT_EQ(run.r.status, flop::Status::FtolReached);
        expect_points(run.trace, {g0(), s1, g2(), g2(), gc(), probe});
        EXPECT_TRUE(v0111::same_bits(run.r.x, gc()));
    }
}

TEST(V0111Bounds, ABetterProbeRestartsWithOnlyNNewPoints) {
    // The probe (0.25, 1.5) is better, so the method starts again from it:
    // the probe is slot 0 and is not evaluated again; the n new vertices
    // follow the initial-simplex rule from it (x: no room up, 0.25 down, so
    // to the lower bound, (0, 1.5); y: +2, (0.25, 3.5)).
    const Vec s1{0.25, 0.0};
    const Vec probe{0.25, 1.5};
    Table t = face_table(s1, probe, 0.5);
    t.set({0.25, 3.5}, 4.0);
    const Traced run = v0111::traced(t, g0(), face_options(0.25, 8));
    expect_points(run.trace, {g0(), s1, g2(), g2(), gc(), probe, gc(), {0.25, 3.5}});
    EXPECT_EQ(run.r.status, flop::Status::MaxEvaluationsReached);
    EXPECT_TRUE(v0111::same_bits(run.r.x, probe));
}

TEST(V0111Bounds, FlatCoordinatesAreProbedInCoordinateOrder) {
    {
        // Both probes worse: the stop stands after both.
        const Traced run = v0111::traced(corner_table(5.0, 5.0), c00(), corner_options(100));
        std::vector<Vec> want = corner_path();
        want.push_back(probe0());
        want.push_back(probe1());
        expect_points(run.trace, want);
        EXPECT_EQ(run.r.status, flop::Status::XtolReached);
        EXPECT_EQ(run.r.final_radius, 0.0);
    }
    {
        // The second probe better: restart from it, with its two new
        // vertices (x: +1, (1, eps); y: both rooms below 1, the larger is
        // up, to the bound, (0, 1)).
        const Traced run = v0111::traced(corner_table(5.0, 0.5), c00(), corner_options(12));
        std::vector<Vec> want = corner_path();
        want.push_back(probe0());
        want.push_back(probe1());
        want.push_back({1.0, kEps});
        want.push_back({0.0, 1.0});
        expect_points(run.trace, want);
    }
    {
        // The first probe better: restart at once, the second never probed
        // (x: rooms 1 - eps and eps, to the upper bound, (1, 0); y: +1,
        // (eps, 1)).
        const Traced run = v0111::traced(corner_table(0.5, 5.0), c00(), corner_options(11));
        std::vector<Vec> want = corner_path();
        want.push_back(probe0());
        want.push_back({1.0, 0.0});
        want.push_back({kEps, 1.0});
        expect_points(run.trace, want);
    }
}

TEST(V0111Bounds, ProbesAndRestartPointsAreEvaluationsLikeAnyOther) {
    {
        // The cap refuses the second probe: MaxEvaluationsReached, and the
        // radius is the flat simplex's.
        const Traced run = v0111::traced(corner_table(5.0, 5.0), c00(), corner_options(9));
        EXPECT_EQ(run.r.status, flop::Status::MaxEvaluationsReached);
        EXPECT_EQ(run.r.evaluations, 9u);
        EXPECT_EQ(run.r.final_radius, 0.0);
    }
    {
        // stop_value met at a probe.
        flop::nelder_mead::Options o = corner_options(100);
        o.stopping.stop_value = 0.6;
        const Traced run = v0111::traced(corner_table(5.0, 0.5), c00(), o);
        EXPECT_EQ(run.r.status, flop::Status::StopValueReached);
        EXPECT_EQ(run.r.evaluations, 10u);
        EXPECT_TRUE(v0111::same_bits(run.r.x, probe1()));
    }
    {
        // stop_value met at a restart's first new vertex.
        flop::nelder_mead::Options o = corner_options(100);
        o.stopping.stop_value = 0.2;
        Table t = corner_table(5.0, 0.5);
        t.set({1.0, kEps}, 0.1);
        const Traced run = v0111::traced(t, c00(), o);
        EXPECT_EQ(run.r.status, flop::Status::StopValueReached);
        EXPECT_EQ(run.r.evaluations, 11u);
        EXPECT_TRUE(v0111::same_bits(run.r.x, Vec{1.0, kEps}));
    }
}

TEST(V0111Bounds, FinalRadiusOnACapInsideARestartIsTheLastEvaluatedSimplex) {
    // Pinned red for 0.1.1.2. The cap admits the restart's first new vertex
    // and refuses the second. The last simplex whose every vertex was
    // evaluated is the flat one at the corner, radius 0.
    const Traced run = v0111::traced(corner_table(5.0, 0.5), c00(), corner_options(11));
    ASSERT_EQ(run.r.status, flop::Status::MaxEvaluationsReached);
    EXPECT_EQ(run.r.final_radius, 0.0);
}

TEST(V0111Bounds, RestartsStartFromStrictlyLowerValuesAndTheRunEnds) {
    // A convex problem whose simplex is driven onto two faces more than
    // once: each restart is from a probe strictly better than the best
    // vertex, so there are finitely many, and the run ends before the cap.
    auto f = [](std::span<const double> x) {
        return (x[0] - 0.125) * (x[0] - 0.125) + (x[1] - 0.25) * (x[1] - 0.25);
    };
    const flop::Bounds b = box({0.0, 0.0}, {8.0, 8.0});
    flop::nelder_mead::Options o = boxed(1.0, 5000, b);
    o.stopping.xtol_abs = 1e-10;
    const Traced run = v0111::traced(f, Vec{0.0, 0.0}, o);
    EXPECT_LT(run.r.evaluations, 5000u);
    v0111::ReferenceRun ref;
    const v0111::AuditReport rep =
        v0111::audit(Vec{0.0, 0.0}, o, run.trace, run.r, false, &ref, true);
    EXPECT_TRUE(rep.points_agree && rep.same_length && rep.same_outcome) << rep.detail;
    EXPECT_GE(ref.restarts, 1u);
    for (const v0111::Step& s : ref.steps) {
        if (s.termination != v0111::Termination::Restart) continue;
        EXPECT_LT(s.after[0].f, s.before[0].f);
        EXPECT_EQ(s.evaluations, 1u + 2u);  // the probe and n new vertices
    }
    expect_all_inside(b, run.trace, "restarts");
}

TEST(V0111Bounds, AConvexProblemWithItsMinimumOffTheFaceReachesIt) {
    // Pinned red for 0.1.1.2. f = (x - 0.25)^2 on [0, 8] from 0, h = 1: flat
    // on x = 0, the probe at epsilon is better, the restart's simplex
    // {epsilon, 1 + epsilon} collapses back onto {epsilon, 0} and the run
    // ends XtolReached at x = epsilon, f = 0.0625. In two dimensions the
    // same happens at (epsilon, epsilon) after two restarts.
    {
        auto f = [](std::span<const double> x) { return (x[0] - 0.25) * (x[0] - 0.25); };
        flop::nelder_mead::Options o = boxed(1.0, 5000, box({0.0}, {8.0}));
        o.stopping.xtol_abs = 1e-9;
        const Traced run = v0111::traced(f, Vec{0.0}, o);
        EXPECT_EQ(run.r.status, flop::Status::XtolReached);
        EXPECT_LE(std::fabs(run.r.x[0] - 0.25), 10.0 * o.stopping.xtol_abs) << run.r.x[0];
    }
    {
        auto f = [](std::span<const double> x) {
            return (x[0] - 0.125) * (x[0] - 0.125) + (x[1] - 0.25) * (x[1] - 0.25);
        };
        flop::nelder_mead::Options o = boxed(1.0, 5000, box({0.0, 0.0}, {8.0, 8.0}));
        o.stopping.xtol_abs = 1e-9;
        const Traced run = v0111::traced(f, Vec{0.0, 0.0}, o);
        EXPECT_EQ(run.r.status, flop::Status::XtolReached);
        EXPECT_LE(v0111::max_abs_diff(run.r.x, Vec{0.125, 0.25}), 10.0 * o.stopping.xtol_abs);
    }
}

TEST(V0111Bounds, ARunWithOnlyACapStillGetsTheFaceTest) {
    // Pinned red for 0.1.1.2. The same f, no tolerance: the flat simplex
    // {0, 0} reaches the precision floor and the run ends RoundoffLimited
    // after four evaluations, without the probe at epsilon that a tolerance
    // stop would have made.
    auto f = [](std::span<const double> x) { return (x[0] - 0.25) * (x[0] - 0.25); };
    const Traced run = v0111::traced(f, Vec{0.0}, boxed(1.0, 2000, box({0.0}, {8.0})));
    ASSERT_GE(run.trace.size(), 5u);
    EXPECT_TRUE(v0111::same_bits(run.trace[4].x, Vec{kEps}));
}
