// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.1.1: the O(n) bookkeeping, held through what it decides.
//
// FLOP's iteration is O(n) because three things are kept rather than
// recomputed (nelder_mead_impl.hpp): the sum of the vertices, rebuilt every
// n + 1 replacements; per coordinate the least and greatest vertex value
// and the slot holding each, rescanned only when the discarded vertex held
// one and the new vertex does not take it over; and from those extents the
// diameter D, which decides an x tolerance t without the O(n^2) radius r
// whenever D <= t (so r <= t) or D > 2t (so r > t). None of it is visible
// except through the points the method evaluates and where it stops, so
// that is where it is held, against the reference transcription, which
// keeps none of it.
//
// The stop is the sharp test. For each radius r_k the reference saw at an
// iteration's stopping test, a run with xtol_abs = r_k must stop exactly
// there, and one with xtol_abs just below r_k must not. That puts the
// tolerance at every point of the bracket D/2 <= r <= D the run passes
// through, at both of its ends, on runs that discard extreme vertices in
// every way the extents distinguish.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include "flop/flop.hpp"
#include "v0111_problems.hpp"
#include "v0111_reference_nm.hpp"

namespace {

using Vec = std::vector<double>;

double staircase(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        s += std::floor(4.0 * std::fabs(x[i] - v0111::sphere_centre(i) - 0.11));
    return s;
}

struct Base {
    const char* name;
    double (*f)(std::span<const double>);
    Vec x0;
    double step;
    std::optional<flop::Bounds> bounds;
    std::size_t cap;
};

std::vector<Base> bases() {
    std::vector<Base> out;
    out.push_back({.name = "staircase 2",
                   .f = staircase,
                   .x0 = Vec(2, 0.0),
                   .step = 1.0,
                   .bounds = std::nullopt,
                   .cap = 300});
    out.push_back({.name = "staircase 4",
                   .f = staircase,
                   .x0 = Vec(4, 0.0),
                   .step = 1.0,
                   .bounds = std::nullopt,
                   .cap = 300});
    out.push_back({.name = "sphere 3",
                   .f = v0111::sphere,
                   .x0 = Vec(3, 0.0),
                   .step = 1.0,
                   .bounds = std::nullopt,
                   .cap = 300});
    out.push_back({.name = "rosenbrock 2",
                   .f = v0111::extended_rosenbrock,
                   .x0 = Vec{-1.2, 1.0},
                   .step = 0.5,
                   .bounds = std::nullopt,
                   .cap = 300});
    // A box face holding the minimum: coordinates go flat on the bound, so a
    // discarded vertex can hold both extremes of a coordinate.
    const Vec lo(3, -1.0), hi(3, 0.25);
    out.push_back({.name = "sphere 3 on a face",
                   .f = v0111::sphere,
                   .x0 = Vec(3, 0.0),
                   .step = 0.5,
                   .bounds = flop::Bounds::box(lo, hi),
                   .cap = 300});
    return out;
}

flop::nelder_mead::Options base_options(const Base& b) {
    flop::nelder_mead::Options o = v0111::options(b.step, b.cap);
    o.bounds = b.bounds;
    return o;
}

}  // namespace

TEST(V0111Bookkeeping, TheRunningSumStaysTrueFarFromTheOrigin) {
    // Coordinates near 1e8 with steps near 1e-3: every update of the sum
    // rounds at 1e8's scale, so without the rebuild the centroid would drift
    // by the accumulated rounding of thousands of updates. With it, every
    // point stays within the rounding of n + 1 updates of the reference's.
    for (const std::size_t n : {std::size_t{4}, std::size_t{8}}) {
        const double offset = 1.0e8;
        auto far = [&](std::span<const double> x) {
            double s = 0.0;
            for (std::size_t i = 0; i < x.size(); ++i) {
                const double d = x[i] - offset - v0111::sphere_centre(i);
                s += d * d;
            }
            return s;
        };
        const Vec x0(n, offset);
        const flop::nelder_mead::Options o = v0111::options(1e-3, 20000);
        const v0111::Traced run = v0111::traced(far, x0, o);
        v0111::ReferenceRun ref;
        const v0111::AuditReport rep = v0111::audit(x0, o, run.trace, run.r, false, &ref);
        EXPECT_TRUE(rep.points_agree) << rep.detail;
        EXPECT_TRUE(rep.same_length) << rep.detail;
        EXPECT_TRUE(rep.same_outcome) << rep.detail;
        EXPECT_GE(ref.top_radius.size(), 10 * (n + 1));
    }
}

TEST(V0111Bookkeeping, TheStopIsDecidedExactlyAtEveryRadiusTheRunPasses) {
    std::size_t runs = 0;
    for (const Base& b : bases()) {
        const flop::nelder_mead::Options o = base_options(b);
        const v0111::Traced base_run = v0111::traced(b.f, b.x0, o);
        v0111::ReferenceRun ref;
        ASSERT_TRUE(v0111::audit(b.x0, o, base_run.trace, base_run.r, false, &ref).points_agree);
        std::vector<double> radii = ref.top_radius;
        std::ranges::sort(radii);
        const auto dup = std::ranges::unique(radii);
        radii.erase(dup.begin(), dup.end());
        for (const double r : radii) {
            if (!(r > 0.0)) continue;
            for (const double t : {r, std::nextafter(r, 0.0)}) {
                flop::nelder_mead::Options ot = o;
                ot.stopping.xtol_abs = t;
                const v0111::Traced run = v0111::traced(b.f, b.x0, ot);
                const v0111::AuditReport rep = v0111::audit(b.x0, ot, run.trace, run.r, false);
                EXPECT_TRUE(rep.points_agree && rep.same_length && rep.same_outcome)
                    << b.name << ", xtol_abs " << t << ": " << rep.detail;
                EXPECT_TRUE(rep.same_radius) << b.name << ", xtol_abs " << t << ": " << rep.detail;
                ++runs;
            }
        }
    }
    EXPECT_GT(runs, 200u);
}

TEST(V0111Bookkeeping, TheRunsDiscardExtremesInEveryWayTheExtentsDistinguish) {
    // The cases the extents treat differently, counted on the runs above:
    // the discarded vertex alone held the least value of a coordinate, or
    // alone the greatest, and the new vertex did not take it over (a
    // rescan); a coordinate in which every vertex sat at one value, so the
    // discarded vertex held both extremes and the new vertex, whose
    // component there is the centroid's, takes both over at equality; and a
    // new vertex landing exactly on an extreme of the simplex it joins.
    std::size_t only_least = 0, only_greatest = 0, both = 0, at_extreme = 0;
    for (const Base& b : bases()) {
        const flop::nelder_mead::Options o = base_options(b);
        const v0111::Traced run = v0111::traced(b.f, b.x0, o);
        v0111::ReferenceRun ref;
        ASSERT_TRUE(v0111::audit(b.x0, o, run.trace, run.r, false, &ref, true).points_agree);
        for (const v0111::Step& s : ref.steps) {
            if (s.termination == v0111::Termination::Shrink ||
                s.termination == v0111::Termination::Restart)
                continue;
            const std::size_t n = s.before.size() - 1;
            const v0111::Vertex& gone = s.before[n];
            // The accepted point: the one vertex of the new simplex that is
            // not a vertex of the old.
            const v0111::Vertex* added = nullptr;
            for (const v0111::Vertex& v : s.after) {
                const bool old = std::ranges::any_of(
                    s.before, [&](const v0111::Vertex& w) { return v0111::same_bits(v.x, w.x); });
                if (!old) added = &v;
            }
            if (!added) continue;
            for (std::size_t i = 0; i < n; ++i) {
                double kept_lo = std::numeric_limits<double>::max();
                double kept_hi = std::numeric_limits<double>::lowest();
                for (std::size_t k = 0; k < n; ++k) {
                    kept_lo = std::min(kept_lo, s.before[k].x[i]);
                    kept_hi = std::max(kept_hi, s.before[k].x[i]);
                }
                const double g = gone.x[i];
                const double a = added->x[i];
                if (g < kept_lo && a > g) ++only_least;
                if (g > kept_hi && a < g) ++only_greatest;
                if (g == kept_lo && g == kept_hi) ++both;
                const double lo = std::min(kept_lo, g);
                const double hi = std::max(kept_hi, g);
                if (a == lo || a == hi) ++at_extreme;
            }
        }
    }
    EXPECT_GT(only_least, 0u);
    EXPECT_GT(only_greatest, 0u);
    EXPECT_GT(both, 0u);
    EXPECT_GT(at_extreme, 0u);
}
