// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.1.3: Nelder-Mead with its minimiser on a face of the box.
//
// The gradient at a minimiser on a face points through the bound, so the
// simplex keeps finding descent toward it, never contracts and never meets
// a stopping test: from n = 6 with Gao and Han's coefficients and n = 10
// with the standard ones, a sphere whose minimiser lies on a face ran to
// any cap, gaps of 1e-4 to 1, and the ladder, which runs only before a stop,
// never ran. The face test polls x_1 onto every bound within the diameter
// of the simplex once the box has acted, fixes a coordinate whose poll
// improves, and continues on the free coordinates; the ladder frees a fixed
// coordinate whose inward poll improves. This suite pins the runs that
// stalled, the corner, a bound near an interior minimiser, the reference
// through fixes and frees, both channels, and the stop with every
// coordinate fixed.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <numbers>
#include <optional>
#include <span>
#include <vector>

#include "flop/flop.hpp"
#include "v0111_problems.hpp"
#include "v0111_reference_nm.hpp"

namespace {

using Vec = std::vector<double>;
using v0111::Traced;

constexpr double kCentre = 0.1;  // every coordinate of the sphere's centre
constexpr double kXtol = 1e-8;
constexpr double kStep = 0.3;

double sphere(std::span<const double> x) {
    double s = 0.0;
    for (const double v : x) s += (v - kCentre) * (v - kCentre);
    return s;
}

// Which coordinates carry the raised lower bound.
enum class Face : std::uint8_t { Even, All };

// [-2 pi, 2 pi]^n with the lower bound `lower` on the even coordinates, or
// on all of them. With lower > kCentre the minimiser is on that face (on the
// corner for All); with lower < kCentre it is the interior centre.
flop::Bounds face_box(std::size_t n, double lower, Face face) {
    Vec lo(n, -2.0 * std::numbers::pi), hi(n, 2.0 * std::numbers::pi);
    for (std::size_t i = 0; i < n; i += face == Face::All ? 1 : 2) lo[i] = lower;
    return flop::Bounds::box(lo, hi);
}

// The minimiser of the sphere on the box: each coordinate's centre clipped
// to its bounds.
Vec minimiser(const flop::Bounds& b) {
    Vec x(b.lower.size(), kCentre);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const std::optional<double>& lo = b.lower[i];
        if (lo.has_value() && x[i] < *lo) x[i] = *lo;
    }
    return x;
}

// Deterministic starts inside every box above, away from the faces.
Vec start(std::size_t n, int k) {
    Vec x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = 0.5 + 0.2 * std::sin(1.3 * static_cast<double>(i + 1) + 2.1 * k);
    return x;
}

flop::nelder_mead::Options options(const flop::Bounds& b, std::size_t n, bool adaptive) {
    flop::nelder_mead::Options o = v0111::options(kStep, 2000 * n, adaptive);
    o.bounds = b;
    o.stopping.xtol_abs = kXtol;
    return o;
}

}  // namespace

TEST(V0113ActiveSet, AMinimiserOnAFaceIsReached) {
    // The rows that ran to the cap: n = 6, 8 and 16, both coefficient sets,
    // four starts. Each now stops on its own within ten tolerances of the
    // minimiser.
    for (const std::size_t n : std::initializer_list<std::size_t>{6, 8, 16}) {
        const flop::Bounds b = face_box(n, 0.2, Face::Even);
        const Vec xs = minimiser(b);
        for (const bool adaptive : {true, false}) {
            for (int k = 0; k < 4; ++k) {
                const flop::nelder_mead::Options o = options(b, n, adaptive);
                const flop::Result r = flop::nelder_mead::minimize(sphere, start(n, k), o);
                EXPECT_EQ(r.status, flop::Status::XtolReached)
                    << "n " << n << (adaptive ? " adaptive" : " standard") << " k " << k;
                EXPECT_LE(v0111::max_abs_diff(r.x, xs), 10.0 * kXtol)
                    << "n " << n << (adaptive ? " adaptive" : " standard") << " k " << k;
            }
        }
    }
}

TEST(V0113ActiveSet, ACornerIsReachedOnItsBoundsToTheBit) {
    // Every coordinate's minimiser on its lower bound. A fixed coordinate is
    // copied, never averaged, so the result is the corner itself, not a
    // point a rounding away from it.
    for (const std::size_t n : std::initializer_list<std::size_t>{3, 8, 16}) {
        const flop::Bounds b = face_box(n, 0.2, Face::All);
        for (int k = 0; k < 4; ++k) {
            const flop::Result r =
                flop::nelder_mead::minimize(sphere, start(n, k), options(b, n, true));
            EXPECT_EQ(r.status, flop::Status::XtolReached) << "n " << n << " k " << k;
            EXPECT_TRUE(v0111::same_bits(r.x, minimiser(b))) << "n " << n << " k " << k;
        }
    }
}

TEST(V0113ActiveSet, TheReferenceAgreesThroughFixesAndFrees) {
    // FLOP against the reference transcription, point by point, on the face,
    // the corner, and a bound just below the centre, where the face test
    // fixes coordinates the ladder later frees.
    struct Row {
        double lower;
        Face face;
    };
    std::size_t fixes = 0, frees = 0;
    for (const Row row :
         {Row{.lower = 0.2, .face = Face::Even}, Row{.lower = 0.2, .face = Face::All},
          Row{.lower = 0.09, .face = Face::Even}}) {
        for (const std::size_t n : std::initializer_list<std::size_t>{2, 4, 8}) {
            const flop::Bounds b = face_box(n, row.lower, row.face);
            for (const bool adaptive : {true, false}) {
                for (int k = 0; k < 2; ++k) {
                    const flop::nelder_mead::Options o = options(b, n, adaptive);
                    const Vec x0 = start(n, k);
                    const Traced run = v0111::traced(sphere, x0, o);
                    v0111::ReferenceRun ref;
                    const v0111::AuditReport rep =
                        v0111::audit(x0, o, run.trace, run.r, false, &ref, true);
                    EXPECT_TRUE(rep.points_agree && rep.same_length && rep.same_outcome &&
                                rep.same_radius)
                        << "lower " << row.lower << " n " << n << " k " << k << ": " << rep.detail;
                    fixes += ref.fixes;
                    frees += ref.frees;
                }
            }
        }
    }
    EXPECT_GT(fixes, 0u);
    EXPECT_GT(frees, 0u);
}

TEST(V0113ActiveSet, ABoundNearAnInteriorMinimiserIsFreedAgain) {
    // The lower bound 0.09 sits just below the centre 0.1. While the simplex
    // is wider than the gap the poll onto the bound can be better than x_1,
    // and the coordinate is fixed; the ladder's inward poll frees it. The
    // minimiser is interior, so no coordinate can end fixed: every fix is
    // undone, and the run ends at the centre.
    for (const std::size_t n : std::initializer_list<std::size_t>{4, 8}) {
        const flop::Bounds b = face_box(n, 0.09, Face::Even);
        for (int k = 0; k < 4; ++k) {
            const flop::nelder_mead::Options o = options(b, n, true);
            const Vec x0 = start(n, k);
            const Traced run = v0111::traced(sphere, x0, o);
            EXPECT_EQ(run.r.status, flop::Status::XtolReached) << "n " << n << " k " << k;
            EXPECT_LE(v0111::max_abs_diff(run.r.x, Vec(n, kCentre)), 10.0 * kXtol)
                << "n " << n << " k " << k;
            v0111::ReferenceRun ref;
            ASSERT_TRUE(v0111::audit(x0, o, run.trace, run.r, false, &ref, true).points_agree);
            EXPECT_EQ(ref.fixes, ref.frees) << "n " << n << " k " << k;
        }
    }
}

TEST(V0113ActiveSet, BothChannelsEvaluateTheSamePointsThroughFixes) {
    // The face test's candidates go out as one call with a batch objective;
    // the points and their order are those of the scalar channel.
    for (const Face face : {Face::Even, Face::All}) {
        for (const std::size_t n : std::initializer_list<std::size_t>{4, 8}) {
            const flop::Bounds b = face_box(n, 0.2, face);
            const flop::nelder_mead::Options o = options(b, n, true);
            const Vec x0 = start(n, 0);
            const Traced scalar = v0111::traced(sphere, x0, o);
            auto g = v0111::batched(sphere);
            const Traced batch = v0111::traced_batch(g, x0, o);
            ASSERT_EQ(batch.trace.size(), scalar.trace.size()) << "n " << n;
            for (std::size_t k = 0; k < batch.trace.size(); ++k)
                ASSERT_TRUE(v0111::same_bits(batch.trace[k].x, scalar.trace[k].x))
                    << "n " << n << ", evaluation " << k;
            EXPECT_EQ(batch.r.status, scalar.r.status);
            EXPECT_TRUE(v0111::same_bits(batch.r.x, scalar.r.x));
        }
    }
}

TEST(V0113ActiveSet, WithEveryCoordinateFixedTheLadderDecidesTheStop) {
    // On the corner every coordinate ends fixed, the simplex is the one
    // point, and the stopping tests conclude at once. The ladder then polls
    // the corner + d e_i for d = kStep, kStep / 2, ... down to kXtol, the
    // minus points clipping back onto the corner and left out; every one is
    // worse (the centre is below every bound), so the stop stands. Those
    // rungs are the last evaluations of the run, in rung, then coordinate,
    // order.
    const std::size_t n = 3;
    const flop::Bounds b = face_box(n, 0.2, Face::All);
    const Vec corner = minimiser(b);
    const Traced run = v0111::traced(sphere, start(n, 0), options(b, n, true));
    ASSERT_EQ(run.r.status, flop::Status::XtolReached);
    std::vector<Vec> rungs;
    double d = kStep;
    while (d >= kXtol) {
        for (std::size_t i = 0; i < n; ++i) {
            Vec p = corner;
            p[i] = corner[i] + d;
            rungs.push_back(p);
        }
        d *= 0.5;
    }
    ASSERT_GE(run.trace.size(), rungs.size());
    const std::size_t first = run.trace.size() - rungs.size();
    for (std::size_t k = 0; k < rungs.size(); ++k)
        EXPECT_TRUE(v0111::same_bits(run.trace[first + k].x, rungs[k])) << "rung point " << k;
    EXPECT_TRUE(v0111::same_bits(run.r.x, corner));
}
