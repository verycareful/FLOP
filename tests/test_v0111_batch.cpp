// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.1.1: Nelder-Mead's batch channel.
//
// The method has independent points in three places: the initial simplex
// (n + 1 points, x0 first), a shrink (n points, step 5 of Lagarias et al.),
// and a restart after the face test (n points). The batch objective gets
// each of those as one call, when the cap allows the whole call, and every
// other evaluation alone. The contract is on the shape of the calls, held
// here against the reference transcription's record of which iteration
// shrank, and on the points: the batch run evaluates what the scalar run
// evaluates, in the same order, to the bit.
//
// stop_value met inside a batch call is the one place the channels differ,
// by design: the call has already evaluated every point in it, so all of
// them are counted, traced, and eligible as the result.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <numeric>
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

// The call sizes the documented contract implies for a run the reference
// replayed: the initial simplex in one call, each shrink's n points in one,
// everything else alone, including whatever the run evaluated after its
// last completed iteration.
std::vector<std::size_t> expected_calls(std::size_t n, const v0111::ReferenceRun& ref,
                                        std::size_t total) {
    std::vector<std::size_t> sizes{n + 1};
    std::size_t done = n + 1;
    for (const v0111::Step& s : ref.steps) {
        if (s.termination == v0111::Termination::Shrink) {
            sizes.insert(sizes.end(), {1, 1, n});
        } else if (s.termination == v0111::Termination::Restart) {
            sizes.push_back(1);
            sizes.push_back(n);
        } else {
            sizes.insert(sizes.end(), s.evaluations, 1);
        }
        done += s.evaluations;
    }
    sizes.insert(sizes.end(), total - done, 1);
    return sizes;
}

void expect_same_run(const Traced& a, const Traced& b, const char* what) {
    ASSERT_EQ(a.trace.size(), b.trace.size()) << what;
    for (std::size_t k = 0; k < a.trace.size(); ++k)
        ASSERT_TRUE(v0111::same_bits(a.trace[k].x, b.trace[k].x)) << what << ", evaluation " << k;
    EXPECT_EQ(a.r.status, b.r.status) << what;
    EXPECT_EQ(a.r.evaluations, b.r.evaluations) << what;
    EXPECT_TRUE(v0111::same_bits(a.r.x, b.r.x)) << what;
    EXPECT_TRUE(v0111::same_bits(a.r.f, b.r.f)) << what;
    EXPECT_TRUE(v0111::same_bits(a.r.final_radius, b.r.final_radius)) << what;
}

}  // namespace

TEST(V0111Batch, TheFirstCallIsTheWholeInitialSimplexWithX0First) {
    for (const std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{5}, std::size_t{16}}) {
        Vec x0(n);
        for (std::size_t i = 0; i < n; ++i) x0[i] = 0.1 * static_cast<double>(i + 1);
        auto f = v0111::batched(v0111::sphere);
        const Traced run = v0111::traced_batch(f, x0, v0111::options(0.3, 200));
        ASSERT_FALSE(f.sizes.empty());
        EXPECT_EQ(f.sizes[0], n + 1);
        const v0111::Simplex want = v0111::axis_simplex(x0, 0.3);
        ASSERT_GE(run.trace.size(), n + 1);
        for (std::size_t k = 0; k <= n; ++k)
            EXPECT_TRUE(v0111::same_bits(run.trace[k].x, want[k])) << "n " << n << ", vertex " << k;
    }
}

TEST(V0111Batch, EveryShrinkIsOneCallOfNAndEveryOtherPointGoesAlone) {
    struct Config {
        std::size_t n;
        bool adaptive;
    };
    for (const Config cfg : {Config{.n = 2, .adaptive = false}, Config{.n = 2, .adaptive = true},
                             Config{.n = 3, .adaptive = false}, Config{.n = 3, .adaptive = true},
                             Config{.n = 4, .adaptive = false}, Config{.n = 4, .adaptive = true}}) {
        {
            const std::size_t n = cfg.n;
            const bool adaptive = cfg.adaptive;
            const Vec x0(n, 0.0);
            const flop::nelder_mead::Options o = v0111::options(1.0, 100000, adaptive);
            const Traced scalar = v0111::traced(staircase, x0, o);
            v0111::ReferenceRun ref;
            ASSERT_TRUE(
                v0111::audit(x0, o, scalar.trace, scalar.r, false, &ref, true).points_agree);
            ASSERT_GT(ref.shrinks, 0u) << "n " << n << ": the run must shrink to say anything";
            auto f = v0111::batched(staircase);
            const Traced batch = v0111::traced_batch(f, x0, o);
            expect_same_run(scalar, batch, "staircase");
            EXPECT_EQ(f.sizes, expected_calls(n, ref, scalar.trace.size())) << "n " << n;
            EXPECT_EQ(std::accumulate(f.sizes.begin(), f.sizes.end(), std::size_t{0}),
                      batch.r.evaluations);
            for (std::size_t k = 0; k < batch.trace.size(); ++k) EXPECT_EQ(batch.trace[k].index, k);
        }
    }
}

TEST(V0111Batch, ARestartIsOneCallOfN) {
    // The corner scenario of the bounds suite: five single steps onto the
    // corner, two single probes, then the restart's two new vertices in one
    // call.
    const double eps = std::numeric_limits<double>::epsilon();
    Table t;
    t.set({0.0, 0.0}, 1.0).set({1.0, 0.0}, 3.0).set({0.0, 1.0}, 2.0).set({0.0, 0.75}, 1.5);
    t.set({eps, 0.0}, 5.0).set({0.0, eps}, 0.5);
    auto f = v0111::batched([&t](std::span<const double> x) { return t(x); });
    flop::nelder_mead::Options o = v0111::options(1.0, 12);
    o.bounds = flop::Bounds::box(Vec{0.0, 0.0}, Vec{1.0, 1.0});
    o.stopping.xtol_abs = 0.5;
    const Traced run = v0111::traced_batch(f, Vec{0.0, 0.0}, o);
    EXPECT_EQ(f.sizes, (std::vector<std::size_t>{3, 1, 1, 1, 1, 1, 1, 1, 2}));
    EXPECT_EQ(run.r.evaluations, 12u);
}

TEST(V0111Batch, ACapTooSmallForACallSendsThePointsOneAtATime) {
    const std::size_t n = 4;
    const Vec x0(n, 0.0);
    for (std::size_t cap = 1; cap <= n + 1; ++cap) {
        auto f = v0111::batched(v0111::sphere);
        const Traced run = v0111::traced_batch(f, x0, v0111::options(1.0, cap));
        EXPECT_EQ(run.r.evaluations, cap);
        if (cap <= n)
            EXPECT_EQ(f.sizes, std::vector<std::size_t>(cap, 1)) << "cap " << cap;
        else
            EXPECT_EQ(f.sizes, std::vector<std::size_t>{n + 1});
    }
    // A cap that admits only part of a shrink: the shrink's points go one at
    // a time up to the cap, and the run is the scalar run's prefix.
    const flop::nelder_mead::Options o = v0111::options(1.0, 100000, false);
    const Traced scalar = v0111::traced(staircase, x0, o);
    v0111::ReferenceRun ref;
    ASSERT_TRUE(v0111::audit(x0, o, scalar.trace, scalar.r, false, &ref, true).points_agree);
    const v0111::Step* shrink = nullptr;
    for (const v0111::Step& s : ref.steps)
        if (s.termination == v0111::Termination::Shrink) {
            shrink = &s;
            break;
        }
    ASSERT_NE(shrink, nullptr);
    const std::size_t cap = shrink->first_evaluation + 2 + 2;  // two of the n shrink points
    auto f = v0111::batched(staircase);
    const Traced batch = v0111::traced_batch(f, x0, v0111::options(1.0, cap, false));
    EXPECT_EQ(batch.r.evaluations, cap);
    ASSERT_GE(f.sizes.size(), 2u);
    EXPECT_EQ(f.sizes.back(), 1u);
    EXPECT_EQ(f.sizes[f.sizes.size() - 2], 1u);
    for (std::size_t k = 0; k < cap; ++k)
        EXPECT_TRUE(v0111::same_bits(batch.trace[k].x, scalar.trace[k].x)) << k;
}

TEST(V0111Batch, TheBatchAndScalarChannelsRunTheSameTrajectory) {
    struct Case {
        const char* what;
        double (*f)(std::span<const double>);
        Vec x0;
        double h;
    };
    const std::vector<Case> cases{
        {.what = "sphere 8", .f = v0111::sphere, .x0 = Vec(8, 0.0), .h = 1.0},
        {.what = "rosenbrock", .f = v0111::extended_rosenbrock, .x0 = {-1.2, 1.0}, .h = 0.5},
        {.what = "beale", .f = v0111::beale, .x0 = {1.0, 1.0}, .h = 0.5},
        {.what = "powell singular",
         .f = v0111::powell_singular,
         .x0 = {3.0, -1.0, 0.0, 1.0},
         .h = 1.0},
        {.what = "staircase 4", .f = staircase, .x0 = Vec(4, 0.0), .h = 1.0},
    };
    for (const Case& c : cases) {
        for (const bool adaptive : {false, true}) {
            flop::nelder_mead::Options o = v0111::options(c.h, 20000, adaptive);
            o.stopping.xtol_abs = 1e-10;
            auto f = v0111::batched(c.f);
            expect_same_run(v0111::traced(c.f, c.x0, o), v0111::traced_batch(f, c.x0, o), c.what);
        }
    }
}

TEST(V0111Batch, StopValueMetInsideACallCountsTheWholeCall) {
    {
        // The initial simplex at n = 3: the second point meets stop_value, the
        // third is lower still. The scalar channel stops at the second; the
        // batch call has evaluated all four, and they all count.
        Table t;
        t.set({0.0, 0.0, 0.0}, 5.0).set({1.0, 0.0, 0.0}, 1.0).set({0.0, 1.0, 0.0}, 0.5);
        t.set({0.0, 0.0, 1.0}, 3.0);
        flop::nelder_mead::Options o = v0111::options(1.0, 100);
        o.stopping.stop_value = 2.0;
        const Traced scalar = v0111::traced(t, Vec(3, 0.0), o);
        EXPECT_EQ(scalar.r.status, flop::Status::StopValueReached);
        EXPECT_EQ(scalar.r.evaluations, 2u);
        EXPECT_TRUE(v0111::same_bits(scalar.r.x, Vec{1.0, 0.0, 0.0}));
        auto f = v0111::batched([&t](std::span<const double> x) { return t(x); });
        const Traced batch = v0111::traced_batch(f, Vec(3, 0.0), o);
        EXPECT_EQ(batch.r.status, flop::Status::StopValueReached);
        EXPECT_EQ(batch.r.evaluations, 4u);
        EXPECT_EQ(batch.trace.size(), 4u);
        EXPECT_TRUE(v0111::same_bits(batch.r.x, Vec{0.0, 1.0, 0.0}));
        EXPECT_TRUE(v0111::same_bits(batch.r.f, 0.5));
    }
    {
        // A shrink at n = 2: the first new point meets stop_value, the second
        // is lower.
        const Vec s0{0.0, 0.0}, s1{1.0, 0.0}, s2{0.0, 1.0};
        const v0111::Simplex ranked{s0, s1, s2};
        const v0111::Coefficients c = v0111::standard();
        const Vec v1 = v0111::shrink_point(s0, s1, c.sigma);
        const Vec v2 = v0111::shrink_point(s0, s2, c.sigma);
        Table t;
        t.set(s0, 1.0).set(s1, 2.0).set(s2, 3.0);
        t.set(v0111::trial_point(ranked, c.rho), 4.0)
            .set(v0111::trial_point(ranked, -c.gamma), 3.5);
        t.set(v1, 0.75).set(v2, 0.25);
        flop::nelder_mead::Options o = v0111::options(1.0, 100);
        o.stopping.stop_value = 0.8;
        const Traced scalar = v0111::traced(t, s0, o);
        EXPECT_EQ(scalar.r.evaluations, 6u);
        EXPECT_TRUE(v0111::same_bits(scalar.r.x, v1));
        auto f = v0111::batched([&t](std::span<const double> x) { return t(x); });
        const Traced batch = v0111::traced_batch(f, s0, o);
        EXPECT_EQ(batch.r.status, flop::Status::StopValueReached);
        EXPECT_EQ(batch.r.evaluations, 7u);
        EXPECT_TRUE(v0111::same_bits(batch.r.x, v2));
        EXPECT_EQ(f.sizes.back(), 2u);
    }
}

TEST(V0111Batch, AStrictlyConvexFunctionNeverSendsAShrinkCall) {
    // Lemma 3.5 of Lagarias et al., seen from the caller: after the initial
    // simplex every call is one point.
    for (const std::size_t n : {std::size_t{2}, std::size_t{4}, std::size_t{8}, std::size_t{16}}) {
        for (const bool adaptive : {false, true}) {
            flop::nelder_mead::Options o = v0111::options(1.0, 200000, adaptive);
            o.stopping.xtol_abs = 1e-7;
            for (auto* g : {&v0111::sphere, &coupled}) {
                auto f = v0111::batched(g);
                const Traced run = v0111::traced_batch(f, Vec(n, 0.0), o);
                EXPECT_EQ(run.r.status, flop::Status::XtolReached);
                for (std::size_t k = 1; k < f.sizes.size(); ++k)
                    ASSERT_EQ(f.sizes[k], 1u) << "n " << n << ", call " << k;
            }
        }
    }
}
