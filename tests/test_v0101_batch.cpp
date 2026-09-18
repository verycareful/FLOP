// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.0.1 test wave: the batch channel.
//
// minimize_batch exists so a caller who can evaluate several points at once
// (n + 1 circuits in parallel) gets them in one call. The contract is on
// the shape of the calls: the first call carries exactly x0 and the n
// initial vertices, every later call carries one point, and the count in
// Result is the number of points, not the number of calls. The trajectory
// is not the scalar one (the initial simplex cannot walk its base when the
// values are not yet known), so the two entry points are compared at the
// optimum, not along the way.

#include <gtest/gtest.h>

#include <cstddef>
#include <span>
#include <vector>

#include "flop/flop.hpp"
#include "v0101_problems.hpp"

namespace {

struct BatchSphere {
    std::vector<std::size_t> sizes;  // points per call, in call order
    std::size_t points = 0;
    void operator()(std::span<const std::span<const double>> xs, std::span<double> out) {
        sizes.push_back(xs.size());
        points += xs.size();
        for (std::size_t i = 0; i < xs.size(); ++i) out[i] = v0101::sphere(xs[i]);
    }
};

}  // namespace

TEST(V0101Batch, TheFirstCallIsTheWholeInitialSimplex) {
    const std::size_t n = 16;
    const std::vector<double> x0(n, 0.0);
    BatchSphere f;
    const flop::Result r = flop::cobyla::minimize_batch(f, x0, v0101::options(0.3, 1e-9, 5000));
    ASSERT_FALSE(f.sizes.empty());
    EXPECT_EQ(f.sizes[0], n + 1);
    for (std::size_t k = 1; k < f.sizes.size(); ++k) EXPECT_EQ(f.sizes[k], 1u) << "call " << k;
    EXPECT_EQ(f.points, r.evaluations);
}

TEST(V0101Batch, TheFirstCallHoldsX0AndTheCoordinatePokes) {
    const std::vector<double> x0{0.1, 0.2, 0.3};
    std::vector<std::vector<double>> first;
    auto f = [&](std::span<const std::span<const double>> xs, std::span<double> out) {
        if (first.empty())
            for (const auto& x : xs) first.emplace_back(x.begin(), x.end());
        for (std::size_t i = 0; i < xs.size(); ++i) out[i] = v0101::sphere(xs[i]);
    };
    (void)flop::cobyla::minimize_batch(f, x0, v0101::options(0.3, 1e-9, 1000));
    ASSERT_EQ(first.size(), 4u);
    EXPECT_TRUE(v0101::same_bits(first[0], x0));
    for (std::size_t i = 0; i < 3; ++i) {
        std::vector<double> poke = x0;
        poke[i] += 0.3;
        EXPECT_TRUE(v0101::same_bits(first[i + 1], poke)) << "vertex " << i;
    }
}

TEST(V0101Batch, ReachesTheScalarOptimumOnTheStandardProblems) {
    struct Case {
        const char* name;
        double (*f)(std::span<const double>);
        std::vector<double> x0;
        double step;
        double xtol;
    };
    const std::vector<Case> cases{
        {.name = "sphere 16d",
         .f = v0101::sphere,
         .x0 = std::vector<double>(16, 0.0),
         .step = 0.3,
         .xtol = 1e-6},
        {.name = "rosenbrock 2d",
         .f = v0101::rosenbrock,
         .x0 = {-1.2, 1.0},
         .step = 0.5,
         .xtol = 1e-3},
        {.name = "beale", .f = v0101::beale, .x0 = {1.0, 1.0}, .step = 0.5, .xtol = 1e-3},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        auto fb = [&c](std::span<const std::span<const double>> xs, std::span<double> out) {
            for (std::size_t i = 0; i < xs.size(); ++i) out[i] = c.f(xs[i]);
        };
        const flop::Result scalar =
            flop::cobyla::minimize(c.f, c.x0, v0101::options(c.step, 1e-9, 50000));
        const flop::Result batch =
            flop::cobyla::minimize_batch(fb, c.x0, v0101::options(c.step, 1e-9, 50000));
        EXPECT_TRUE(flop::converged(scalar.status)) << flop::to_string(scalar.status);
        EXPECT_TRUE(flop::converged(batch.status)) << flop::to_string(batch.status);
        EXPECT_LE(v0101::max_abs_diff(scalar.x, batch.x), c.xtol);
    }
}

TEST(V0101Batch, ConstrainedBatchReachesTheConstrainedOptimum) {
    const v0101::Constrained p = v0101::powell_1994_problems()[7];  // Rosen-Suzuki
    auto fb = [&p](std::span<const std::span<const double>> xs, std::span<double> out) {
        for (std::size_t i = 0; i < xs.size(); ++i) out[i] = p.f(xs[i]);
    };
    const flop::Result r =
        flop::cobyla::minimize_batch(fb, p.c, p.m, p.x0, v0101::options(0.5, 2e-9, 20000));
    EXPECT_TRUE(flop::converged(r.status)) << flop::to_string(r.status);
    EXPECT_NEAR(r.f, p.f_opt, 1e-8);
    EXPECT_LE(v0101::max_abs_diff(r.x, p.x_opt), 1e-5);
    EXPECT_LE(r.max_constraint_violation, 1e-9);
}

TEST(V0101Batch, TheTraceSeesEveryBatchPointSinglyAndInOrder) {
    const std::size_t n = 8;
    const std::vector<double> x0(n, 0.0);
    std::vector<std::size_t> indices;
    flop::cobyla::Options o = v0101::options(0.3, 1e-9, 2000);
    o.on_evaluation = [&](const flop::Evaluation& e) { indices.push_back(e.index); };
    BatchSphere f;
    const flop::Result r = flop::cobyla::minimize_batch(f, x0, o);
    ASSERT_EQ(indices.size(), r.evaluations);
    for (std::size_t k = 0; k < indices.size(); ++k) EXPECT_EQ(indices[k], k);
}

TEST(V0101Batch, ACapBelowTheSimplexIsHonouredExactly) {
    // A batch is never split, so with a cap smaller than n + 1 the initial
    // simplex goes out one point per call and the cap lands exactly.
    const std::size_t n = 8;
    const std::vector<double> x0(n, 0.0);
    BatchSphere f;
    const flop::Result r = flop::cobyla::minimize_batch(f, x0, v0101::options(0.3, 1e-9, 4));
    EXPECT_EQ(r.status, flop::Status::MaxEvaluationsReached);
    EXPECT_EQ(r.evaluations, 4u);
    EXPECT_EQ(f.points, 4u);
    for (std::size_t k = 0; k < f.sizes.size(); ++k) EXPECT_EQ(f.sizes[k], 1u) << "call " << k;
}
