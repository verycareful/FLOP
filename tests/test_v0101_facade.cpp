// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.0.1 test wave: the Minimizer facade and the version string.
//
// The facade is the same algorithm behind a name and a std::function,
// compiled once into the library under the library's floating-point flags.
// When the test binary and the library share a model, the template entry
// point and the facade run the same code the same way and are held to the
// same trajectory to the bit. When they differ (the fast binary against a
// strict library, or the strict binary against a library built with
// FLOP_FAST_MATH=ON), they are held to the same optimum within tolerance
// and to the same status. Which case this is comes from the build.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "flop/flop.hpp"
#include "v0101_problems.hpp"

namespace {

// The test binary's model against the library's: FLOP_TEST_FAST_MATH says
// this binary is the fast one, FLOP_TEST_LIBRARY_FAST_MATH that the library
// was built with FLOP_FAST_MATH=ON. Same model on both sides means the same
// trajectory to the bit.
#if defined(FLOP_TEST_FAST_MATH) == defined(FLOP_TEST_LIBRARY_FAST_MATH)
constexpr bool kSameModelAsLibrary = true;
#else
constexpr bool kSameModelAsLibrary = false;
#endif

}  // namespace

TEST(V0101Facade, NamesListsCobylaAndCreateAcceptsIt) {
    const auto names = flop::Minimizer::names();
    EXPECT_NE(std::ranges::find(names, std::string_view("COBYLA")), names.end());
    const flop::Minimizer m = flop::Minimizer::create("COBYLA");
    EXPECT_EQ(m.name(), std::string_view("COBYLA"));
}

TEST(V0101Facade, EveryListedNameCreates) {
    for (const std::string_view name : flop::Minimizer::names())
        EXPECT_NO_THROW((void)flop::Minimizer::create(name)) << name;
}

TEST(V0101Facade, TheFacadeAndTheTemplateAgree) {
    // Rosenbrock at a cap for the bit comparison (same model, same code,
    // same trajectory, wherever the cap lands); the 16-d sphere, which
    // converges, for the comparison across models, since two capped runs
    // under different rounding are two different points and say nothing.
    const flop::Minimizer m = flop::Minimizer::create("COBYLA");
    if (kSameModelAsLibrary) {
        const std::vector<double> x0{-1.2, 1.0};
        std::vector<std::vector<double>> a_points, b_points;
        flop::cobyla::Options oa = v0101::options(0.5, 1e-9, 5000);
        flop::cobyla::Options ob = oa;
        oa.on_evaluation = [&](const flop::Evaluation& e) {
            a_points.emplace_back(e.x.begin(), e.x.end());
        };
        ob.on_evaluation = [&](const flop::Evaluation& e) {
            b_points.emplace_back(e.x.begin(), e.x.end());
        };
        const flop::Result a = flop::cobyla::minimize(v0101::rosenbrock, x0, oa);
        const flop::Minimizer::Objective fo = v0101::rosenbrock;
        const flop::Result b = m.minimize(fo, x0, ob);
        EXPECT_EQ(a.status, b.status);
        ASSERT_EQ(a_points.size(), b_points.size());
        for (std::size_t k = 0; k < a_points.size(); ++k)
            ASSERT_TRUE(v0101::same_bits(a_points[k], b_points[k])) << "evaluation " << k;
        EXPECT_TRUE(v0101::same_bits(a.x, b.x));
        EXPECT_TRUE(v0101::same_bits(a.f, b.f));
        EXPECT_EQ(a.evaluations, b.evaluations);
    }
    const std::size_t n = 16;
    const std::vector<double> x0(n, 0.0);
    const flop::cobyla::Options o = v0101::options(0.3, 1e-9, 5000);
    const flop::Result a = flop::cobyla::minimize(v0101::sphere, x0, o);
    const flop::Minimizer::Objective fo = v0101::sphere;
    const flop::Result b = m.minimize(fo, x0, o);
    EXPECT_EQ(a.status, b.status);
    EXPECT_TRUE(flop::converged(a.status)) << flop::to_string(a.status);
    EXPECT_LE(v0101::max_abs_diff(a.x, b.x), 1e-6);
    EXPECT_NEAR(a.f, b.f, 1e-10);
}

TEST(V0101Facade, ConstrainedThroughTheFacade) {
    const v0101::Constrained p = v0101::powell_1994_problems()[7];  // Rosen-Suzuki
    const flop::Minimizer m = flop::Minimizer::create("COBYLA");
    const flop::Minimizer::Objective fo = p.f;
    const flop::Minimizer::Constraints co = p.c;
    const flop::Result r = m.minimize(fo, co, p.m, p.x0, v0101::options(0.5, 2e-9, 20000));
    EXPECT_TRUE(flop::converged(r.status)) << flop::to_string(r.status);
    EXPECT_NEAR(r.f, p.f_opt, 1e-8);
    EXPECT_LE(v0101::max_abs_diff(r.x, p.x_opt), 1e-5);
    EXPECT_LE(r.max_constraint_violation, 1e-9);
}

TEST(V0101Facade, BatchThroughTheFacade) {
    const std::size_t n = 8;
    const std::vector<double> x0(n, 0.0);
    std::vector<std::size_t> sizes;
    const flop::Minimizer::BatchObjective fb = [&](std::span<const std::span<const double>> xs,
                                                   std::span<double> out) {
        sizes.push_back(xs.size());
        for (std::size_t i = 0; i < xs.size(); ++i) out[i] = v0101::sphere(xs[i]);
    };
    const flop::Minimizer m = flop::Minimizer::create("COBYLA");
    const flop::Result r = m.minimize(fb, x0, v0101::options(0.3, 1e-9, 5000));
    ASSERT_FALSE(sizes.empty());
    EXPECT_EQ(sizes[0], n + 1);
    EXPECT_TRUE(flop::converged(r.status)) << flop::to_string(r.status);
    std::vector<double> x_opt(n);
    for (std::size_t i = 0; i < n; ++i) x_opt[i] = v0101::sphere_centre(i);
    EXPECT_LE(v0101::max_abs_diff(r.x, x_opt), 1e-7);
}

TEST(V0101Facade, SetFinalTrustRadiusReachesTheSolver) {
    const std::vector<double> x0{0.0, 0.0};
    flop::Minimizer m = flop::Minimizer::create("COBYLA");
    m.set_final_trust_radius(1e-2);
    const flop::Minimizer::Objective fo = v0101::sphere;
    const flop::Result r = m.minimize(fo, x0, v0101::options(0.3, 1e-9, 5000));
    EXPECT_EQ(r.status, flop::Status::XtolReached);
    EXPECT_NEAR(r.final_trust_radius, 1e-2, 0.5e-2);
}

TEST(V0101Facade, AMinimizerIsMovable) {
    flop::Minimizer a = flop::Minimizer::create("COBYLA");
    flop::Minimizer b = std::move(a);
    EXPECT_EQ(b.name(), std::string_view("COBYLA"));
    flop::Minimizer c = flop::Minimizer::create("COBYLA");
    c = std::move(b);
    EXPECT_EQ(c.name(), std::string_view("COBYLA"));
}

TEST(V0101Facade, TheVersionIsTheFourComponentLabel) {
    const std::string_view v = flop::version();
    EXPECT_EQ(std::ranges::count(v, '.'), 3);
    EXPECT_FALSE(v.empty());
    for (const char ch : v) EXPECT_TRUE((ch >= '0' && ch <= '9') || ch == '.') << ch;
}
