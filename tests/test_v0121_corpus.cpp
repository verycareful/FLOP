// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: BOBYQA on the problems of the literature.
//
// The documented examples do what their comments say; the More, Garbow and
// Hillstrom functions of Gao and Han's Table 2 that have a closed-form
// minimiser reach it; the sphere is solved in every dimension; and the report's
// own test problems of section 7 are run: the trigonometric sum of squares
// (7.1), five instances each at n = 10 and n = 20 with m = 2n + 1, n + 6 and
// (n + 1)(n + 2)/2, rho_beg = 0.1 and rho_end = 1e-6, against the report's
// Tables 1 to 3, and the points in a square (7.2) to (7.4).
//
// The instances of (7.1) are drawn as the report describes: S_ij and C_ij
// random integers in [-100, 100], sigma_j in [1, 10], x* random, f_i set by
// F(x*) = 0, and (x0 - x*)_j / sigma_j in [-pi/10, pi/10]. They are not the
// report's random numbers, which are not published, so the evaluation counts
// are held to the report's greatest count in each row with a quarter more
// allowed, and the accuracy to three times the greatest accuracy of any row,
// 1.9e-5. The error in x at the end is not bounded by rho_end: the Hessian of
// the sum is nearly singular in some directions, more so with few points, and
// the run is sensitive to rounding. Over 120 instances of the row at n = 20
// and m = n + 6 the error had a median of 6e-6 and a greatest value of 3.3e-5,
// and a different compiler's rounding of one of the five instances here
// reached 4.0e-5, so a bound near the report's own would fail one instance in
// about a hundred on any platform.
//
// There is no NLopt recording of BOBYQA among the files in tests/data/
// nlopt-corpus, which are COBYLA's; none is made up.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

#include "flop/flop.hpp"
#include "v0111_problems.hpp"
#include "v0121_problems.hpp"
#include "v0121_reference.hpp"

namespace {

using v0121::Draw;
using v0121::Vec;

// The report's (7.1).
struct TrigSum {
    std::size_t n = 0;
    std::vector<double> s, c, sigma, f, xstar;
    [[nodiscard]] double operator()(std::span<const double> x) const {
        double total = 0.0;
        for (std::size_t i = 0; i < 2 * n; ++i) {
            double v = f[i];
            for (std::size_t j = 0; j < n; ++j)
                v -= s[i * n + j] * std::sin(x[j] / sigma[j]) +
                     c[i * n + j] * std::cos(x[j] / sigma[j]);
            total += v * v;
        }
        return total;
    }
};

TrigSum make_trig(std::size_t n, std::uint64_t seed, Vec& x0) {
    Draw rng(seed);
    const auto uniform = [&rng](double lo, double hi) {
        return lo + (hi - lo) * 0.5 * (rng.next() + 1.0);
    };
    TrigSum t;
    t.n = n;
    t.s.resize(2 * n * n);
    t.c.resize(2 * n * n);
    t.sigma.resize(n);
    t.f.assign(2 * n, 0.0);
    t.xstar.resize(n);
    for (double& v : t.s) v = std::floor(uniform(-100.0, 101.0));
    for (double& v : t.c) v = std::floor(uniform(-100.0, 101.0));
    for (double& v : t.sigma) v = uniform(1.0, 10.0);
    for (std::size_t j = 0; j < n; ++j) t.xstar[j] = 3.0 * t.sigma[j] * rng.next();
    for (std::size_t i = 0; i < 2 * n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            t.f[i] += t.s[i * n + j] * std::sin(t.xstar[j] / t.sigma[j]) +
                      t.c[i * n + j] * std::cos(t.xstar[j] / t.sigma[j]);
    x0.resize(n);
    for (std::size_t j = 0; j < n; ++j)
        x0[j] =
            t.xstar[j] + t.sigma[j] * uniform(-std::numbers::pi / 10.0, std::numbers::pi / 10.0);
    return t;
}

struct Row {
    std::size_t evaluations = 0;
    double error = 0.0;
    bool converged = false;
};

// Three times the greatest accuracy of any row of the report's Tables 1 to 3.
constexpr double kTrigAccuracy = 3.0 * 1.9e-5;

Row solve_trig(const TrigSum& t, const Vec& x0, std::size_t m, bool newuoa) {
    flop::bobyqa::Options o;
    o.initial_step = 0.1;
    o.final_trust_radius = 1e-6;
    o.stopping.max_evaluations = 20000;
    o.interpolation_points = m;
    o.newuoa_weight = newuoa;
    const flop::Result r = flop::bobyqa::minimize(t, x0, o);
    Row row;
    row.evaluations = r.evaluations;
    row.converged = flop::converged(r.status);
    for (std::size_t j = 0; j < t.n; ++j)
        row.error = std::max(row.error, std::fabs(r.x[j] - t.xstar[j]));
    return row;
}

}  // namespace

TEST(V0121Corpus, TheHeaderExampleRunsAndFindsTheMinimumOfRosenbrock) {
    // The example in include/flop/bobyqa.hpp.
    auto rosenbrock = [](std::span<const double> x) {
        const double a = 1.0 - x[0], b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    };
    flop::bobyqa::Options opts;
    opts.stopping.xtol_abs = 1e-8;
    opts.stopping.max_evaluations = 2000;
    opts.initial_step = 0.5;
    const double x0[2] = {-1.2, 1.0};
    const flop::Result r = flop::bobyqa::minimize(rosenbrock, x0, opts);
    EXPECT_EQ(r.status, flop::Status::XtolReached);
    EXPECT_NEAR(r.x[0], 1.0, 1e-5);
    EXPECT_NEAR(r.x[1], 1.0, 1e-5);
    EXPECT_LT(r.f, 1e-10);
}

TEST(V0121Corpus, TheApiPageExampleReturnsZero) {
    // The example in docs/api/bobyqa.md: Rosenbrock in a box.
    const auto rosenbrock = [](std::span<const double> x) {
        const double a = 1.0 - x[0], b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    };
    flop::bobyqa::Options opts;
    opts.stopping.xtol_abs = 1e-8;
    opts.stopping.max_evaluations = 2000;
    opts.initial_step = 0.5;
    const double lo[2] = {-2.0, -2.0};
    const double hi[2] = {2.0, 2.0};
    opts.bounds = flop::Bounds::box(lo, hi);
    const double x0[2] = {-1.2, 1.0};
    const flop::Result r = flop::bobyqa::minimize(rosenbrock, x0, opts);
    EXPECT_TRUE(flop::converged(r.status));
    EXPECT_NEAR(r.x[0], 1.0, 1e-5);
    EXPECT_NEAR(r.x[1], 1.0, 1e-5);
}

TEST(V0121Corpus, TheSphereIsSolvedInEveryDimension) {
    for (const std::size_t n :
         {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{5}, std::size_t{8},
          std::size_t{12}, std::size_t{20}, std::size_t{32}}) {
        flop::bobyqa::Options o = v0121::options_xtol(0.5, 1e-10, 20000);
        const flop::Result r = flop::bobyqa::minimize(v0111::sphere, Vec(n, 0.0), o);
        EXPECT_EQ(r.status, flop::Status::XtolReached) << "n " << n;
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(r.x[i], v0111::sphere_centre(i), 1e-7) << "n " << n << " coordinate " << i;
    }
}

TEST(V0121Corpus, TheMoreGarbowHillstromProblemsReachTheirMinimisers) {
    for (const v0111::Mgh& p : v0111::mgh_problems()) {
        flop::bobyqa::Options o = v0121::options_xtol(0.5, 1e-9, 20000);
        const flop::Result r = flop::bobyqa::minimize(p.f, p.x0, o);
        EXPECT_TRUE(flop::converged(r.status)) << p.name << " " << flop::to_string(r.status);
        // Powell's singular function has a singular Hessian at its minimiser,
        // where the error in x goes as the fourth root of the error in f.
        const double tolerance = std::string(p.name) == "powell singular" ? 1e-3 : 1e-6;
        EXPECT_LT(v0121::max_abs_diff(r.x, p.x_opt), tolerance) << p.name;
        EXPECT_LT(r.f, 1e-10) << p.name;
    }
}

TEST(V0121Corpus, TheTrigonometricSumOfSquaresOfSection7MatchesTheReportsTables) {
    // (7.1) with the report's settings. The report's greatest numbers of
    // evaluations of its five instances (Tables 1 to 3) are 427, 927 for
    // m = 2n + 1, 637, 1706 for m = n + 6, and 254, 853 for the full m at
    // n = 10 and 20; a quarter more is allowed for instances that are not its
    // own. The accuracy of every row of the tables is at most 1.9e-5, and
    // three times that is allowed (see the file header).
    struct Row1 {
        std::size_t n, m_kind, paper_max;  // m_kind: 0 = 2n + 1, 1 = n + 6, 2 = full
    };
    const Row1 rows[] = {
        {.n = 10, .m_kind = 0, .paper_max = 427}, {.n = 20, .m_kind = 0, .paper_max = 927},
        {.n = 10, .m_kind = 1, .paper_max = 637}, {.n = 20, .m_kind = 1, .paper_max = 1706},
        {.n = 10, .m_kind = 2, .paper_max = 254}, {.n = 20, .m_kind = 2, .paper_max = 853}};
    for (const Row1& row : rows) {
        const std::size_t n = row.n;
        const std::size_t m = row.m_kind == 0   ? 2 * n + 1
                              : row.m_kind == 1 ? n + 6
                                                : (n + 1) * (n + 2) / 2;
        for (std::uint64_t seed = 1; seed <= 5; ++seed) {
            Vec x0;
            const TrigSum t = make_trig(n, seed * 7919 + n, x0);
            const Row r = solve_trig(t, x0, m, true);
            EXPECT_TRUE(r.converged) << "n " << n << " m " << m << " seed " << seed;
            EXPECT_LE(r.evaluations, row.paper_max * 5 / 4)
                << "n " << n << " m " << m << " seed " << seed;
            EXPECT_LE(r.error, kTrigAccuracy) << "n " << n << " m " << m << " seed " << seed;
        }
    }
}

TEST(V0121Corpus, TheReportsWeightNeedsMoreEvaluationsThanNewuoasOnTheTrigonometricSum) {
    // The reason for the choice of weight (docs/algorithms/bobyqa.md,
    // "Deviations"): over the same instances, the report's weight (6.1) takes more
    // evaluations in total than NEWUOA's, and both reach the minimiser.
    for (const std::size_t n : {std::size_t{10}, std::size_t{20}}) {
        std::size_t with_newuoa = 0, with_report = 0;
        for (std::uint64_t seed = 1; seed <= 5; ++seed) {
            Vec x0;
            const TrigSum t = make_trig(n, seed * 7919 + n, x0);
            const Row a = solve_trig(t, x0, 2 * n + 1, true);
            const Row b = solve_trig(t, x0, 2 * n + 1, false);
            EXPECT_TRUE(a.converged);
            EXPECT_TRUE(b.converged);
            EXPECT_LE(b.error, kTrigAccuracy);
            with_newuoa += a.evaluations;
            with_report += b.evaluations;
        }
        EXPECT_GT(with_report, with_newuoa) << "n " << n;
    }
}

namespace {

// (7.3): the sum over pairs of points of min(1 / |p_i - p_j|, 1000).
double points_in_square(std::span<const double> x) {
    double total = 0.0;
    const std::size_t pts = x.size() / 2;
    for (std::size_t i = 1; i < pts; ++i)
        for (std::size_t j = 0; j < i; ++j) {
            const double dx = x[2 * i] - x[2 * j], dy = x[2 * i + 1] - x[2 * j + 1];
            const double dist = std::sqrt(dx * dx + dy * dy);
            total += dist > 1e-3 ? 1.0 / dist : 1e3;
        }
    return total;
}

}  // namespace

TEST(V0121Corpus, PointsInASquareStayInTheSquareAndTheObjectiveFalls) {
    // (7.2) to (7.4): n = 20 variables, the bounds 0 <= x_i <= 1, from a
    // pseudo-random start. The report notes many local minima; what is held is
    // what it requires of any run: every evaluation inside the square, and an
    // objective that does not rise.
    Draw rng(0x5C0A4E5ULL);
    for (const std::size_t m : {std::size_t{41}, std::size_t{26}}) {
        Vec x0(20);
        for (double& v : x0) v = 0.5 * (rng.next() + 1.0);
        flop::bobyqa::Options o;
        o.initial_step = 0.1;
        o.stopping.xtol_abs = 1e-6;
        o.stopping.max_evaluations = 20000;
        o.interpolation_points = m;
        o.bounds = flop::Bounds::box(Vec(20, 0.0), Vec(20, 1.0));
        v0121::Trace trace;
        o.on_evaluation = v0121::recorder(trace);
        const flop::Result r = flop::bobyqa::minimize(points_in_square, x0, o);
        EXPECT_TRUE(flop::converged(r.status)) << "m " << m << " " << flop::to_string(r.status);
        for (const auto& p : trace)
            for (const double v : p.x) {
                EXPECT_GE(v, 0.0);
                EXPECT_LE(v, 1.0);
            }
        EXPECT_LE(r.f, points_in_square(x0)) << "m " << m;
    }
}
