// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: BOBYQA's trust-region step, TRSBOX.
//
// Powell, DAMTP 2009/NA06, section 3, first half. The step d approximately
// minimises Q(x_k + d) in the box and the trust region, by an active-set
// truncated conjugate gradient method from d = 0 and then by moves round the
// trust-region boundary. The rules held here: the initial active set (3.3),
// the line search of CG (the least of alpha_B, alpha_Delta and alpha_Q), the
// restart after a bound and its termination test (3.4), the termination test
// (3.5) and the arc (3.6) of the move round the boundary, and what the report
// promises of the result: a step inside the box and the trust region, on a
// bound exactly where the index is in the active set, and a reduction that is
// the one the model gives.
//
// The report asks for an approximate solution, and the method is one that
// never removes an index from the active set and stops at a stationary
// point, so a step is not held to the global minimiser of a nonconvex or
// bounded problem. It is held to the minimiser where the report's method finds
// it: a convex quadratic with the Newton step inside the radius, an
// isotropic one that runs to the boundary, a diagonal one on the boundary,
// and the bound cases of closed form below. Random problems are held to the
// properties that must hold whatever the problem is.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "flop/detail/bobyqa_model.hpp"
#include "flop/detail/bobyqa_trsbox.hpp"
#include "v0121_problems.hpp"
#include "v0121_reference.hpp"

namespace {

using flop::detail::bobyqa::Box;
using flop::detail::bobyqa::Model;
using flop::detail::bobyqa::Side;
using flop::detail::bobyqa::trsbox;
using flop::detail::bobyqa::TrsboxResult;
using flop::detail::bobyqa::TrsboxWork;
using v0121::Draw;

// A model with no interpolation point but x_k, the second derivative matrix h
// (n x n, symmetric) and the gradient g, and the box as offsets from the
// origin, x_k included.
struct Problem {
    std::size_t n;
    Model q;
    Box box;
    Problem(std::size_t n_, const std::vector<double>& xk, const std::vector<double>& g,
            const std::vector<double>& h)
        : n(n_), q(n_, 1), box(n_) {
        for (std::size_t i = 0; i < n; ++i) {
            q.points[i] = xk[i];
            q.g[i] = g[i];
        }
        for (std::size_t i = 0; i < n * n; ++i) q.hq[i] = h[i];
        q.kopt = 0;
    }
    void lower(std::size_t i, double v) {
        box.has_lo[i] = 1;
        box.lo[i] = v;
    }
    void upper(std::size_t i, double v) {
        box.has_hi[i] = 1;
        box.hi[i] = v;
    }
};

struct Step {
    std::vector<double> d, gd;
    std::vector<Side> side;
    TrsboxResult r;
};

Step solve(const Problem& p, double delta) {
    Step s;
    s.d.assign(p.n, 0.0);
    s.gd.assign(p.n, 0.0);
    s.side.assign(p.n, Side::Free);
    TrsboxWork w(p.n);
    s.r = trsbox(p.q, p.box, delta, s.d, s.side, s.gd, w);
    return s;
}

// Q(x_k + d) - Q(x_k) = g.d + (1/2) d^T H d.
double model_change(const Problem& p, const std::vector<double>& d) {
    double s = 0.0;
    for (std::size_t i = 0; i < p.n; ++i) {
        s += p.q.g[i] * d[i];
        for (std::size_t k = 0; k < p.n; ++k) s += 0.5 * d[i] * p.q.hq[i * p.n + k] * d[k];
    }
    return s;
}

double norm(const std::vector<double>& v) {
    double s = 0.0;
    for (const double x : v) s += x * x;
    return std::sqrt(s);
}

// Solves H x = b by Gaussian elimination with partial pivoting (H small and
// symmetric positive definite in every use).
std::vector<double> solve_linear(std::vector<double> h, std::vector<double> b) {
    const std::size_t n = b.size();
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t piv = c;
        for (std::size_t r = c + 1; r < n; ++r)
            if (std::fabs(h[r * n + c]) > std::fabs(h[piv * n + c])) piv = r;
        for (std::size_t k = 0; k < n; ++k) std::swap(h[c * n + k], h[piv * n + k]);
        std::swap(b[c], b[piv]);
        for (std::size_t r = c + 1; r < n; ++r) {
            const double f = h[r * n + c] / h[c * n + c];
            for (std::size_t k = c; k < n; ++k) h[r * n + k] -= f * h[c * n + k];
            b[r] -= f * b[c];
        }
    }
    std::vector<double> x(n);
    for (std::size_t c = n; c-- > 0;) {
        double s = b[c];
        for (std::size_t k = c + 1; k < n; ++k) s -= h[c * n + k] * x[k];
        x[c] = s / h[c * n + c];
    }
    return x;
}

std::vector<double> identity2() {
    return {1.0, 0.0, 0.0, 1.0};
}

}  // namespace

TEST(V0121Trsbox, TheInitialActiveSetIsTheCoordinatesAtABoundWithTheGradientPointingOut) {
    // (3.3): at a lower bound with g_i >= 0, or at an upper bound with
    // g_i <= 0, d_i is fixed at zero; equality in the gradient counts.
    const std::vector<double> h = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    Problem p(3, {0.0, 1.0, 0.5}, {0.0, 0.0, -1.0}, h);
    p.lower(0, 0.0);
    p.upper(1, 1.0);
    p.lower(2, 0.0);
    p.upper(2, 2.0);
    const Step s = solve(p, 10.0);
    EXPECT_EQ(s.side[0], Side::Lower);
    EXPECT_EQ(s.side[1], Side::Upper);
    EXPECT_EQ(s.d[0], 0.0);
    EXPECT_EQ(s.d[1], 0.0);
    EXPECT_NEAR(s.d[2], 1.0, 1e-12) << "the free coordinate goes to its minimiser";

    // A gradient pointing into the box leaves the coordinate free: d_0 > 0.
    Problem q(3, {0.0, 1.0, 0.5}, {-1.0, 1.0, -1.0}, h);
    q.lower(0, 0.0);
    q.upper(1, 1.0);
    const Step t = solve(q, 10.0);
    EXPECT_NEAR(t.d[0], 1.0, 1e-12);
    EXPECT_NEAR(t.d[1], -1.0, 1e-12);
    // A coordinate at x_k = bound with no bound on that side is free.
    Problem r(3, {0.0, 1.0, 0.5}, {1.0, -1.0, -1.0}, h);
    r.upper(0, 5.0);
    r.lower(1, -5.0);
    const Step u = solve(r, 10.0);
    EXPECT_EQ(u.side[0], Side::Free);
    EXPECT_NEAR(u.d[0], -1.0, 1e-12);
    EXPECT_NEAR(u.d[1], 1.0, 1e-12);
}

TEST(V0121Trsbox, WhenEveryCoordinateIsInTheInitialActiveSetTheStepIsZero) {
    Problem p(2, {0.0, 1.0}, {0.5, -0.25}, identity2());
    p.lower(0, 0.0);
    p.upper(1, 1.0);
    const Step s = solve(p, 3.0);
    EXPECT_EQ(s.d[0], 0.0);
    EXPECT_EQ(s.d[1], 0.0);
    EXPECT_EQ(s.r.qred, 0.0);
    EXPECT_EQ(s.side[0], Side::Lower);
    EXPECT_EQ(s.side[1], Side::Upper);
    EXPECT_EQ(s.gd[0], 0.5);
    EXPECT_EQ(s.gd[1], -0.25);
    EXPECT_FALSE(s.r.have_curvature);
}

TEST(V0121Trsbox, ANewtonStepInsideTheRadiusIsTheMinimiserOfAConvexQuadratic) {
    // CG on a quadratic with n free coordinates is exact in n steps, so the
    // limit of one step per free coordinate leaves the minimiser reachable.
    Draw rng(0x7A5B0ULL);
    for (std::size_t n = 1; n <= 4; ++n)
        for (int trial = 0; trial < 20; ++trial) {
            // H = A^T A + I from a random A, so that it is positive definite.
            std::vector<double> a(n * n), h(n * n, 0.0), g(n), xk(n, 0.0);
            for (double& v : a) v = 0.7 * rng.next();
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t k = 0; k < n; ++k) {
                    for (std::size_t l = 0; l < n; ++l) h[i * n + k] += a[l * n + i] * a[l * n + k];
                    if (i == k) h[i * n + k] += 1.0;
                }
            for (double& v : g) v = 0.5 * rng.next();
            const Problem p(n, xk, g, h);
            const std::vector<double> newton = solve_linear(h, g);
            const double radius = 2.0 * norm(newton) + 0.1;
            const Step s = solve(p, radius);
            double qstar = 0.0;
            for (std::size_t i = 0; i < n; ++i) qstar += 0.5 * g[i] * newton[i];
            // (3.4) ends the conjugate gradient steps once a step reduces Q by
            // no more than 1 percent of the reduction so far, so the reduction
            // is within that of the optimum, and since H >= I the step is within
            // sqrt(2 (Q - Q*)) of the Newton step.
            EXPECT_GE(s.r.qred, 0.99 * qstar) << "n " << n << " trial " << trial;
            EXPECT_LE(s.r.qred, qstar * (1.0 + 1e-9));
            double err = 0.0;
            for (std::size_t i = 0; i < n; ++i) err += (s.d[i] + newton[i]) * (s.d[i] + newton[i]);
            EXPECT_LE(std::sqrt(err), std::sqrt(0.02 * qstar) + 1e-12)
                << "n " << n << " trial " << trial;
            if (n == 1) {
                EXPECT_NEAR(s.d[0], -newton[0], 1e-12) << "one coordinate is one exact step";
            }
        }
}

TEST(V0121Trsbox, AnIsotropicModelRunsStraightToTheBoundaryOfTheTrustRegion) {
    // Q = g.d + (h/2)|d|^2 with |g| / h above the radius: the solution is
    // -Delta g / |g|, whatever the sign of h, and CG reaches it in one step.
    for (const double hh : {2.0, 0.5, 0.0, -1.0}) {
        for (const std::size_t n : {std::size_t{1}, std::size_t{3}}) {
            std::vector<double> h(n * n, 0.0), g(n), xk(n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                h[i * n + i] = hh;
                g[i] = (i % 2 == 0 ? 1.0 : -1.0) * (1.0 + 0.5 * static_cast<double>(i));
            }
            const Problem p(n, xk, g, h);
            const double delta = 0.3;
            const Step s = solve(p, delta);
            const double gn = norm(g);
            for (std::size_t i = 0; i < n; ++i)
                EXPECT_NEAR(s.d[i], -delta * g[i] / gn, 1e-12) << "h " << hh << " n " << n;
            EXPECT_NEAR(norm(s.d), delta, 1e-12);
            EXPECT_NEAR(s.r.qred, delta * gn - 0.5 * hh * delta * delta, 1e-12);
            EXPECT_FALSE(s.r.have_curvature) << "the boundary stopped the step";
        }
    }
}

TEST(V0121Trsbox, ADiagonalModelWithItsSolutionOnTheBoundaryIsSolvedByTheMovesRoundIt) {
    // Q = g.d + (1/2) d^T diag(h) d, solution on |d| = Delta: d_i = -g_i /
    // (h_i + lambda) with lambda >= 0 from |d| = Delta, found here by
    // bisection. TRSBOX reaches the boundary by CG and then moves round it by
    // (3.6); the reduction comes within the 1 percent of (3.4) of the optimum.
    struct Case {
        std::vector<double> h, g;
        double delta;
    };
    for (const Case& c : {Case{.h = {1.0, 10.0}, .g = {-3.0, -3.0}, .delta = 1.0},
                          Case{.h = {1.0, 4.0, 9.0}, .g = {-2.0, 2.0, -2.0}, .delta = 0.8},
                          Case{.h = {0.5, 20.0}, .g = {-1.0, -4.0}, .delta = 0.4},
                          Case{.h = {2.0, 3.0}, .g = {5.0, -5.0}, .delta = 0.5}}) {
        const std::size_t n = c.h.size();
        const auto length = [&](double lambda) {
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double v = c.g[i] / (c.h[i] + lambda);
                s += v * v;
            }
            return std::sqrt(s);
        };
        double lo = 0.0, hi = 1.0;
        while (length(hi) > c.delta) hi *= 2.0;
        for (int k = 0; k < 200; ++k) {
            const double mid = 0.5 * (lo + hi);
            (length(mid) > c.delta ? lo : hi) = mid;
        }
        std::vector<double> dstar(n);
        double qstar = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            dstar[i] = -c.g[i] / (c.h[i] + hi);
            qstar += -(c.g[i] * dstar[i] + 0.5 * c.h[i] * dstar[i] * dstar[i]);
        }
        std::vector<double> hm(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i) hm[i * n + i] = c.h[i];
        const Problem p(n, std::vector<double>(n, 0.0), c.g, hm);
        const Step s = solve(p, c.delta);
        EXPECT_NEAR(norm(s.d), c.delta, 1e-12 * c.delta) << "the step stays on the boundary";
        EXPECT_GE(s.r.qred, 0.97 * qstar) << "the optimum is " << qstar;
        EXPECT_LE(s.r.qred, qstar * (1.0 + 1e-9));
    }
}

TEST(V0121Trsbox, ABoundStopsTheLineSearchAndTheMethodRestartsOnTheFreeCoordinates) {
    // Q = (1/2)|d|^2 - c.d with c = (2, 1/2), x_k = 0 and an upper bound 1 on
    // the first coordinate. The first search direction s = c reaches the bound
    // at alpha = 1/2, d = (1, 1/4); that index joins the active set, the
    // second coordinate is restarted from there and goes to its minimiser 1/2.
    Problem p(2, {0.0, 0.0}, {-2.0, -0.5}, identity2());
    p.upper(0, 1.0);
    const Step s = solve(p, 10.0);
    EXPECT_EQ(s.side[0], Side::Upper);
    EXPECT_EQ(s.d[0], 1.0) << "on the bound exactly";
    EXPECT_NEAR(s.d[1], 0.5, 1e-12);
    // Q(d) = (1/2)(1 + 1/4) - 2 - 1/4 = -1.625.
    EXPECT_NEAR(s.r.qred, 1.625, 1e-12);
    // The same with the bound below and the signs flipped.
    Problem q(2, {0.0, 0.0}, {2.0, 0.5}, identity2());
    q.lower(0, -1.0);
    const Step t = solve(q, 10.0);
    EXPECT_EQ(t.side[0], Side::Lower);
    EXPECT_EQ(t.d[0], -1.0);
    EXPECT_NEAR(t.d[1], -0.5, 1e-12);
}

TEST(V0121Trsbox, TheBoundRestartStopsWhenTheRemainingGradientIsWithinTheOnePercentOf34) {
    // As above with c = (2, 1/1000). After the first coordinate stops on its
    // bound, d = (1, 1/2000), the reduction so far is about 1.5 and
    // |P_I(grad Q)| Delta = (1/2000) * 5 is below 0.01 of it: (3.4) ends the
    // method there, with d_2 = 1/2000 and not the minimiser 1/1000.
    Problem p(2, {0.0, 0.0}, {-2.0, -0.001}, identity2());
    p.upper(0, 1.0);
    const Step s = solve(p, 5.0);
    EXPECT_EQ(s.d[0], 1.0);
    EXPECT_NEAR(s.d[1], 0.0005, 1e-12);
    EXPECT_EQ(s.side[1], Side::Free);
    // With the second coordinate's gradient large the same bound restarts and
    // the method goes on to its minimiser.
    Problem q(2, {0.0, 0.0}, {-2.0, -0.4}, identity2());
    q.upper(0, 1.0);
    const Step t = solve(q, 5.0);
    EXPECT_NEAR(t.d[1], 0.4, 1e-12);
}

TEST(V0121Trsbox, ACoordinateThatReachesItsBoundInAnUnboundedDirectionStaysThere) {
    // Active sets only grow: once d_1 is on its bound it is never moved,
    // however the arc and the later steps go.
    Problem p(2, {0.0, 0.0}, {-3.0, -3.0}, std::vector<double>{1.0, 0.0, 0.0, 10.0});
    p.upper(0, 0.25);
    const Step s = solve(p, 1.0);
    EXPECT_EQ(s.side[0], Side::Upper);
    EXPECT_EQ(s.d[0], 0.25);
    EXPECT_LE(norm(s.d), 1.0 + 1e-12);
    EXPECT_GT(s.d[1], 0.0);
}

TEST(V0121Trsbox, TheLeastCurvatureOfTheFullStepsIsReportedForSection6) {
    // The set S of (6.10): s^T H s / |s|^2 over the full CG steps. In one
    // dimension with h > 0 and a Newton step inside the radius, the first step
    // is a full one and its curvature is h; a step stopped by the radius or by
    // a bound is not a full one.
    for (const double hh : {0.5, 3.0}) {
        const Problem p(1, {0.0}, {-1.0}, std::vector<double>{hh});
        const Step s = solve(p, 10.0);
        ASSERT_TRUE(s.r.have_curvature);
        EXPECT_NEAR(s.r.curvature, hh, 1e-12);
    }
    {
        const Problem p(1, {0.0}, {-1.0}, std::vector<double>{0.5});
        EXPECT_FALSE(solve(p, 0.5).r.have_curvature) << "stopped by the radius";
    }
    {
        Problem p(1, {0.0}, {-1.0}, std::vector<double>{0.5});
        p.upper(0, 0.5);
        EXPECT_FALSE(solve(p, 10.0).r.have_curvature) << "stopped by the bound";
    }
}

TEST(V0121Trsbox, EveryStepIsInsideTheBoxAndTheTrustRegionAndItsReductionIsTheModelsOwn) {
    // Random problems of every kind, convex or not, with the box tight, loose,
    // one-sided and absent, x_k inside it or on a bound. What must hold
    // whatever the problem: d inside the trust region and the box; a
    // coordinate flagged as on a bound is on it to the bit and an unflagged
    // one is not outside; a coordinate fixed in the initial active set has
    // d_i = 0; qred and gd are the model's reduction and gradient at x_k + d;
    // and the model does not rise.
    Draw rng(0x7E57B0CULL);
    std::size_t boundary_steps = 0, bound_steps = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const std::size_t n = 1 + rng.below(4);
        std::vector<double> a(n * n), h(n * n, 0.0), g(n), xk(n, 0.0);
        for (double& v : a) v = 1.5 * rng.next();
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < n; ++k) h[i * n + k] = a[i * n + k] + a[k * n + i];
        for (double& v : g) v = 2.0 * rng.next();
        Problem p(n, xk, g, h);
        std::vector<char> initially_fixed(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t kind = rng.below(5);
            const bool at_bound = rng.below(2) == 0;
            if (kind == 0 || kind == 1 || kind == 4)  // a lower bound
                p.lower(i, at_bound ? 0.0 : -(0.1 + 1.4 * std::fabs(rng.next())));
            if (kind == 0 || kind == 2 || kind == 4)  // an upper bound
                p.upper(i, at_bound ? 0.0 : 0.1 + 1.4 * std::fabs(rng.next()));
            if (kind == 4) {  // box shut on a side at x_k
                p.lower(i, 0.0);
            }
            if (p.box.has_lo[i] && p.box.lo[i] == 0.0 && g[i] >= 0.0) initially_fixed[i] = 1;
            if (p.box.has_hi[i] && p.box.hi[i] == 0.0 && g[i] <= 0.0) initially_fixed[i] = 1;
        }
        const double delta = 0.1 + 1.5 * std::fabs(rng.next());
        const Step s = solve(p, delta);
        EXPECT_LE(norm(s.d), delta * (1.0 + 1e-12)) << "trial " << trial;
        for (std::size_t i = 0; i < n; ++i) {
            if (p.box.has_lo[i]) {
                EXPECT_GE(s.d[i], p.box.lo[i]) << "trial " << trial;
            }
            if (p.box.has_hi[i]) {
                EXPECT_LE(s.d[i], p.box.hi[i]) << "trial " << trial;
            }
            if (s.side[i] == Side::Lower) {
                EXPECT_EQ(s.d[i], p.box.lo[i]) << "trial " << trial << " coordinate " << i;
            }
            if (s.side[i] == Side::Upper) {
                EXPECT_EQ(s.d[i], p.box.hi[i]) << "trial " << trial << " coordinate " << i;
            }
            if (initially_fixed[i]) {
                EXPECT_EQ(s.d[i], 0.0) << "trial " << trial;
            }
            if (s.side[i] != Side::Free) ++bound_steps;
        }
        const double change = model_change(p, s.d);
        EXPECT_NEAR(-change, s.r.qred, 1e-9 * (1.0 + std::fabs(change))) << "trial " << trial;
        EXPECT_LE(change, 1e-12) << "the model must not rise, trial " << trial;
        EXPECT_GE(s.r.qred, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double want = g[i];
            for (std::size_t k = 0; k < n; ++k) want += h[i * n + k] * s.d[k];
            EXPECT_NEAR(s.gd[i], want, 1e-9 * (1.0 + std::fabs(want))) << "trial " << trial;
        }
        if (std::fabs(norm(s.d) - delta) <= 1e-12 * delta) ++boundary_steps;
        // A step of zero length reduces nothing.
        if (norm(s.d) == 0.0) {
            EXPECT_EQ(s.r.qred, 0.0);
        }
    }
    EXPECT_GT(boundary_steps, 300u) << "the trust-region boundary must have been exercised";
    EXPECT_GT(bound_steps, 300u) << "and so must the bounds";
}
