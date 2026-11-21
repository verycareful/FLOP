// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: BOBYQA's quadratic model.
//
// Powell, DAMTP 2009/NA06, section 4. The second derivative matrix is held as
// M + sum_l mu_l y_l y_l^T (4.6), the gradient at the best point (4.7), and a
// step of the method replaces one point and adds to the model the change
// (4.3) whose parameters are a column of the new H times F(x+) - Q(x+), with
// the gradient at x_k carried over by (4.8). The pieces are held to the
// definitions: the products with the matrix against the matrix assembled in
// full, the move of a term from mu into M after (4.6), the rank two matrix of
// (6.17) that keeps the matrix when the origin moves, and the whole update of
// section 4, composed in the order the report gives from the library's
// Inverse and Model, against the same update done explicitly with the
// inverse of W.
//
// What is not reachable from here is the solver's own composition of those
// pieces (it is private); the end-to-end behaviour of the method on problems
// whose model is exact is held in the suites on the starting procedure and on
// the runs.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "flop/detail/bobyqa_model.hpp"
#include "v0121_problems.hpp"
#include "v0121_reference.hpp"

namespace {

using flop::detail::bobyqa::Inverse;
using flop::detail::bobyqa::Model;
using v0121::Dense;
using v0121::Draw;
using v0121::Place;
using v0121::Points;

std::size_t max_m(std::size_t n) {
    return (n + 1) * (n + 2) / 2;
}

// Any smooth, non-quadratic function: the model has something to interpolate
// that it cannot match everywhere.
double smooth(std::span<const double> x) {
    const std::size_t n = x.size();
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        s += std::cos(1.3 * x[i] + 0.2 * static_cast<double>(i)) + 0.5 * x[i] * x[i] +
             0.3 * x[i] * x[(i + 1) % n];
    return s;
}

// The matrix M + sum_l mu_l y_l y_l^T in full.
Dense full_hessian(const Model& q) {
    Dense h(q.n);
    for (std::size_t i = 0; i < q.n; ++i)
        for (std::size_t k = 0; k < q.n; ++k) {
            double s = q.hq[i * q.n + k];
            for (std::size_t l = 0; l < q.m; ++l) s += q.mu[l] * q.point(l)[i] * q.point(l)[k];
            h(i, k) = s;
        }
    return h;
}

std::vector<double> random_vector(Draw& rng, std::size_t n, double scale) {
    std::vector<double> v(n);
    for (double& x : v) x = scale * rng.next();
    return v;
}

// Fills a model with random contents: points, values, a symmetric M, mu, g.
void fill_randomly(Model& q, Draw& rng) {
    for (double& v : q.points) v = rng.next();
    for (double& v : q.fval) v = rng.next();
    for (double& v : q.mu) v = rng.next();
    for (double& v : q.g) v = rng.next();
    for (std::size_t i = 0; i < q.n; ++i)
        for (std::size_t k = i; k < q.n; ++k) {
            const double v = rng.next();
            q.hq[i * q.n + k] = v;
            q.hq[k * q.n + i] = v;
        }
}

}  // namespace

TEST(V0121Model, ProductsWithTheSecondDerivativeMatrixAreTheMatrixInFull) {
    Draw rng(0xB0B0ULL);
    // Sizes on both sides of the row-block width, and a model with zero mu.
    for (const std::size_t n :
         {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{5}, std::size_t{9}}) {
        for (const std::size_t m : {n + 2, 2 * n + 1, max_m(n) < 40 ? max_m(n) : n + 7}) {
            Model q(n, m);
            fill_randomly(q, rng);
            const Dense h = full_hessian(q);
            for (int trial = 0; trial < 4; ++trial) {
                const std::vector<double> v = random_vector(rng, n, 1.0);
                std::vector<double> out(n);
                q.hess_mul(v, out);
                for (std::size_t i = 0; i < n; ++i) {
                    double want = 0.0;
                    for (std::size_t k = 0; k < n; ++k) want += h(i, k) * v[k];
                    EXPECT_NEAR(out[i], want, 1e-12 * (1.0 + std::fabs(want)))
                        << "n " << n << " m " << m << " row " << i;
                }
                // The change of the model along v: g.v + (1/2) v^T (matrix) v.
                std::vector<double> work(n);
                const double change = q.change(v, work);
                double want = 0.0, quad = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    want += q.g[i] * v[i];
                    for (std::size_t k = 0; k < n; ++k) quad += v[i] * h(i, k) * v[k];
                }
                want += 0.5 * quad;
                EXPECT_NEAR(change, want, 1e-12 * (1.0 + std::fabs(want)));
            }
            for (std::size_t i = 0; i < n; ++i)
                EXPECT_NEAR(q.hess_diag(i), h(i, i), 1e-12 * (1.0 + std::fabs(h(i, i))));
            // With every mu zero the matrix is M alone.
            for (double& v : q.mu) v = 0.0;
            const std::vector<double> v = random_vector(rng, n, 1.0);
            std::vector<double> out(n);
            q.hess_mul(v, out);
            for (std::size_t i = 0; i < n; ++i) {
                double want = 0.0;
                for (std::size_t k = 0; k < n; ++k) want += q.hq[i * n + k] * v[k];
                EXPECT_NEAR(out[i], want, 1e-12 * (1.0 + std::fabs(want)));
            }
        }
    }
}

TEST(V0121Model, AbsorbingATermMovesItFromMuIntoM) {
    // After (4.6): mu_t y_t y_t^T is added to M and mu_t is zero, and the
    // matrix is the same.
    Draw rng(0xAB50ULL);
    for (const std::size_t n : {std::size_t{2}, std::size_t{5}})
        for (const std::size_t m : {n + 2, 2 * n + 1}) {
            Model q(n, m);
            fill_randomly(q, rng);
            for (std::size_t t = 0; t < m; ++t) {
                const Dense before = full_hessian(q);
                const double mu_t = q.mu[t];
                std::vector<double> m_before(q.hq.begin(), q.hq.end());
                q.absorb(t);
                EXPECT_EQ(q.mu[t], 0.0);
                const Dense after = full_hessian(q);
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t k = 0; k < n; ++k) {
                        EXPECT_NEAR(after(i, k), before(i, k),
                                    1e-12 * (1.0 + std::fabs(before(i, k))));
                        EXPECT_NEAR(q.hq[i * n + k],
                                    m_before[i * n + k] + mu_t * q.point(t)[i] * q.point(t)[k],
                                    1e-12);
                    }
                // Absorbing a zero term changes nothing at all.
                std::vector<double> again(q.hq.begin(), q.hq.end());
                q.absorb(t);
                for (std::size_t i = 0; i < again.size(); ++i)
                    EXPECT_TRUE(v0121::same_bits(q.hq[i], again[i]));
            }
        }
}

TEST(V0121Model, MovingTheOriginKeepsTheSecondDerivativeMatrix) {
    // (6.17): with the points moved by -s and the mu kept, M gains the rank
    // two matrix, and the matrix M + sum mu_l y_l y_l^T is what it was.
    Draw rng(0x5F1F7ULL);
    for (const std::size_t n : {std::size_t{1}, std::size_t{3}, std::size_t{6}})
        for (const std::size_t m : {n + 2, 2 * n + 1}) {
            Model q(n, m);
            fill_randomly(q, rng);
            const Dense before = full_hessian(q);
            const std::vector<double> s = random_vector(rng, n, 0.8);
            std::vector<double> work(n);
            q.shift_hessian(s, work);
            for (std::size_t j = 0; j < m; ++j)
                for (std::size_t i = 0; i < n; ++i) q.points[j * n + i] -= s[i];
            const Dense after = full_hessian(q);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t k = 0; k < n; ++k)
                    EXPECT_NEAR(after(i, k), before(i, k), 1e-11 * (1.0 + std::fabs(before(i, k))))
                        << "n " << n << " m " << m;
        }
}

namespace {

// The explicit quadratic of the report, c + g.x + (1/2) x^T G x about the
// origin, updated by (4.3) from the columns of the explicit H.
struct Explicit {
    double c = 0.0;
    std::vector<double> g;
    Dense big;
    explicit Explicit(std::size_t n) : g(n, 0.0), big(n) {}
    [[nodiscard]] double at(std::span<const double> x) const {
        double s = c;
        for (std::size_t i = 0; i < g.size(); ++i) {
            s += g[i] * x[i];
            for (std::size_t k = 0; k < g.size(); ++k) s += 0.5 * x[i] * big(i, k) * x[k];
        }
        return s;
    }
    [[nodiscard]] std::vector<double> gradient(std::span<const double> x) const {
        std::vector<double> r(g);
        for (std::size_t i = 0; i < g.size(); ++i)
            for (std::size_t k = 0; k < g.size(); ++k) r[i] += big(i, k) * x[k];
        return r;
    }
};

// Q = c + g.x + (1/2) sum_l lambda_l (y_l.x)^2 added to the explicit model,
// (lambda, c, g) being the scaled column t of h: the change (4.3).
void add_change(Explicit& q, const Dense& h, const Points& y, std::size_t n, std::size_t m,
                std::size_t t, double scale) {
    q.c += scale * h(m, t);
    for (std::size_t k = 0; k < n; ++k) q.g[k] += scale * h(m + 1 + k, t);
    for (std::size_t l = 0; l < m; ++l)
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < n; ++k)
                q.big(i, k) += scale * h(l, t) * y[l * n + i] * y[l * n + k];
}

}  // namespace

TEST(V0121Model, TheUpdateOfSection4InterpolatesAndMatchesTheExplicitUpdate) {
    // Start from the least Frobenius norm interpolant of the starting points,
    // (lambda, c, g) = H (F, 0) of (2.6), then replace one point at a time:
    // the library's inverse is updated by (4.9) to (4.14) and its column t
    // gives the change (4.3) scaled by F(x+) - Q(x+), whose gradient at x_k
    // is carried by (4.8). After every step the model interpolates every
    // point, its matrix, gradient and values are those of the explicit update
    // done with the inverse of the new W, and the multipliers satisfy (4.2).
    using flop::detail::bobyqa::add_weighted_rows;
    Draw rng(0xC0FFEEULL);
    std::size_t steps = 0, improving = 0, non_improving = 0;
    for (std::size_t n = 1; n <= 4; ++n)
        for (const std::size_t m : {n + 2, 2 * n + 1, max_m(n)}) {
            std::vector<Place> place(n);
            for (std::size_t i = 0; i < n; ++i) place[i] = static_cast<Place>(i % 3);
            std::vector<double> alpha, beta;
            v0121::multipliers(place, 0.5, alpha, beta);
            Model q(n, m);
            Points y = v0121::coordinate_points(alpha, beta, m);
            std::ranges::copy(y, q.points.begin());
            for (std::size_t j = 0; j < m; ++j) q.fval[j] = smooth(q.point(j));
            Inverse inv(n, m);
            inv.set_coordinate(alpha, beta);

            // The first model: the explicit least-norm interpolant.
            Explicit ex(n);
            {
                const Dense h = v0121::explicit_h(y, n, m);
                std::vector<double> r(m + n + 1, 0.0);
                for (std::size_t j = 0; j < m; ++j) r[j] = q.fval[j];
                const std::vector<double> sol = v0121::times(h, r);
                ex.c = sol[m];
                for (std::size_t k = 0; k < n; ++k) ex.g[k] = sol[m + 1 + k];
                for (std::size_t l = 0; l < m; ++l)
                    for (std::size_t i = 0; i < n; ++i)
                        for (std::size_t k = 0; k < n; ++k)
                            ex.big(i, k) += sol[l] * y[l * n + i] * y[l * n + k];
                for (std::size_t l = 0; l < m; ++l) q.mu[l] = sol[l];
            }
            q.kopt = 0;
            for (std::size_t j = 1; j < m; ++j)
                if (q.fval[j] < q.fval[q.kopt]) q.kopt = j;
            {
                const std::vector<double> gk = ex.gradient(q.xopt());
                std::ranges::copy(gk, q.g.begin());
            }

            std::vector<double> vlag(m + n), wv(m), zw(m - n - 1), d(n), sum(n), hcol(m + n),
                ucol(m + n), lam(m), work(n), work2(n), xold(n);
            for (int step = 0; step < 10; ++step) {
                // A new point near the others, replacing a point other than x_k.
                const std::size_t s = q.kopt;
                std::size_t t = rng.below(m);
                if (t == s) t = (t + 1) % m;
                const std::vector<double> x = random_vector(rng, n, 0.7);
                const double b = inv.vlag_beta(q.points, s, x, vlag, wv, zw, d, sum);
                const double sigma = inv.omega_diag(t) * b + vlag[t] * vlag[t];
                if (!(sigma > 1e-4)) continue;
                ++steps;

                // F(x+) - Q(x+), the model before the update at the new point.
                std::vector<double> dk(n);
                for (std::size_t k = 0; k < n; ++k) dk[k] = x[k] - q.xopt()[k];
                const double q_new_point = q.fopt() + q.change(dk, work);
                EXPECT_NEAR(q_new_point, ex.at(x), 1e-8 * (1.0 + std::fabs(q_new_point)));
                const double fplus = smooth(x) + 0.3 * rng.next();
                const double diff = fplus - q_new_point;

                // The library's pieces, in the order of section 4.
                inv.update(t, vlag, b, hcol, ucol);
                std::ranges::copy(q.xopt(), xold.begin());
                const bool improved = fplus < q.fopt();
                q.absorb(t);
                for (std::size_t k = 0; k < n; ++k) q.points[t * n + k] = x[k];
                q.fval[t] = fplus;
                inv.omega_col_updated(t, lam);
                double lam_sum = 0.0;
                std::vector<double> lam_moment(n, 0.0);
                for (std::size_t l = 0; l < m; ++l) {
                    lam[l] *= diff;
                    q.mu[l] += lam[l];
                    lam_sum += lam[l];
                    for (std::size_t k = 0; k < n; ++k) lam_moment[k] += lam[l] * q.point(l)[k];
                }
                // (4.2).
                EXPECT_NEAR(lam_sum, 0.0, 1e-8 * (1.0 + std::fabs(diff)));
                for (std::size_t k = 0; k < n; ++k)
                    EXPECT_NEAR(lam_moment[k], 0.0, 1e-8 * (1.0 + std::fabs(diff)));
                // (4.8): the gradient at the old x_k.
                for (std::size_t k = 0; k < n; ++k) q.g[k] += diff * inv.b(t, k);
                add_weighted_rows<false>(q.points.data(), m, n, lam.data(), xold.data(),
                                         q.g.data());
                if (improved) {
                    for (std::size_t k = 0; k < n; ++k) work2[k] = x[k] - xold[k];
                    q.hess_mul(work2, work);
                    for (std::size_t k = 0; k < n; ++k) q.g[k] += work[k];
                    q.kopt = t;
                }
                (improved ? improving : non_improving) += 1;

                // The explicit update with the inverse of the new W.
                std::ranges::copy(x, y.begin() + static_cast<std::ptrdiff_t>(t * n));
                const Dense hnew = v0121::explicit_h(y, n, m);
                ASSERT_EQ(hnew.size, m + n + 1);
                add_change(ex, hnew, y, n, m, t, diff);

                // Interpolation at every point, to rounding.
                double spread = 1.0;
                for (std::size_t j = 0; j < m; ++j) spread = std::max(spread, std::fabs(q.fval[j]));
                for (std::size_t j = 0; j < m; ++j) {
                    for (std::size_t k = 0; k < n; ++k) dk[k] = q.point(j)[k] - q.xopt()[k];
                    const double qj = q.fopt() + q.change(dk, work);
                    EXPECT_NEAR(qj, q.fval[j], 1e-8 * spread)
                        << "n " << n << " m " << m << " step " << step << " point " << j;
                }
                // The model against the explicit one at random points, and its
                // gradient and matrix at x_k.
                for (int probe = 0; probe < 3; ++probe) {
                    const std::vector<double> z = random_vector(rng, n, 1.0);
                    for (std::size_t k = 0; k < n; ++k) dk[k] = z[k] - q.xopt()[k];
                    const double qz = q.fopt() + q.change(dk, work);
                    EXPECT_NEAR(qz, ex.at(z), 1e-7 * (1.0 + std::fabs(qz)))
                        << "n " << n << " m " << m << " step " << step;
                }
                const std::vector<double> gx = ex.gradient(q.xopt());
                for (std::size_t k = 0; k < n; ++k)
                    EXPECT_NEAR(q.g[k], gx[k], 1e-7 * (1.0 + std::fabs(gx[k])))
                        << "n " << n << " m " << m << " step " << step << " g_" << k;
                const Dense hess = full_hessian(q);
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t k = 0; k < n; ++k)
                        EXPECT_NEAR(hess(i, k), ex.big(i, k), 1e-7 * (1.0 + std::fabs(hess(i, k))));
                EXPECT_EQ(q.mu[t], lam[t]);
            }
        }
    EXPECT_GT(steps, 60u);
    EXPECT_GT(improving, 5u) << "the step that moves x_k must be exercised";
    EXPECT_GT(non_improving, 5u) << "and the step that keeps it";
}
