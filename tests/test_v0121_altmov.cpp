// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: BOBYQA's alternative step, ALTMOV.
//
// Powell, DAMTP 2009/NA06, section 3, second half. With the point y_t to be
// replaced already chosen (3.7), the step aims at a large |Lambda_t(x_k + d)|
// for the Lagrange function Lambda_t of (3.8), whose parameters are column t
// of H (4.15)-(4.16). Two steps are made: d on one of the lines through x_k
// and another interpolation point, the line and the point maximising (3.11)
// with the alpha_j that maximises |phi_j| in the box and the trust region;
// and the Cauchy step c, the exact solution of the linear problem (3.13) by
// (3.14) and (3.15) for Lambda_t and for -Lambda_t, scaled by the multiple in
// [0, 1] that minimises the function, whichever of the two gives the larger
// |Lambda_t(x_k + c)|. The test (3.12) that chooses between d and c is the
// caller's, since it needs the beta of the update.
//
// Everything is held against an independent computation from explicit
// quantities: Lambda_t from the inverse of W (v0121_reference.hpp), each
// phi_j evaluated on the function itself, the line's feasible range found
// coordinate by coordinate, (3.11) scored with H_tt, and (3.13) solved by
// bisection on the multiplier of s_i = clamp(-mu g_i, lower_i, upper_i)
// where the report iterates on the set S.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "flop/detail/bobyqa_altmov.hpp"
#include "flop/detail/bobyqa_model.hpp"
#include "v0121_problems.hpp"
#include "v0121_reference.hpp"

namespace {

using flop::detail::bobyqa::altmov;
using flop::detail::bobyqa::AltmovWork;
using flop::detail::bobyqa::Box;
using flop::detail::bobyqa::Inverse;
using flop::detail::bobyqa::Model;
using flop::detail::bobyqa::Side;
using v0121::Dense;
using v0121::Draw;
using v0121::Place;
using v0121::Points;

// A point set, the library's inverse for it, and the explicit Lambda_t.
struct Scene {
    std::size_t n = 0, m = 0, t = 0;
    Points y;
    std::vector<double> xk;
    Dense h = Dense(0);
    std::vector<double> lambda, glam;  // lambda_l and the gradient at the origin
    double c0 = 0.0;
};

double lag_at(const Scene& s, const std::vector<double>& x) {
    double v = s.c0;
    for (std::size_t k = 0; k < s.n; ++k) v += s.glam[k] * x[k];
    for (std::size_t l = 0; l < s.m; ++l) {
        double yx = 0.0;
        for (std::size_t k = 0; k < s.n; ++k) yx += s.y[l * s.n + k] * x[k];
        v += 0.5 * s.lambda[l] * yx * yx;
    }
    return v;
}

std::vector<double> lag_gradient_at(const Scene& s, const std::vector<double>& x) {
    std::vector<double> g = s.glam;
    for (std::size_t l = 0; l < s.m; ++l) {
        double yx = 0.0;
        for (std::size_t k = 0; k < s.n; ++k) yx += s.y[l * s.n + k] * x[k];
        for (std::size_t k = 0; k < s.n; ++k) g[k] += s.lambda[l] * yx * s.y[l * s.n + k];
    }
    return g;
}

double lag_curvature_along(const Scene& s, const std::vector<double>& v) {
    double c = 0.0;
    for (std::size_t l = 0; l < s.m; ++l) {
        double yv = 0.0;
        for (std::size_t k = 0; k < s.n; ++k) yv += s.y[l * s.n + k] * v[k];
        c += s.lambda[l] * yv * yv;
    }
    return c;
}

// The solution of (3.13) for sign * Lambda: min sign g.s over |s| <= delta and
// sl <= s <= su. KKT: s_i = clamp(-mu sign g_i, sl_i, su_i) with mu from
// |s| = delta, or the corner when that is inside the radius.
std::vector<double> solve_linear_problem(const std::vector<double>& g, double sign,
                                         const std::vector<double>& sl,
                                         const std::vector<double>& su, double delta) {
    const std::size_t n = g.size();
    const auto at = [&](double mu) {
        std::vector<double> s(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double gi = sign * g[i];
            double v;
            if (gi > 0.0)
                v = std::max(sl[i], -mu * gi);
            else if (gi < 0.0)
                v = std::min(su[i], -mu * gi);
            else
                v = 0.0;
            s[i] = v;
        }
        return s;
    };
    const auto len = [](const std::vector<double>& v) {
        double q = 0.0;
        for (const double x : v) q += x * x;
        return std::sqrt(q);
    };
    double hi = 1.0;
    for (int k = 0; k < 2000 && len(at(hi)) < delta; ++k) {
        const std::vector<double> corner = at(1e300);
        if (len(corner) <= delta) return corner;
        hi *= 2.0;
    }
    if (len(at(1e300)) <= delta) return at(1e300);
    double lo = 0.0;
    for (int k = 0; k < 300; ++k) {
        const double mid = 0.5 * (lo + hi);
        (len(at(mid)) < delta ? lo : hi) = mid;
    }
    return at(hi);
}

// The multiple gamma in [0, 1] of s that minimises sign * Lambda along it.
struct Cauchy {
    std::vector<double> c;
    double value = 0.0;  // sign * Lambda_t(x_k + c)
};

Cauchy cauchy_of(const Scene& s, const std::vector<double>& g, double sign,
                 const std::vector<double>& step) {
    double lin = 0.0;
    for (std::size_t i = 0; i < s.n; ++i) lin += sign * g[i] * step[i];
    const double crv = sign * lag_curvature_along(s, step);
    double gamma = 1.0;
    if (crv > 0.0 && -lin < crv) gamma = -lin / crv;
    if (gamma < 0.0) gamma = 0.0;
    Cauchy r;
    r.c.resize(s.n);
    for (std::size_t i = 0; i < s.n; ++i) r.c[i] = gamma * step[i];
    r.value = gamma * lin + 0.5 * gamma * gamma * crv;
    return r;
}

// A random scene: points of the starting procedure with a few replacements, a
// best point x_k, an index t other than it, and a box around x_k.
struct Built {
    Scene scene;
    Box box;
    double delta = 1.0;
    explicit Built(std::size_t n) : box(n) {}
};

bool build(Built& b, Draw& rng, std::size_t n, std::size_t m, int replacements) {
    Scene& s = b.scene;
    s.n = n;
    s.m = m;
    std::vector<Place> place(n);
    for (std::size_t i = 0; i < n; ++i) place[i] = static_cast<Place>(rng.below(3));
    std::vector<double> alpha, beta;
    v0121::multipliers(place, 0.4 + 0.3 * std::fabs(rng.next()), alpha, beta);
    s.y = v0121::coordinate_points(alpha, beta, m);
    Inverse inv(n, m);
    inv.set_coordinate(alpha, beta);
    std::vector<double> vlag(m + n), wv(m), zw(m - n - 1), d(n), sum(n), hcol(m + n), ucol(m + n);
    for (int r = 0; r < replacements; ++r) {
        const std::size_t sidx = rng.below(m);
        std::size_t t = rng.below(m);
        if (t == sidx) t = (t + 1) % m;
        std::vector<double> x(n);
        for (double& v : x) v = 0.8 * rng.next();
        const Dense h = v0121::explicit_h(s.y, n, m);
        const std::vector<double> w = v0121::w_vector(s.y, n, m, x);
        const std::vector<double> hw = v0121::times(h, w);
        const double beta_x = inv.vlag_beta(s.y, sidx, x, vlag, wv, zw, d, sum);
        const double sigma = h(t, t) * beta_x + hw[t] * hw[t];
        if (!(sigma > 1e-3)) continue;
        inv.update(t, vlag, beta_x, hcol, ucol);
        for (std::size_t k = 0; k < n; ++k) s.y[t * n + k] = x[k];
    }
    s.t = rng.below(m);
    std::size_t kopt = rng.below(m);
    if (kopt == s.t) kopt = (kopt + 1) % m;
    s.xk.assign(s.y.begin() + static_cast<std::ptrdiff_t>(kopt * n),
                s.y.begin() + static_cast<std::ptrdiff_t>((kopt + 1) * n));
    s.h = v0121::explicit_h(s.y, n, m);
    if (s.h.size != m + n + 1) return false;
    s.lambda.resize(m);
    s.glam.resize(n);
    for (std::size_t l = 0; l < m; ++l) s.lambda[l] = s.h(l, s.t);
    s.c0 = s.h(m, s.t);
    for (std::size_t k = 0; k < n; ++k) s.glam[k] = s.h(m + 1 + k, s.t);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t kind = rng.below(4);
        const double a = kind == 0 ? 0.0 : 0.05 + 1.2 * std::fabs(rng.next());
        const double c = kind == 1 ? 0.0 : 0.05 + 1.2 * std::fabs(rng.next());
        if (kind != 2) {
            b.box.has_lo[i] = 1;
            b.box.lo[i] = s.xk[i] - a;
        }
        if (kind != 3) {
            b.box.has_hi[i] = 1;
            b.box.hi[i] = s.xk[i] + c;
        }
    }
    b.delta = 0.3 + 1.2 * std::fabs(rng.next());
    return true;
}

// The model ALTMOV reads: the points and the index of x_k. Nothing else of it
// is used.
struct Fixture {
    Model q;
    Inverse inv;
    Fixture(const Scene& s, std::size_t kopt) : q(s.n, s.m), inv(s.n, s.m) {
        std::ranges::copy(s.y, q.points.begin());
        q.kopt = kopt;
    }
};

// Writes the explicit H into the library's Inverse: B from its blocks, and Z
// as a pivoted Cholesky factor of Omega (Omega is positive semidefinite of
// rank at most m - n - 1, and any Z with Z Z^T = Omega serves).
void install_inverse(Inverse& inv, const Scene& s) {
    const std::size_t n = s.n, m = s.m, nz = m - n - 1;
    std::vector<double> a(m * m);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) a[i * m + j] = s.h(i, j);
    std::vector<std::vector<double>> cols;
    for (std::size_t c = 0; c < nz; ++c) {
        std::size_t piv = 0;
        double best = -1.0;
        for (std::size_t i = 0; i < m; ++i) {
            if (a[i * m + i] > best) {
                best = a[i * m + i];
                piv = i;
            }
        }
        std::vector<double> col(m, 0.0);
        if (best > 1e-14) {
            const double r = std::sqrt(best);
            for (std::size_t i = 0; i < m; ++i) col[i] = a[i * m + piv] / r;
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t j = 0; j < m; ++j) a[i * m + j] -= col[i] * col[j];
        }
        cols.push_back(col);
    }
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t c = 0; c < nz; ++c) inv.z(i, c) = cols[c][i];
    for (std::size_t j = 0; j < m; ++j)
        for (std::size_t k = 0; k < n; ++k) inv.b(j, k) = s.h(m + 1 + k, j);
    for (std::size_t p = 0; p < n; ++p)
        for (std::size_t q = 0; q < n; ++q) inv.b(m + p, q) = s.h(m + 1 + p, m + 1 + q);
}

std::size_t index_of(const Scene& s) {
    for (std::size_t j = 0; j < s.m; ++j) {
        bool same = true;
        for (std::size_t k = 0; k < s.n; ++k) same = same && s.y[j * s.n + k] == s.xk[k];
        if (same) return j;
    }
    return s.m;
}

}  // namespace

TEST(V0121Altmov, TheLineStepIsTheOneThatMaximises311OnItsOwnFeasibleRange) {
    Draw rng(0xA17A0FULL);
    std::size_t checked = 0, on_bound = 0, interior_alpha = 0, in_trust_region = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const std::size_t n = 2 + rng.below(2);
        const std::size_t m = n + 2 + rng.below((n + 1) * (n + 2) / 2 - n - 1);
        Built b(n);
        if (!build(b, rng, n, m, 6)) continue;
        Scene& s = b.scene;
        const std::size_t kopt = index_of(s);
        if (kopt == s.m) continue;
        Fixture f(s, kopt);
        install_inverse(f.inv, s);
        AltmovWork w(n, m);
        std::vector<double> d(n), c(n);
        std::vector<Side> dside(n), cside(n);
        (void)altmov(f.q, f.inv, b.box, s.t, b.delta, d, dside, c, cside, w);

        // The independent line step.
        const double htt = s.h(s.t, s.t);
        bool have = false;
        double best_score = 0.0, best_alpha = 0.0;
        std::size_t best_j = 0;
        std::vector<double> best_dir;
        double second = 0.0;
        for (std::size_t j = 0; j < m; ++j) {
            if (j == kopt) continue;
            std::vector<double> dir(n);
            double len2 = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                dir[k] = s.y[j * n + k] - s.xk[k];
                len2 += dir[k] * dir[k];
            }
            if (!(len2 > 0.0)) continue;
            const double len = std::sqrt(len2);
            double ahi = b.delta / len, alo = -ahi;
            for (std::size_t k = 0; k < n; ++k) {
                if (dir[k] == 0.0) continue;
                const double up = b.box.has_hi[k] ? (b.box.hi[k] - s.xk[k]) / dir[k] : 0.0;
                const double dn = b.box.has_lo[k] ? (b.box.lo[k] - s.xk[k]) / dir[k] : 0.0;
                if (dir[k] > 0.0) {
                    if (b.box.has_hi[k]) ahi = std::min(ahi, up);
                    if (b.box.has_lo[k]) alo = std::max(alo, dn);
                } else {
                    if (b.box.has_lo[k]) ahi = std::min(ahi, dn);
                    if (b.box.has_hi[k]) alo = std::max(alo, up);
                }
            }
            ahi = std::max(ahi, 0.0);
            alo = std::min(alo, 0.0);
            // phi from the function itself: phi(a) = p a + q a^2.
            const auto phi = [&](double a) {
                std::vector<double> x(n);
                for (std::size_t k = 0; k < n; ++k) x[k] = s.xk[k] + a * dir[k];
                return lag_at(s, x);
            };
            EXPECT_NEAR(phi(0.0), 0.0, 1e-9) << "Lambda_t(x_k) = 0";
            EXPECT_NEAR(phi(1.0), j == s.t ? 1.0 : 0.0, 1e-8) << "phi_j(1) = delta_jt";
            const double p = 0.5 * (phi(1.0) - phi(-1.0));
            const double q = 0.5 * (phi(1.0) + phi(-1.0));
            const auto val = [&](double a) { return a * (p + q * a); };
            double a_best = ahi;
            if (std::fabs(val(alo)) > std::fabs(val(a_best))) a_best = alo;
            if (q != 0.0) {
                const double astar = -p / (2.0 * q);
                if (astar > alo && astar < ahi && std::fabs(val(astar)) > std::fabs(val(a_best)))
                    a_best = astar;
            }
            const double phi2 = val(a_best) * val(a_best);
            const double om = 1.0 - a_best;
            const double score =
                phi2 * (0.5 * htt * a_best * a_best * om * om * len2 * len2 + phi2);
            if (!have || score > best_score) {
                second = best_score;
                have = true;
                best_score = score;
                best_alpha = a_best;
                best_j = j;
                best_dir = dir;
            } else if (score > second) {
                second = score;
            }
        }
        if (!have) continue;
        // Skip a near tie: either line is then a right answer.
        if (second > best_score * (1.0 - 1e-6)) continue;
        ++checked;
        for (std::size_t k = 0; k < n; ++k) {
            EXPECT_NEAR(d[k], best_alpha * best_dir[k], 1e-8 * (1.0 + b.delta))
                << "trial " << trial << " n " << n << " m " << m << " j " << best_j;
        }
        // The step is feasible.
        double dn = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            dn += d[k] * d[k];
            if (b.box.has_lo[k]) {
                EXPECT_GE(s.xk[k] + d[k], b.box.lo[k] - 1e-12);
            }
            if (b.box.has_hi[k]) {
                EXPECT_LE(s.xk[k] + d[k], b.box.hi[k] + 1e-12);
            }
            if (dside[k] == Side::Lower) {
                EXPECT_NEAR(s.xk[k] + d[k], b.box.lo[k], 1e-10);
                ++on_bound;
            }
            if (dside[k] == Side::Upper) {
                EXPECT_NEAR(s.xk[k] + d[k], b.box.hi[k], 1e-10);
                ++on_bound;
            }
        }
        EXPECT_LE(std::sqrt(dn), b.delta * (1.0 + 1e-12));
        if (best_alpha != 0.0 && std::fabs(best_alpha) < 1.0 - 1e-9) ++interior_alpha;
        if (std::sqrt(dn) < b.delta * (1.0 - 1e-6)) ++in_trust_region;
    }
    EXPECT_GT(checked, 150u);
    EXPECT_GT(on_bound, 10u) << "a line step that ends on a bound must have been seen";
    EXPECT_GT(interior_alpha, 10u) << "and one that ends at the vertex of phi";
    EXPECT_GT(in_trust_region, 10u);
}

TEST(V0121Altmov, TheCauchyStepIsTheExactSolutionOf313ScaledIntoZeroToOne) {
    Draw rng(0xCA0C47ULL);
    std::size_t checked = 0, corner = 0, mixed = 0, round = 0, both_signs[2] = {0, 0};
    for (int trial = 0; trial < 500; ++trial) {
        const std::size_t n = 2 + rng.below(3);
        const std::size_t m = n + 2 + rng.below((n + 1) * (n + 2) / 2 - n - 1);
        Built b(n);
        if (!build(b, rng, n, m, 6)) continue;
        Scene& s = b.scene;
        const std::size_t kopt = index_of(s);
        if (kopt == s.m) continue;
        Fixture f(s, kopt);
        install_inverse(f.inv, s);
        AltmovWork w(n, m);
        std::vector<double> d(n), c(n);
        std::vector<Side> dside(n), cside(n);
        const double value = altmov(f.q, f.inv, b.box, s.t, b.delta, d, dside, c, cside, w);

        // The independent Cauchy step.
        const std::vector<double> g = lag_gradient_at(s, s.xk);
        std::vector<double> sl(n, -1e300), su(n, 1e300);
        for (std::size_t k = 0; k < n; ++k) {
            if (b.box.has_lo[k]) sl[k] = b.box.lo[k] - s.xk[k];
            if (b.box.has_hi[k]) su[k] = b.box.hi[k] - s.xk[k];
        }
        double gn = 0.0, gmax = 0.0;
        for (const double v : g) {
            gn += v * v;
            gmax = std::max(gmax, std::fabs(v));
        }
        if (gn < 1e-12) continue;
        // A component that is zero up to rounding has no sign: which way a
        // step along it goes is then decided by the rounding, in the library's
        // arithmetic and in this one, so such a gradient is not a test of the
        // rule.
        bool unsigned_component = false;
        for (const double v : g)
            unsigned_component = unsigned_component || (v != 0.0 && std::fabs(v) < 1e-9 * gmax);
        if (unsigned_component) continue;
        Cauchy best_c;
        double best_abs = -1.0, other_abs = -1.0;
        double best_sign = 0.0;
        for (const double sign : {1.0, -1.0}) {
            const std::vector<double> step = solve_linear_problem(g, sign, sl, su, b.delta);
            const Cauchy cc = cauchy_of(s, g, sign, step);
            if (std::fabs(cc.value) > best_abs) {
                other_abs = best_abs;
                best_abs = std::fabs(cc.value);
                best_c = cc;
                best_sign = sign;
            } else {
                other_abs = std::fabs(cc.value);
            }
        }
        if (best_abs <= 1e-12) continue;
        if (other_abs > best_abs * (1.0 - 1e-6)) continue;  // a tie between the two signs
        ++checked;
        for (std::size_t k = 0; k < n; ++k)
            EXPECT_NEAR(c[k], best_c.c[k], 1e-7 * (1.0 + b.delta))
                << "trial " << trial << " n " << n << " m " << m << " coordinate " << k;
        EXPECT_NEAR(value, best_abs * best_abs, 1e-7 * (1.0 + best_abs * best_abs))
            << "trial " << trial;
        // The returned value is Lambda_t(x_k + c)^2 of the function itself.
        std::vector<double> x(n);
        for (std::size_t k = 0; k < n; ++k) x[k] = s.xk[k] + c[k];
        const double lam = lag_at(s, x);
        EXPECT_NEAR(lam * lam, value, 1e-7 * (1.0 + value)) << "trial " << trial;
        // c is feasible; a coordinate flagged as on a bound is on it, to the bit.
        double cn = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            cn += c[k] * c[k];
            if (b.box.has_lo[k]) {
                EXPECT_GE(s.xk[k] + c[k], b.box.lo[k] - 1e-12);
            }
            if (b.box.has_hi[k]) {
                EXPECT_LE(s.xk[k] + c[k], b.box.hi[k] + 1e-12);
            }
            if (cside[k] == Side::Lower) {
                EXPECT_EQ(c[k], b.box.lo[k] - s.xk[k]) << "trial " << trial;
            }
            if (cside[k] == Side::Upper) {
                EXPECT_EQ(c[k], b.box.hi[k] - s.xk[k]) << "trial " << trial;
            }
        }
        EXPECT_LE(std::sqrt(cn), b.delta * (1.0 + 1e-12));
        // Which regime of (3.14) and (3.15) this trial was.
        std::vector<double> step = solve_linear_problem(g, best_sign, sl, su, b.delta);
        double sn = 0.0;
        std::size_t on_bounds = 0;
        for (std::size_t k = 0; k < n; ++k) {
            sn += step[k] * step[k];
            if ((b.box.has_lo[k] && step[k] == sl[k]) || (b.box.has_hi[k] && step[k] == su[k]))
                ++on_bounds;
        }
        if (std::sqrt(sn) < b.delta * (1.0 - 1e-9))
            ++corner;
        else if (on_bounds == 0)
            ++round;
        else
            ++mixed;
        ++both_signs[best_sign > 0.0 ? 0 : 1];
    }
    EXPECT_GT(checked, 200u);
    EXPECT_GT(corner, 10u) << "(3.14) with the corner inside the radius";
    EXPECT_GT(round, 10u) << "(3.15) with S empty, a step on the trust-region boundary";
    EXPECT_GT(mixed, 10u) << "(3.15) with some coordinates on their bounds";
    EXPECT_GT(both_signs[0], 10u) << "Lambda_t minimised";
    EXPECT_GT(both_signs[1], 10u) << "and -Lambda_t minimised";
}
