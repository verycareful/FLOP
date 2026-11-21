// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: BOBYQA's RESCUE.
//
// Powell, DAMTP 2009/NA06, section 5. When rounding has spoiled the
// denominator of the update (5.2), H is thrown away and rebuilt for the
// points gamma_j of (5.4) and (5.5) around x_k, with the multipliers chosen
// there to keep every gamma_j feasible; old points then replace new ones one
// at a time, the old point of least positive score psi first, into the slot
// that maximises (5.6), while (5.7) holds, a point that fails having its score
// raised by the greatest score; and the model is made to interpolate the new
// points by (5.12) after the move of the origin (5.3).
//
// rescue_plan is held to: the gamma points, bit for bit, in every slot it
// leaves new; a transcription of the iteration written out again with an
// explicit inverse at every step (the replacements it makes and the order
// of its candidates must be the same); the inverse it leaves, which must be
// that of the final point set; and the cases whose answer is known, old
// points that are the gamma points in another order (nothing is new) and old
// points that are all at x_k (nothing is old). The update of the model, (5.3)
// and (5.12), is composed in the test from the library's pieces in the report's
// order and held to the explicit update.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "flop/detail/bobyqa_model.hpp"
#include "flop/detail/bobyqa_rescue.hpp"
#include "v0121_problems.hpp"
#include "v0121_reference.hpp"

namespace {

using flop::detail::bobyqa::Box;
using flop::detail::bobyqa::Inverse;
using flop::detail::bobyqa::kNewPoint;
using flop::detail::bobyqa::Model;
using flop::detail::bobyqa::rescue_plan;
using flop::detail::bobyqa::RescuePlan;
using v0121::Dense;
using v0121::Draw;
using v0121::Points;

constexpr double kSafe = 0.01;  // the multiplier of (5.7)

std::size_t max_m(std::size_t n) {
    return (n + 1) * (n + 2) / 2;
}

// The multipliers of (5.4) for a box around x_k = 0: lo <= 0 <= hi when present.
struct Multipliers {
    std::vector<double> alpha, beta;
};

Multipliers multipliers_54(const Box& box, std::size_t n, double delta) {
    Multipliers r;
    r.alpha.assign(n, 0.0);
    r.beta.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const bool up = !box.has_hi[i] || delta <= box.hi[i];
        const bool down = !box.has_lo[i] || -delta >= box.lo[i];
        if (up && down) {
            r.alpha[i] = delta;
            r.beta[i] = -delta;
            continue;
        }
        r.alpha[i] = up ? delta : -delta;
        r.beta[i] = r.alpha[i] > 0.0 ? box.lo[i] : box.hi[i];
        if (std::fabs(r.beta[i]) < 0.5 * delta) r.beta[i] = 0.5 * r.alpha[i];
    }
    return r;
}

// A model whose points are the given offsets from x_k (row kopt is the zero
// vector), with nothing else the plan reads.
struct Scene {
    std::size_t n, m;
    Model q;
    Box box;
    Scene(std::size_t n_, std::size_t m_) : n(n_), m(m_), q(n_, m_), box(n_) {}
    void set_points(const Points& y, std::size_t kopt) {
        std::ranges::copy(y, q.points.begin());
        q.kopt = kopt;
    }
};

// An old point set around x_k: slot kopt at 0, the rest random within scale.
Points old_points(Draw& rng, std::size_t n, std::size_t m, std::size_t kopt, double scale) {
    Points y(m * n, 0.0);
    for (std::size_t j = 0; j < m; ++j) {
        if (j == kopt) continue;
        for (std::size_t k = 0; k < n; ++k) y[j * n + k] = scale * rng.next();
    }
    return y;
}

}  // namespace

TEST(V0121Rescue, TheMultipliersKeepTheNewPointsFeasibleAndSeparated) {
    // (5.4): alpha = Delta, beta = -Delta where both sides fit; otherwise alpha
    // is the side that fits and beta the distance to the other bound, or
    // alpha / 2 when that is under Delta / 2. Every old point is at x_k, so
    // nothing is old and every slot but the first is a gamma point.
    const std::size_t n = 3, m = 7;
    const double delta = 1.0;
    struct Case {
        double lo, hi;  // one coordinate's bounds as offsets from x_k; 0 = none
        bool has_lo, has_hi;
        double alpha, beta;
    };
    const Case cases[] = {
        {.lo = -5.0, .hi = 5.0, .has_lo = true, .has_hi = true, .alpha = 1.0, .beta = -1.0},
        {.lo = 0.0, .hi = 0.0, .has_lo = false, .has_hi = false, .alpha = 1.0, .beta = -1.0},
        {.lo = -0.7, .hi = 4.0, .has_lo = true, .has_hi = true, .alpha = 1.0, .beta = -0.7},
        {.lo = -0.3, .hi = 4.0, .has_lo = true, .has_hi = true, .alpha = 1.0, .beta = 0.5},
        {.lo = -4.0, .hi = 0.6, .has_lo = true, .has_hi = true, .alpha = -1.0, .beta = 0.6},
        {.lo = -4.0, .hi = 0.2, .has_lo = true, .has_hi = true, .alpha = -1.0, .beta = -0.5},
        {.lo = -0.5, .hi = 5.0, .has_lo = true, .has_hi = true, .alpha = 1.0, .beta = -0.5},
        {.lo = -1.0, .hi = 1.0, .has_lo = true, .has_hi = true, .alpha = 1.0, .beta = -1.0},
        {.lo = -2.0, .hi = 0.0, .has_lo = true, .has_hi = true, .alpha = -1.0, .beta = -0.5},
    };
    for (const Case& c : cases) {
        Scene s(n, m);
        // Coordinate 0 gets the case, the others have both sides free.
        if (c.has_lo) {
            s.box.has_lo[0] = 1;
            s.box.lo[0] = c.lo;
        }
        if (c.has_hi) {
            s.box.has_hi[0] = 1;
            s.box.hi[0] = c.hi;
        }
        s.set_points(Points(m * n, 0.0), 0);
        Inverse inv(n, m);
        const RescuePlan plan = rescue_plan(s.q, s.box, delta, inv);
        for (std::size_t j = 1; j < m; ++j) EXPECT_EQ(plan.origin[j], kNewPoint) << j;
        EXPECT_EQ(plan.origin[0], 0u);
        // Slot 1 is alpha_1 e_1 and slot n + 1 is beta_1 e_1 (the report's 1-based
        // gamma_2 and gamma_{n+2}), exactly.
        EXPECT_EQ(plan.points[1 * n + 0], c.alpha) << "lo " << c.lo << " hi " << c.hi;
        EXPECT_EQ(plan.points[(n + 1) * n + 0], c.beta) << "lo " << c.lo << " hi " << c.hi;
        // Feasible and separated: |alpha - beta| >= Delta / 2, |beta| >= Delta / 2.
        EXPECT_GE(std::fabs(c.alpha - c.beta), 0.5 * delta);
        if (c.has_lo) {
            EXPECT_GE(plan.points[1 * n], c.lo);
            EXPECT_GE(plan.points[(n + 1) * n], c.lo);
        }
        if (c.has_hi) {
            EXPECT_LE(plan.points[1 * n], c.hi);
            EXPECT_LE(plan.points[(n + 1) * n], c.hi);
        }
    }
}

TEST(V0121Rescue, WithNoOldPointsTheNewSetIsTheGammaSetAndTheInverseIsItsOwn) {
    Draw rng(0x6A33A5ULL);
    for (std::size_t n = 1; n <= 4; ++n)
        for (std::size_t m = n + 2; m <= max_m(n); ++m) {
            Scene s(n, m);
            // A box that squeezes some coordinates, x_k = 0 inside it.
            for (std::size_t i = 0; i < n; ++i) {
                if (rng.below(2) == 0) {
                    s.box.has_lo[i] = 1;
                    s.box.lo[i] = -(0.2 + 2.0 * std::fabs(rng.next()));
                }
                if (rng.below(2) == 0) {
                    s.box.has_hi[i] = 1;
                    s.box.hi[i] = 0.2 + 2.0 * std::fabs(rng.next());
                }
            }
            const double delta = 0.4 + 0.6 * std::fabs(rng.next());
            const std::size_t kopt = rng.below(m);
            s.set_points(Points(m * n, 0.0), kopt);
            Inverse inv(n, m);
            const RescuePlan plan = rescue_plan(s.q, s.box, delta, inv);
            const Multipliers mu = multipliers_54(s.box, n, delta);
            const Points gamma = v0121::coordinate_points(mu.alpha, mu.beta, m);
            ASSERT_EQ(plan.points.size(), gamma.size());
            for (std::size_t j = 0; j < m; ++j) {
                if (j == 0) {
                    EXPECT_EQ(plan.origin[0], kopt);
                } else {
                    EXPECT_EQ(plan.origin[j], kNewPoint)
                        << "n " << n << " m " << m << " slot " << j;
                }
                for (std::size_t k = 0; k < n; ++k)
                    EXPECT_TRUE(v0121::same_bits(plan.points[j * n + k], gamma[j * n + k]))
                        << "n " << n << " m " << m << " slot " << j << " coordinate " << k;
            }
            const Dense h = v0121::explicit_h(plan.points, n, m);
            ASSERT_EQ(h.size, m + n + 1);
            EXPECT_LT(v0121::compare_to_h(inv, h, n, m).worst(), 1e-9) << "n " << n << " m " << m;
        }
}

TEST(V0121Rescue, OldPointsThatAreTheGammaPointsAreAllKeptAndNothingIsNew) {
    // Each candidate is a point of the gamma set in another slot, so putting
    // it where it belongs has tau_t = 1 and beta = 0, sigma = 1, which passes
    // (5.7): the plan is a permutation of the old points, T is empty, and
    // only H changes.
    Draw rng(0x9E4D17ULL);
    for (std::size_t n = 1; n <= 3; ++n)
        for (const std::size_t m : {n + 2, 2 * n + 1, max_m(n)}) {
            Scene s(n, m);
            const double delta = 0.5;
            const Multipliers mu = multipliers_54(s.box, n, delta);
            const Points gamma = v0121::coordinate_points(mu.alpha, mu.beta, m);
            // Slot kopt keeps x_k = 0 = gamma_0; the others are the gamma points
            // shuffled (Fisher-Yates with the generator).
            std::vector<std::size_t> order(m - 1);
            for (std::size_t i = 0; i + 1 < m; ++i) order[i] = i + 1;
            for (std::size_t i = m - 2; i > 0; --i) std::swap(order[i], order[rng.below(i + 1)]);
            const std::size_t kopt = rng.below(m);
            Points y(m * n, 0.0);
            std::vector<std::size_t> gamma_of(m, 0);  // old slot -> gamma slot
            std::size_t next = 0;
            for (std::size_t j = 0; j < m; ++j) {
                if (j == kopt) continue;
                gamma_of[j] = order[next++];
                for (std::size_t k = 0; k < n; ++k) y[j * n + k] = gamma[gamma_of[j] * n + k];
            }
            s.set_points(y, kopt);
            Inverse inv(n, m);
            const RescuePlan plan = rescue_plan(s.q, s.box, delta, inv);
            for (std::size_t slot = 0; slot < m; ++slot) {
                ASSERT_NE(plan.origin[slot], kNewPoint) << "n " << n << " m " << m;
                for (std::size_t k = 0; k < n; ++k)
                    EXPECT_TRUE(v0121::same_bits(plan.points[slot * n + k], gamma[slot * n + k]));
            }
            // slot j holds the old point whose gamma slot is j.
            for (std::size_t j = 0; j < m; ++j) {
                if (j == kopt) continue;
                EXPECT_EQ(plan.origin[gamma_of[j]], j) << "n " << n << " m " << m;
            }
            EXPECT_EQ(plan.origin[0], kopt);
            const Dense h = v0121::explicit_h(plan.points, n, m);
            ASSERT_EQ(h.size, m + n + 1);
            EXPECT_LT(v0121::compare_to_h(inv, h, n, m).worst(), 1e-9);
        }
}

namespace {

// The iteration of section 5 written out again, with the explicit inverse of
// the current set at every step. Returns false when a decision of it was too
// close to call, which the comparison with the library then skips.
struct Transcript {
    std::vector<std::size_t> origin;
    Points points;
    bool clear = true;
};

Transcript transcribe(const Points& old_y, std::size_t kopt, const Points& gamma, std::size_t n,
                      std::size_t m) {
    Transcript t;
    t.points = gamma;
    t.origin.assign(m, kNewPoint);
    t.origin[0] = kopt;
    std::vector<double> psi(m, 0.0);
    double psi_max = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        if (i == kopt) continue;
        double s = 0.0;
        for (std::size_t k = 0; k < n; ++k) s += old_y[i * n + k] * old_y[i * n + k];
        psi[i] = std::sqrt(s);
        psi_max = std::max(psi_max, psi[i]);
    }
    for (;;) {
        std::vector<char> tried(m, 0);
        bool replaced = false;
        for (;;) {
            std::size_t ell = m;
            for (std::size_t i = 0; i < m; ++i)
                if (psi[i] > 0.0 && (ell == m || psi[i] < psi[ell])) ell = i;
            if (ell == m || tried[ell]) break;
            tried[ell] = 1;
            // A near tie between the two least scores is not a test of the rule.
            for (std::size_t i = 0; i < m; ++i)
                if (i != ell && psi[i] > 0.0 && std::fabs(psi[i] - psi[ell]) < 1e-9 * psi_max)
                    t.clear = false;
            const Dense h = v0121::explicit_h(t.points, n, m);
            if (h.size != m + n + 1) {
                t.clear = false;
                return t;
            }
            const std::vector<double> x(old_y.begin() + static_cast<std::ptrdiff_t>(ell * n),
                                        old_y.begin() + static_cast<std::ptrdiff_t>((ell + 1) * n));
            const std::vector<double> w = v0121::w_vector(t.points, n, m, x);
            const std::vector<double> hw = v0121::times(h, w);
            const double beta = 0.5 * std::pow(v0121::dot_all(x, x), 2) - v0121::dot_all(w, hw);
            std::size_t best = m;
            double best_sigma = 0.0, second_sigma = -1e300, tau_max = 0.0;
            for (std::size_t j = 1; j < m; ++j) {
                tau_max = std::max(tau_max, hw[j] * hw[j]);
                if (t.origin[j] != kNewPoint) continue;
                const double sigma = h(j, j) * beta + hw[j] * hw[j];
                if (best == m || sigma > best_sigma) {
                    second_sigma = best_sigma;
                    best = j;
                    best_sigma = sigma;
                } else if (sigma > second_sigma) {
                    second_sigma = sigma;
                }
            }
            const double scale = 1e-9 * (1.0 + std::fabs(best_sigma) + tau_max);
            if (best < m && best_sigma - second_sigma < scale && second_sigma > -1e299)
                t.clear = false;
            if (best < m && std::fabs(best_sigma - kSafe * tau_max) < scale) t.clear = false;
            if (best < m && best_sigma > kSafe * tau_max && best_sigma > 0.0) {
                for (std::size_t k = 0; k < n; ++k) t.points[best * n + k] = x[k];
                t.origin[best] = ell;
                psi[ell] = 0.0;
                replaced = true;
                break;
            }
            psi[ell] += psi_max;
        }
        if (!replaced) break;
    }
    return t;
}

}  // namespace

TEST(V0121Rescue, TheIterationIsTheOneOfSection5WrittenOutWithAnExplicitInverse) {
    Draw rng(0x5C0DE5ULL);
    std::size_t compared = 0, with_new = 0, with_old = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const std::size_t n = 1 + rng.below(3);
        const std::size_t m = n + 2 + rng.below(max_m(n) - n - 1);
        Scene s(n, m);
        for (std::size_t i = 0; i < n; ++i) {
            if (rng.below(3) == 0) {
                s.box.has_lo[i] = 1;
                s.box.lo[i] = -(0.1 + 1.5 * std::fabs(rng.next()));
            }
            if (rng.below(3) == 0) {
                s.box.has_hi[i] = 1;
                s.box.hi[i] = 0.1 + 1.5 * std::fabs(rng.next());
            }
        }
        const double delta = 0.3 + 0.7 * std::fabs(rng.next());
        const std::size_t kopt = rng.below(m);
        // Old points: some near x_k, some far, a few close to the gamma points.
        const Points y = old_points(rng, n, m, kopt, 1.0 + 2.0 * std::fabs(rng.next()));
        s.set_points(y, kopt);
        Inverse inv(n, m);
        const RescuePlan plan = rescue_plan(s.q, s.box, delta, inv);
        const Multipliers mu = multipliers_54(s.box, n, delta);
        const Points gamma = v0121::coordinate_points(mu.alpha, mu.beta, m);
        const Transcript t = transcribe(y, kopt, gamma, n, m);
        if (!t.clear) continue;
        ++compared;
        std::size_t olds = 0;
        for (std::size_t j = 0; j < m; ++j) {
            EXPECT_EQ(plan.origin[j], t.origin[j]) << "trial " << trial << " slot " << j;
            for (std::size_t k = 0; k < n; ++k)
                EXPECT_TRUE(v0121::same_bits(plan.points[j * n + k], t.points[j * n + k]))
                    << "trial " << trial << " slot " << j;
            if (plan.origin[j] != kNewPoint) ++olds;
        }
        if (olds < m) ++with_new;
        if (olds > 1) ++with_old;
        // The inverse left behind is that of the final set.
        const Dense h = v0121::explicit_h(plan.points, n, m);
        ASSERT_EQ(h.size, m + n + 1);
        EXPECT_LT(v0121::compare_to_h(inv, h, n, m).worst(), 1e-6) << "trial " << trial;
        // No old point is used twice, and x_k is slot 0.
        std::vector<char> used(m, 0);
        for (std::size_t j = 0; j < m; ++j) {
            if (plan.origin[j] == kNewPoint) continue;
            EXPECT_EQ(used[plan.origin[j]], 0) << "an old point twice, trial " << trial;
            used[plan.origin[j]] = 1;
        }
        EXPECT_EQ(plan.origin[0], kopt);
        for (std::size_t k = 0; k < n; ++k) EXPECT_EQ(plan.points[k], 0.0);
    }
    EXPECT_GT(compared, 200u);
    EXPECT_GT(with_new, 20u) << "plans that need new values must have been seen";
    EXPECT_GT(with_old, 20u) << "and plans that keep old points";
}

namespace {

double smooth(std::span<const double> x) {
    const std::size_t n = x.size();
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        s += std::cos(1.1 * x[i] + 0.3 * static_cast<double>(i)) + 0.4 * x[i] * x[i] +
             0.2 * x[i] * x[(i + 1) % n];
    return s;
}

}  // namespace

TEST(V0121Rescue, TheModelIsMadeToInterpolateTheNewPointsBy512AfterTheMoveOfOrigin53) {
    // The model before RESCUE is the least Frobenius norm interpolant of the
    // old values. The origin moves to x_k (the rank two matrix of (5.3) keeps
    // the second derivative matrix), the plan is made, the old points that are
    // not kept have their term of (4.6) moved into M, and each new point t
    // adds {F(y_t) - Q(y_t)} Lambda_t, (5.12), whose multipliers are column t
    // of Omega and whose gradient is row t of B. The result interpolates every
    // point of the new set and equals the explicit update with the inverse of
    // the new W. The composition is the report's, written here from the
    // library's pieces.
    Draw rng(0xE5CAFEULL);
    std::size_t trials_run = 0, with_new = 0;
    for (int trial = 0; trial < 120; ++trial) {
        const std::size_t n = 1 + rng.below(3);
        const std::size_t m = n + 2 + rng.below(max_m(n) - n - 1);
        // Old points about an origin that is not x_k.
        std::vector<double> x0(n);
        for (double& v : x0) v = 0.3 * rng.next();
        Points y(m * n);
        std::vector<double> fval(m);
        for (std::size_t j = 0; j < m; ++j) {
            std::vector<double> p(n);
            for (std::size_t k = 0; k < n; ++k) {
                y[j * n + k] = 0.9 * rng.next();
                p[k] = y[j * n + k] + x0[k];
            }
            fval[j] = smooth(p);
        }
        std::vector<double> lambda;
        v0121::Quadratic qx = v0121::least_norm_interpolant(y, n, m, fval, lambda);
        if (std::ranges::all_of(lambda, [](double v) { return v == 0.0; })) continue;
        Model q(n, m);
        std::ranges::copy(y, q.points.begin());
        std::ranges::copy(fval, q.fval.begin());
        for (std::size_t l = 0; l < m; ++l) q.mu[l] = lambda[l];
        q.kopt = 0;
        for (std::size_t j = 1; j < m; ++j)
            if (q.fval[j] < q.fval[q.kopt]) q.kopt = j;
        {
            const std::vector<double> gk = qx.gradient(q.xopt());
            std::ranges::copy(gk, q.g.begin());
        }
        // The origin moves to x_k.
        const std::vector<double> xk(q.xopt().begin(), q.xopt().end());
        std::vector<double> work(n);
        q.shift_hessian(xk, work);
        for (std::size_t j = 0; j < m; ++j)
            for (std::size_t k = 0; k < n; ++k) q.points[j * n + k] -= xk[k];
        Box box(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (rng.below(3) == 0) {
                box.has_lo[i] = 1;
                box.lo[i] = -(0.2 + 1.5 * std::fabs(rng.next()));
            }
            if (rng.below(3) == 0) {
                box.has_hi[i] = 1;
                box.hi[i] = 0.2 + 1.5 * std::fabs(rng.next());
            }
        }
        const double delta = 0.3 + 0.6 * std::fabs(rng.next());
        Inverse inv(n, m);
        const RescuePlan plan = rescue_plan(q, box, delta, inv);

        // The composition of the report.
        std::vector<char> kept(m, 0);
        for (std::size_t j = 0; j < m; ++j)
            if (plan.origin[j] != kNewPoint) kept[plan.origin[j]] = 1;
        for (std::size_t i = 0; i < m; ++i)
            if (!kept[i]) q.absorb(i);
        std::vector<double> nf(m, 0.0), nmu(m, 0.0);
        std::vector<std::size_t> fresh;
        for (std::size_t j = 0; j < m; ++j) {
            if (plan.origin[j] == kNewPoint) {
                fresh.push_back(j);
            } else {
                nf[j] = q.fval[plan.origin[j]];
                nmu[j] = q.mu[plan.origin[j]];
            }
        }
        std::ranges::copy(plan.points, q.points.begin());
        std::ranges::copy(nf, q.fval.begin());
        std::ranges::copy(nmu, q.mu.begin());
        q.kopt = 0;
        // The explicit quantities for the new set, about the origin x_k: the old
        // quadratic expanded about x_k, then (5.12) for each new point in turn
        // with the explicit H of the new set.
        const Dense hnew = v0121::explicit_h(plan.points, n, m);
        ASSERT_EQ(hnew.size, m + n + 1);
        v0121::Quadratic ex(n);
        ex.c = qx.at(xk);
        ex.g = qx.gradient(xk);
        ex.big = qx.big;
        std::vector<double> vals(m, 0.0);
        for (const std::size_t t : fresh) {
            std::vector<double> p(n);
            for (std::size_t k = 0; k < n; ++k) p[k] = plan.points[t * n + k] + xk[k] + x0[k];
            vals[t] = smooth(p);
        }
        for (const std::size_t t : fresh) {
            // The library's pieces, in the order of (5.12).
            const std::span<const double> yt = q.point(t);
            const double diff = vals[t] - q.fopt() - q.change(yt, work);
            std::vector<double> lam(m);
            inv.omega_col(t, lam);
            for (std::size_t l = 0; l < m; ++l) q.mu[l] += diff * lam[l];
            for (std::size_t k = 0; k < n; ++k) q.g[k] += diff * inv.b(t, k);
            q.fval[t] = vals[t];
            // The same with the explicit inverse.
            const std::vector<double> ytv(yt.begin(), yt.end());
            const double explicit_diff = vals[t] - ex.at(ytv);
            EXPECT_NEAR(diff, explicit_diff, 1e-7 * (1.0 + std::fabs(explicit_diff)))
                << "trial " << trial;
            v0121::add_change(ex, hnew, plan.points, n, m, t, explicit_diff);
        }
        // x_k is the best point again; a new point that is lower moves it.
        std::size_t kopt = 0;
        for (std::size_t j = 1; j < m; ++j)
            if (q.fval[j] < q.fval[kopt]) kopt = j;
        if (kopt != 0) {
            q.hess_mul(q.point(kopt), work);
            for (std::size_t k = 0; k < n; ++k) q.g[k] += work[k];
            q.kopt = kopt;
        }
        // Interpolation at every point of the new set.
        double spread = 1.0;
        for (std::size_t j = 0; j < m; ++j) spread = std::max(spread, std::fabs(q.fval[j]));
        std::vector<double> dk(n);
        for (std::size_t j = 0; j < m; ++j) {
            for (std::size_t k = 0; k < n; ++k) dk[k] = q.point(j)[k] - q.xopt()[k];
            EXPECT_NEAR(q.fopt() + q.change(dk, work), q.fval[j], 1e-7 * spread)
                << "trial " << trial << " point " << j;
        }
        // And the model is the explicit one at random points.
        for (int probe = 0; probe < 3; ++probe) {
            std::vector<double> z(n);
            for (std::size_t k = 0; k < n; ++k) {
                z[k] = 0.8 * rng.next();
                dk[k] = z[k] - q.xopt()[k];
            }
            const double ql = q.fopt() + q.change(dk, work);
            EXPECT_NEAR(ql, ex.at(z), 1e-6 * (1.0 + std::fabs(ql))) << "trial " << trial;
        }
        ++trials_run;
        if (!fresh.empty()) ++with_new;
    }
    EXPECT_GT(trials_run, 60u);
    EXPECT_GT(with_new, 20u);
}
