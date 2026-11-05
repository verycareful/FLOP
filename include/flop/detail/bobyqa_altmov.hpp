// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// bobyqa_altmov - the step of an "alternative" iteration (section 3, ALTMOV)
// =============================================================================
//
// Powell, DAMTP 2009/NA06, section 3, second half. The point y_t to be
// replaced is chosen first (3.7); the step aims at a large |Lambda_t(x_k +
// d)|, Lambda_t the Lagrange function of (3.8), so that the denominator
// (3.9) of the update of H is large.
//
//   line step    for each interpolation point y_j != x_k, phi_j(alpha) =
//                Lambda_t(x_k + alpha (y_j - x_k)) is a quadratic known from
//                phi_j(0) = 0, phi_j'(0) and phi_j(1) = delta_jt; alpha_j
//                maximises |phi_j| in the box and the trust region, and d is
//                the alpha_j (y_j - x_k) that maximises (3.11).
//   Cauchy step  for Lambda_t and for -Lambda_t: the exact solution of the
//                linear problem (3.13) by the active-set iteration after
//                (3.14)-(3.15), then the multiple of it that minimises the
//                function in [0, 1]. c is the one of the two with the larger
//                |Lambda_t|.
//
// The caller compares (3.12) once the update parameters of d are known.

#pragma once

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "flop/detail/bobyqa_model.hpp"

namespace flop::detail::bobyqa {

struct AltmovWork {
    AltmovWork(std::size_t n, std::size_t m)
        : lambda(m),
          glag(n),
          dir(n),
          s(n),
          best_s(n),
          shat(n),
          hv(n),
          inset(n),
          sside(n),
          best_side(n) {}
    std::vector<double> lambda, glag, dir, s, best_s, shat, hv;
    std::vector<char> inset;
    std::vector<Side> sside, best_side;
};

namespace altmov_detail {

// out = (sum_l lambda_l y_l y_l^T) v, the second derivative of Lambda_t
// (4.16) times v.
inline void lag_hess_mul(const Model& model, std::span<const double> lambda,
                         std::span<const double> v, std::span<double> out) noexcept {
    for (std::size_t i = 0; i < model.n; ++i) out[i] = 0.0;
    for (std::size_t l = 0; l < model.m; ++l) {
        std::span<const double> yl = model.point(l);
        const double c = lambda[l] * dot(yl, v);
        for (std::size_t i = 0; i < model.n; ++i) out[i] += c * yl[i];
    }
}

}  // namespace altmov_detail

// d, dside: the line step; c, cside: the Cauchy step. Returns
// Lambda_t(x_k + c)^2, the left side of (3.12).
inline double altmov(const Model& model, const Inverse& inv, const Box& box, std::size_t t,
                     double delta, std::span<double> d, std::span<Side> dside, std::span<double> c,
                     std::span<Side> cside, AltmovWork& w) {
    const std::size_t n = model.n, m = model.m;
    std::span<const double> xk = model.xopt();

    // grad Lambda_t(x_k) = B row t + (nabla^2 Lambda_t) x_k (section 4,
    // after (4.17)).
    inv.omega_col(t, w.lambda);
    altmov_detail::lag_hess_mul(model, w.lambda, xk, w.glag);
    for (std::size_t i = 0; i < n; ++i) w.glag[i] += inv.b(t, i);
    const double htt = inv.omega_diag(t);

    // ========================================================== the line step
    bool have = false;
    double best_score = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        d[i] = 0.0;
        dside[i] = Side::Free;
    }
    for (std::size_t j = 0; j < m; ++j) {
        if (j == model.kopt) continue;
        std::span<const double> yj = model.point(j);
        double len2 = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            w.dir[i] = yj[i] - xk[i];
            len2 += w.dir[i] * w.dir[i];
        }
        if (!(len2 > 0.0)) continue;
        const double len = std::sqrt(len2);
        // The feasible alpha: |alpha| len <= Delta and the box, coordinate
        // by coordinate. hi_at / lo_at name the coordinate and side that
        // binds at each end, if a bound does.
        double ahi = delta / len, alo = -ahi;
        std::size_t hi_at = n, lo_at = n;
        Side hi_side = Side::Free, lo_side = Side::Free;
        for (std::size_t i = 0; i < n; ++i) {
            const double di = w.dir[i];
            if (di == 0.0) continue;
            if (box.has_hi[i]) {
                const double r = (box.hi[i] - xk[i]) / di;  // >= 0 when di > 0
                if (di > 0.0 && r < ahi) {
                    ahi = r;
                    hi_at = i;
                    hi_side = Side::Upper;
                }
                if (di < 0.0 && r > alo) {
                    alo = r;
                    lo_at = i;
                    lo_side = Side::Upper;
                }
            }
            if (box.has_lo[i]) {
                const double r = (box.lo[i] - xk[i]) / di;
                if (di < 0.0 && r < ahi) {
                    ahi = r;
                    hi_at = i;
                    hi_side = Side::Lower;
                }
                if (di > 0.0 && r > alo) {
                    alo = r;
                    lo_at = i;
                    lo_side = Side::Lower;
                }
            }
        }
        if (ahi < 0.0) ahi = 0.0;
        if (alo > 0.0) alo = 0.0;
        // phi(alpha) = slope alpha + (delta_jt - slope) alpha^2.
        const double slope = dot(w.dir, w.glag);
        const double curv = (j == t ? 1.0 : 0.0) - slope;
        const auto phi = [&](double a) { return a * (slope + curv * a); };
        double a_best = ahi, phi_best = phi(ahi);
        std::size_t at = hi_at;
        Side at_side = hi_side;
        if (std::fabs(phi(alo)) > std::fabs(phi_best)) {
            a_best = alo;
            phi_best = phi(alo);
            at = lo_at;
            at_side = lo_side;
        }
        if (curv != 0.0) {
            const double astar = -slope / (2.0 * curv);
            if (astar > alo && astar < ahi && std::fabs(phi(astar)) > std::fabs(phi_best)) {
                a_best = astar;
                phi_best = phi(astar);
                at = n;
                at_side = Side::Free;
            }
        }
        // (3.11): phi^2 [ (1/2) H_tt alpha^2 (1 - alpha)^2 |y_j - x_k|^4 + phi^2 ].
        const double om = 1.0 - a_best;
        const double phi2 = phi_best * phi_best;
        const double score = phi2 * (0.5 * htt * a_best * a_best * om * om * len2 * len2 + phi2);
        if (!have || score > best_score) {
            have = true;
            best_score = score;
            for (std::size_t i = 0; i < n; ++i) {
                d[i] = a_best * w.dir[i];
                dside[i] = Side::Free;
            }
            if (at < n) dside[at] = at_side;
        }
    }

    // ========================================================== the Cauchy step
    double best_abs = -1.0;
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = 0.0;
        cside[i] = Side::Free;
    }
    const double delta2 = delta * delta;
    for (int sign = 1; sign >= -1; sign -= 2) {
        // (3.14) for the function sign * Lambda_t: s-hat runs to the bound
        // against the gradient; S starts as the coordinates where s-hat is 0.
        for (std::size_t i = 0; i < n; ++i) {
            const double gi = sign * w.glag[i];
            w.inset[i] = 0;
            w.sside[i] = Side::Free;
            w.shat[i] = 0.0;
            if (gi == 0.0) {
                w.inset[i] = 1;
                continue;
            }
            if (gi > 0.0 && box.has_lo[i]) {
                w.shat[i] = box.lo[i] - xk[i];
                w.sside[i] = Side::Lower;
                if (!(w.shat[i] < 0.0)) {
                    w.shat[i] = 0.0;
                    w.inset[i] = 1;
                }
            } else if (gi < 0.0 && box.has_hi[i]) {
                w.shat[i] = box.hi[i] - xk[i];
                w.sside[i] = Side::Upper;
                if (!(w.shat[i] > 0.0)) {
                    w.shat[i] = 0.0;
                    w.inset[i] = 1;
                }
            }
        }
        // (3.15): s_i = s-hat_i on S, -mu g_i off it with |s| = Delta; the
        // coordinates that pass their bound join S, until none does.
        for (;;) {
            double in2 = 0.0, g2 = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double gi = sign * w.glag[i];
                if (w.inset[i])
                    in2 += w.shat[i] * w.shat[i];
                else
                    g2 += gi * gi;
            }
            double mu = 0.0;
            if (g2 > 0.0 && delta2 > in2) mu = std::sqrt((delta2 - in2) / g2);
            bool added = false;
            for (std::size_t i = 0; i < n; ++i) {
                const double gi = sign * w.glag[i];
                if (w.inset[i]) {
                    w.s[i] = w.shat[i];
                    continue;
                }
                w.s[i] = -mu * gi;
                if (w.sside[i] != Side::Free && std::fabs(w.s[i]) > std::fabs(w.shat[i])) {
                    w.inset[i] = 1;
                    added = true;
                }
            }
            if (!added) break;
        }
        // Coordinates of S on a bound: the sides; the step's length can
        // exceed Delta only when mu has no room left, and then it is scaled
        // back into the trust region.
        double s2 = dot(w.s, w.s);
        double scale = 1.0;
        if (s2 > delta2) scale = delta / std::sqrt(s2);
        if (!(s2 > 0.0)) continue;
        // Lambda_t along gamma s for gamma in [0, 1]: g.s gamma + (1/2) s^T
        // (nabla^2 Lambda_t) s gamma^2, for sign * Lambda_t.
        for (std::size_t i = 0; i < n; ++i) w.s[i] *= scale;
        altmov_detail::lag_hess_mul(model, w.lambda, w.s, w.hv);
        const double lin = sign * dot(w.glag, w.s);
        const double crv = sign * dot(w.s, w.hv);
        double gamma = 1.0;
        if (crv > 0.0 && -lin < crv) gamma = -lin / crv;
        if (gamma < 0.0) gamma = 0.0;
        const double val = gamma * lin + 0.5 * gamma * gamma * crv;
        const double mag = std::fabs(val);
        if (mag > best_abs) {
            best_abs = mag;
            for (std::size_t i = 0; i < n; ++i) {
                w.best_s[i] = gamma * w.s[i];
                w.best_side[i] = (gamma == 1.0 && scale == 1.0 && w.inset[i] && w.shat[i] != 0.0)
                                     ? w.sside[i]
                                     : Side::Free;
            }
        }
    }
    if (best_abs < 0.0) return 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = w.best_s[i];
        cside[i] = w.best_side[i];
    }
    return best_abs * best_abs;
}

}  // namespace flop::detail::bobyqa
