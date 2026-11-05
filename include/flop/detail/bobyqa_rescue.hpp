// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// bobyqa_rescue - the choice of points in RESCUE (report section 5)
// =============================================================================
//
// Powell, DAMTP 2009/NA06, section 5. With the origin already at x_k (the
// caller shifts it and the model, and discards H), H and Z are built afresh
// for the points gamma_j of (5.4)-(5.5), which are known exactly by section
// 2, and old points then replace new ones one at a time while the
// replacement is safe by (5.7): the slot t maximising (5.6) among those
// still holding a new point, accepted when its sigma exceeds kRescueSafe
// times the largest tau_j^2 over j != s. The candidates come in order of
// the least positive score psi, the distance from x_k, raised by the
// greatest score whenever a candidate fails; a candidate tried twice in one
// round ends the procedure.
//
// The result is the slot layout of the next point set: slot 0 is x_k, and
// each slot holds either an old point (its index) or a new gamma point not
// yet evaluated. H and Z on return are those of that set.

#pragma once

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "flop/detail/bobyqa_model.hpp"

namespace flop::detail::bobyqa {

// (5.7): the multiplier, the report's own.
constexpr double kRescueSafe = 0.01;

inline constexpr std::size_t kNewPoint = static_cast<std::size_t>(-1);

struct RescuePlan {
    std::vector<double> points;       // m x n, offsets from x_k
    std::vector<std::size_t> origin;  // per slot: the old index, or kNewPoint
};

// model.points are offsets from x_k (model.xopt() is zero). delta is
// Delta_k. inv is overwritten.
inline RescuePlan rescue_plan(const Model& model, const Box& box, double delta, Inverse& inv) {
    const std::size_t n = model.n, m = model.m;
    RescuePlan plan;
    plan.points.assign(m * n, 0.0);
    plan.origin.assign(m, kNewPoint);
    plan.origin[0] = model.kopt;

    // (5.4): alpha_i = +-Delta, beta_i = -+Delta when both sides fit;
    // otherwise alpha_i is the side that fits and beta_i the distance to the
    // other bound, or alpha_i / 2 if that distance is under Delta / 2.
    std::vector<double> alpha(n), beta(n);
    for (std::size_t i = 0; i < n; ++i) {
        const bool up = !box.has_hi[i] || delta <= box.hi[i];
        const bool down = !box.has_lo[i] || -delta >= box.lo[i];
        if (up && down) {
            alpha[i] = delta;
            beta[i] = -delta;
            continue;
        }
        alpha[i] = up ? delta : -delta;
        beta[i] = alpha[i] > 0.0 ? box.lo[i] : box.hi[i];
        if (std::fabs(beta[i]) < 0.5 * delta) beta[i] = 0.5 * alpha[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
        plan.points[(i + 1) * n + i] = alpha[i];
        if (n + i + 1 < m) plan.points[(n + i + 1) * n + i] = beta[i];
    }
    for (std::size_t j = 2 * n + 1; j < m; ++j) {
        const Pair pq = extra_pair(n, j - 2 * n - 1);
        plan.points[j * n + pq.p] = alpha[pq.p];
        plan.points[j * n + pq.q] = alpha[pq.q];
    }
    inv.set_coordinate(alpha, beta);

    // Scores: |y_i - x_k| for the old points, 0 for x_k, which is in.
    std::vector<double> psi(m, 0.0);
    double psi_max = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        if (i == model.kopt) continue;
        psi[i] = std::sqrt(dot(model.point(i), model.point(i)));
        if (psi[i] > psi_max) psi_max = psi[i];
    }
    std::vector<double> vlag(m + n), wv(m), zw(inv.columns()), d(n), sum(n), h(m + n), u(m + n);
    std::vector<char> tried(m, 0);
    for (;;) {
        for (char& x : tried) x = 0;
        bool replaced = false;
        for (;;) {
            // The least positive score.
            std::size_t ell = m;
            for (std::size_t i = 0; i < m; ++i)
                if (psi[i] > 0.0 && (ell == m || psi[i] < psi[ell])) ell = i;
            if (ell == m || tried[ell]) break;
            tried[ell] = 1;
            std::span<const double> y = model.point(ell);
            const double beta_l = inv.vlag_beta(plan.points, 0, y, vlag, wv, zw, d, sum);
            std::size_t best = m;
            double best_sigma = 0.0;
            double tau_max = 0.0;
            for (std::size_t j = 1; j < m; ++j) {
                tau_max = std::fmax(tau_max, vlag[j] * vlag[j]);
                if (plan.origin[j] != kNewPoint) continue;
                const double sigma = inv.omega_diag(j) * beta_l + vlag[j] * vlag[j];
                if (best == m || sigma > best_sigma) {
                    best = j;
                    best_sigma = sigma;
                }
            }
            if (best < m && best_sigma > kRescueSafe * tau_max && best_sigma > 0.0) {
                inv.update(best, vlag, beta_l, h, u);
                for (std::size_t k = 0; k < n; ++k) plan.points[best * n + k] = y[k];
                plan.origin[best] = ell;
                psi[ell] = 0.0;
                replaced = true;
                break;
            }
            psi[ell] += psi_max;
        }
        if (!replaced) break;
    }
    return plan;
}

}  // namespace flop::detail::bobyqa
