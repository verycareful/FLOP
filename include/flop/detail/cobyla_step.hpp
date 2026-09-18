// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// cobyla_step - the trust-region subproblem of COBYLA
// =============================================================================
//
// Given the linear models at the base point b,
//
//     c_k(b + d) ~ cval[k] + G_k . d      (feasible when >= 0, k < m)
//     f(b + d)   ~ f_b     + g . d
//
// find the step d with |d| <= rho that, in Powell's two stages, first makes
// the largest violation of the linearised constraints as small as it can be
// within the ball and then, holding the violation at that level, makes the
// linearised objective as small as it can be (Powell 1994, section 3). Box
// bounds on the variables are respected by every trial point.
//
// Two solvers, chosen by the constraint count:
//
//   m == 0   Closed form. Without bounds, d is the steepest-descent step to
//            the ball boundary. With bounds, the same step is projected onto
//            the box one coordinate at a time: a coordinate that would leave
//            the box is clamped to its bound, and the remaining radius is
//            redistributed over the free coordinates until every coordinate
//            fits. Every point is inside the box and inside the ball, and the
//            decrease is a fixed fraction of the unconstrained one, which is
//            what the trust-region argument needs. This is the whole of the
//            unconstrained path.
//
//   m > 0    Exact to tolerance, by bisection over an inner projection. Box
//            bounds join the constraint set as linear rows. Stage 1 bisects on
//            the violation level s and asks, at each s, whether the polyhedron
//            {violation_k <= s for all k} reaches into the ball, by projecting
//            the origin onto it. Stage 2 bisects on the multiplier nu of the
//            ball: d(nu) = proj_P(-g / nu) is the minimiser of g.d + nu/2 |d|^2
//            over the polyhedron P, |d(nu)| is non-increasing in nu because
//            it is the derivative of a concave dual function, and the step is
//            d at the nu where |d(nu)| = rho, or the smallest nu tried when
//            the objective is bounded on P inside the ball. The projections
//            are Hildreth's cyclic dual coordinate ascent, warm-started across
//            the bisection. Simple and correct rather than fast: this path is
//            the first candidate for an active-set solver later.
//
// No infinity, no NaN: "no bound" is an empty optional, a zero gradient is a
// zero-length step, and every division has its denominator tested first.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "flop/bounds.hpp"

namespace flop::detail::cobyla {

inline double dot(std::span<const double> a, std::span<const double> b) noexcept {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

inline double norm(std::span<const double> a) noexcept {
    return std::sqrt(dot(a, a));
}

struct StepWorkspace {
    std::vector<double> A;         // rows of the polyhedron, row-major, n per row
    std::vector<double> b;         // right-hand sides: A_r . d <= b_r
    std::vector<double> rownorm2;  // |A_r|^2
    std::vector<double> lambda;    // Hildreth multipliers, kept warm across calls
    std::vector<double> q;         // the point whose projection is sought, negated
    std::vector<double> trial;
};

// Hildreth's method: d = argmin |d + q|^2 over {A d <= b}, that is the
// projection of -q onto the polyhedron. lambda is the dual iterate, read on
// entry for a warm start and left for the next call. Returns false when the
// sweep cap is hit before the update falls below tol, or when a zero row
// cannot be satisfied (an empty polyhedron).
inline bool project_polyhedron(std::size_t n, std::size_t rows, const std::vector<double>& A,
                               const std::vector<double>& b, const std::vector<double>& rownorm2,
                               const std::vector<double>& q, std::vector<double>& lambda,
                               std::vector<double>& d, int max_sweeps, double tol) {
    d.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) d[i] = -q[i];
    for (std::size_t r = 0; r < rows; ++r) {
        if (lambda[r] == 0.0) continue;
        for (std::size_t i = 0; i < n; ++i) d[i] -= lambda[r] * A[r * n + i];
    }
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double max_change = 0.0;
        for (std::size_t r = 0; r < rows; ++r) {
            if (rownorm2[r] == 0.0) {
                if (b[r] < 0.0) return false;  // 0 <= b_r with b_r < 0: nothing satisfies it
                continue;
            }
            double ad = 0.0;
            for (std::size_t i = 0; i < n; ++i) ad += A[r * n + i] * d[i];
            const double step = (ad - b[r]) / rownorm2[r];
            const double next = std::max(0.0, lambda[r] + step);
            const double change = next - lambda[r];
            if (change == 0.0) continue;
            lambda[r] = next;
            for (std::size_t i = 0; i < n; ++i) d[i] -= change * A[r * n + i];
            max_change = std::max(max_change, std::fabs(change) * std::sqrt(rownorm2[r]));
        }
        if (max_change <= tol * (1.0 + norm(d))) return true;
    }
    return false;
}

// The unconstrained step: steepest descent to the ball boundary, then clamped
// into the box coordinate by coordinate with the radius redistributed.
inline void unconstrained_step(std::size_t n, std::span<const double> xb, std::span<const double> g,
                               double rho, const Bounds* bounds, std::span<double> d,
                               std::vector<char>& fixed) {
    fixed.assign(n, 0);
    for (std::size_t i = 0; i < n; ++i) d[i] = 0.0;
    for (std::size_t round = 0; round <= n; ++round) {
        double gnorm2 = 0.0, used2 = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            if (fixed[i])
                used2 += d[i] * d[i];
            else
                gnorm2 += g[i] * g[i];
        }
        if (gnorm2 == 0.0) return;
        const double remaining2 = rho * rho - used2;
        if (!(remaining2 > 0.0)) return;
        const double scale = -std::sqrt(remaining2) / std::sqrt(gnorm2);
        bool clamped = false;
        for (std::size_t i = 0; i < n; ++i) {
            if (fixed[i]) continue;
            d[i] = scale * g[i];
            if (!bounds) continue;
            if (bounds->lower[i] && xb[i] + d[i] < *bounds->lower[i]) {
                d[i] = *bounds->lower[i] - xb[i];
                fixed[i] = 1;
                clamped = true;
            } else if (bounds->upper[i] && xb[i] + d[i] > *bounds->upper[i]) {
                d[i] = *bounds->upper[i] - xb[i];
                fixed[i] = 1;
                clamped = true;
            }
        }
        if (!clamped) return;
    }
}

// Fills ws.A / ws.b with the constraint rows at violation level s: for each
// constraint, -(cval_k + G_k . d) <= s, and for each bound the distance from
// the base. Rows for the constraints are written first, the bound rows once
// and left in place; only the b of the constraint rows depends on s.
inline std::size_t build_polyhedron(std::size_t n, std::size_t m, std::span<const double> xb,
                                    std::span<const double> cval, std::span<const double> G,
                                    const Bounds* bounds, double s, StepWorkspace& ws) {
    std::size_t rows = m;
    if (bounds)
        for (std::size_t i = 0; i < n; ++i) {
            if (bounds->lower[i]) ++rows;
            if (bounds->upper[i]) ++rows;
        }
    ws.A.assign(rows * n, 0.0);
    ws.b.assign(rows, 0.0);
    ws.rownorm2.assign(rows, 0.0);
    for (std::size_t k = 0; k < m; ++k) {
        double nn = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            ws.A[k * n + i] = -G[k * n + i];
            nn += G[k * n + i] * G[k * n + i];
        }
        ws.b[k] = cval[k] + s;
        ws.rownorm2[k] = nn;
    }
    std::size_t r = m;
    if (bounds) {
        for (std::size_t i = 0; i < n; ++i) {
            const std::optional<double>& lo = bounds->lower[i];
            const std::optional<double>& hi = bounds->upper[i];
            if (lo.has_value()) {
                ws.A[r * n + i] = -1.0;
                ws.b[r] = xb[i] - *lo;
                ws.rownorm2[r] = 1.0;
                ++r;
            }
            if (hi.has_value()) {
                ws.A[r * n + i] = 1.0;
                ws.b[r] = *hi - xb[i];
                ws.rownorm2[r] = 1.0;
                ++r;
            }
        }
    }
    return rows;
}

inline void set_level(std::size_t m, std::span<const double> cval, double s, StepWorkspace& ws) {
    for (std::size_t k = 0; k < m; ++k) ws.b[k] = cval[k] + s;
}

// Largest violation of the linearised constraints at d, floored at zero.
inline double predicted_violation(std::size_t n, std::size_t m, std::span<const double> cval,
                                  std::span<const double> G, std::span<const double> d) noexcept {
    double worst = 0.0;
    for (std::size_t k = 0; k < m; ++k) {
        double v = cval[k];
        for (std::size_t i = 0; i < n; ++i) v += G[k * n + i] * d[i];
        worst = std::max(worst, -v);
    }
    return worst;
}

constexpr int kProjectionSweeps = 500;
constexpr double kProjectionTol = 1e-13;
constexpr int kBisectionSteps = 60;
constexpr double kBallTol =
    1e-10;  // |d| within this relative distance of rho counts as on the ball

// The constrained step. Writes d; returns the violation level the step was
// computed for (Powell's resmax of the step), which the caller uses as the
// predicted violation.
inline double constrained_step(std::size_t n, std::size_t m, std::span<const double> xb,
                               std::span<const double> cval, std::span<const double> G,
                               std::span<const double> g, double rho, const Bounds* bounds,
                               std::span<double> d, StepWorkspace& ws) {
    const std::size_t rows = build_polyhedron(n, m, xb, cval, G, bounds, 0.0, ws);
    ws.lambda.assign(rows, 0.0);
    ws.q.assign(n, 0.0);
    std::vector<double>& trial = ws.trial;

    // Stage 1: the smallest violation level the ball can reach. s_hi is the
    // violation at d = 0, feasible by construction; s_lo bounds every point of
    // the ball from below, constraint by constraint.
    double s_hi = 0.0, s_lo = 0.0;
    for (std::size_t k = 0; k < m; ++k) {
        double gn = 0.0;
        for (std::size_t i = 0; i < n; ++i) gn += G[k * n + i] * G[k * n + i];
        gn = std::sqrt(gn);
        s_hi = std::max(s_hi, -cval[k]);
        s_lo = std::max(s_lo, -cval[k] - rho * gn);
    }
    double level = 0.0;
    if (s_hi > 0.0) {
        double lo = std::max(0.0, s_lo), hi = s_hi;
        std::vector<double> best_lambda(rows, 0.0);
        for (int it = 0; it < kBisectionSteps && hi - lo > 1e-12 * (1.0 + hi); ++it) {
            const double mid = 0.5 * (lo + hi);
            set_level(m, cval, mid, ws);
            ws.lambda = best_lambda;
            const bool ok = project_polyhedron(n, rows, ws.A, ws.b, ws.rownorm2, ws.q, ws.lambda,
                                               trial, kProjectionSweeps, kProjectionTol);
            if (ok && norm(trial) <= rho * (1.0 + kBallTol)) {
                hi = mid;
                best_lambda = ws.lambda;
            } else {
                lo = mid;
            }
        }
        level = hi;
        ws.lambda = best_lambda;
    }
    set_level(m, cval, level, ws);

    // Stage 2: minimise g . d over the polyhedron at that level, inside the
    // ball, by the multiplier of the ball.
    const double gnorm = norm(g);
    if (gnorm == 0.0) {
        project_polyhedron(n, rows, ws.A, ws.b, ws.rownorm2, ws.q, ws.lambda, trial,
                           kProjectionSweeps, kProjectionTol);
        std::copy(trial.begin(), trial.end(), d.begin());
        return level;
    }
    auto step_at = [&](double nu) {
        for (std::size_t i = 0; i < n; ++i) ws.q[i] = g[i] / nu;
        project_polyhedron(n, rows, ws.A, ws.b, ws.rownorm2, ws.q, ws.lambda, trial,
                           kProjectionSweeps, kProjectionTol);
        return norm(trial);
    };
    double nu_hi = gnorm / rho;
    double r_hi = step_at(nu_hi);
    int grow = 0;
    while (r_hi > rho * (1.0 + kBallTol) && grow < kBisectionSteps) {
        nu_hi *= 2.0;
        r_hi = step_at(nu_hi);
        ++grow;
    }
    if (r_hi > rho * (1.0 + kBallTol)) {
        // The polyhedron's nearest point sits outside the ball at this
        // precision: keep the origin, a zero step the caller treats as short.
        for (std::size_t i = 0; i < n; ++i) d[i] = 0.0;
        return level;
    }
    double nu_lo = nu_hi * 1e-14;
    double r_lo = step_at(nu_lo);
    if (r_lo <= rho * (1.0 + kBallTol)) {
        std::copy(trial.begin(), trial.end(), d.begin());
        return level;
    }
    std::vector<double> best = trial;
    for (int it = 0; it < kBisectionSteps; ++it) {
        const double nu = std::sqrt(nu_lo * nu_hi);
        const double r = step_at(nu);
        if (r > rho * (1.0 + kBallTol)) {
            nu_lo = nu;
        } else {
            nu_hi = nu;
            best = trial;
            if (r >= rho * (1.0 - kBallTol)) break;
        }
    }
    std::copy(best.begin(), best.end(), d.begin());
    return level;
}

// The step, either path. Returns the predicted violation of the linearised
// constraints at d.
inline double trust_region_step(std::size_t n, std::size_t m, std::span<const double> xb,
                                std::span<const double> cval, std::span<const double> G,
                                std::span<const double> g, double rho, const Bounds* bounds,
                                std::span<double> d, StepWorkspace& ws, std::vector<char>& fixed) {
    if (m == 0) {
        unconstrained_step(n, xb, g, rho, bounds, d, fixed);
        return 0.0;
    }
    constrained_step(n, m, xb, cval, G, g, rho, bounds, d, ws);
    return predicted_violation(n, m, cval, G, d);
}

}  // namespace flop::detail::cobyla
