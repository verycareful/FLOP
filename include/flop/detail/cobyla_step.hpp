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
// linearised objective as small as it can be, taking the step of least norm
// when several qualify (Powell 1994, section 2, the definition of x*). The
// paper gives that definition and one sentence on its implementation; the
// solvers here are FLOP's own for that definition. Box bounds on the
// variables are respected by every trial point.
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
//   m > 0    Exact. Box bounds join the constraint set as linear rows that
//            are never relaxed. Both of Powell's stages are a linear
//            objective over the polyhedron P of those rows intersected with
//            the ball, and each is solved by following an exact projection
//            onto P (active_set.hpp) along one scalar parameter.
//
//            Stage 1 finds the least violation level t at which P_t, the
//            polyhedron with the constraint rows relaxed by t, reaches the
//            ball: dist(0, P_t) is nonincreasing and continuous in t, and on
//            a fixed active set W the projection of the origin is affine in
//            t (Q_1 R^-T (b_W + t e_W)), so |proj| = rho is a quadratic in t
//            solved in closed form. Each probe projects, reads the active
//            set, solves for the level on that set and probes again; the
//            level is exact once a probe lands on the set that holds it, and
//            a probe that finds P_t empty returns a lower bound on t from
//            the certifying combination of rows.
//
//            Stage 2 minimises g . d over P at that level inside the ball
//            through the ball's multiplier nu: the minimiser of
//            g . d + nu/2 |d|^2 over P is the projection of -g / nu, whose
//            norm is nonincreasing in nu, and on a fixed active set it is
//            u_W - v_W / nu with u_W the least-norm point of the active
//            equalities and v_W the part of g outside their row space, which
//            are orthogonal, so |d| = rho is again solved in closed form. The
//            nu -> 0 limit is the least-norm minimiser of the linear
//            objective, which is Powell's tie-break, and it is detected as a
//            terminal active set (v_W = 0 and no multiplier decreasing).
//            Both searches keep a bracket and fall back to its midpoint when
//            a candidate leaves it, so they terminate on any input.
//
//            After the step the point is clamped into the box coordinate by
//            coordinate, so the box holds to the bit and not to rounding.
//
// No infinity, no NaN: "no bound" is an empty optional, a zero gradient is a
// zero-length step, "no bracket end yet" is a bool, and every division has
// its denominator tested first.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "flop/bounds.hpp"
#include "flop/detail/active_set.hpp"

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
    std::vector<double> A;   // rows of the polyhedron, row-major, n per row
    std::vector<double> b;   // right-hand sides at the current level: A_r . d <= b_r
    std::vector<double> b0;  // right-hand sides at level zero
    std::vector<double>
        relaxed;  // 1 for a constraint row (its b grows with the level), 0 for a box row
    std::vector<double> rownorm2;  // |A_r|^2
    ActiveSetProjection projection;
    std::vector<double> q;      // the query point of a projection
    std::vector<double> trial;  // the projection's answer
    std::vector<double> u;      // least-norm point of the active equalities
    std::vector<double> v;      // the parametric direction on the active set
    std::vector<double> slope;  // multiplier slopes on the active set
};

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
            const std::optional<double>& lo = bounds->lower[i];
            const std::optional<double>& hi = bounds->upper[i];
            if (lo.has_value() && xb[i] + d[i] < *lo) {
                d[i] = *lo - xb[i];
                fixed[i] = 1;
                clamped = true;
            } else if (hi.has_value() && xb[i] + d[i] > *hi) {
                d[i] = *hi - xb[i];
                fixed[i] = 1;
                clamped = true;
            }
        }
        if (!clamped) return;
    }
}

// Fills the workspace's rows at violation level s: for each constraint,
// -(cval_k + G_k . d) <= s, and for each bound the distance from the base.
// Constraint rows come first and are the relaxed ones; b0 keeps every row's
// right-hand side at level zero so that set_level can move the level.
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
    ws.b0.assign(rows, 0.0);
    ws.relaxed.assign(rows, 0.0);
    ws.rownorm2.assign(rows, 0.0);
    for (std::size_t k = 0; k < m; ++k) {
        double nn = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            ws.A[k * n + i] = -G[k * n + i];
            nn += G[k * n + i] * G[k * n + i];
        }
        ws.b0[k] = cval[k];
        ws.b[k] = cval[k] + s;
        ws.relaxed[k] = 1.0;
        ws.rownorm2[k] = nn;
    }
    std::size_t r = m;
    if (bounds) {
        for (std::size_t i = 0; i < n; ++i) {
            const std::optional<double>& lo = bounds->lower[i];
            const std::optional<double>& hi = bounds->upper[i];
            if (lo.has_value()) {
                ws.A[r * n + i] = -1.0;
                ws.b0[r] = xb[i] - *lo;
                ws.b[r] = ws.b0[r];
                ws.rownorm2[r] = 1.0;
                ++r;
            }
            if (hi.has_value()) {
                ws.A[r * n + i] = 1.0;
                ws.b0[r] = *hi - xb[i];
                ws.b[r] = ws.b0[r];
                ws.rownorm2[r] = 1.0;
                ++r;
            }
        }
    }
    return rows;
}

inline void set_level(double t, StepWorkspace& ws) {
    for (std::size_t r = 0; r < ws.b.size(); ++r) ws.b[r] = ws.b0[r] + t * ws.relaxed[r];
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

// |d| within this relative distance of rho counts as on the ball, and a
// bracket narrower than this relative width counts as closed.
constexpr double kBallTol = 1e-14;
// Probe cap for each parametric search: every probe either lands the exact
// answer on its active set or halves the bracket, so the cap is an internal
// error, never a stopping rule.
constexpr int kProbeCap = 256;

[[noreturn]] inline void step_did_not_converge() {
    throw std::runtime_error(
        "flop: COBYLA trust-region subproblem did not converge (internal error)");
}

// Stage 1: the least level t in [t_lo, t_hi] at which the relaxed polyhedron
// reaches the ball. On return ws.b holds that level and d its projection of
// the origin. Requires t_lo <= t_hi, the origin feasible at t_hi.
inline double violation_level(std::size_t n, double rho, double t_lo, double t_hi,
                              std::span<double> d, StepWorkspace& ws) {
    std::vector<double>& u = ws.u;
    std::vector<double>& w = ws.v;
    std::vector<double>& best = ws.trial;
    u.assign(n, 0.0);
    w.assign(n, 0.0);
    best.assign(n, 0.0);  // the projection at t_hi, the origin, until a better one is seen
    ws.q.assign(n, 0.0);
    ActiveSetProjection& proj = ws.projection;

    double t = t_lo;
    bool have_candidate = false;
    double last_candidate = 0.0;
    for (int probe = 0; probe < kProbeCap; ++probe) {
        set_level(t, ws);
        const bool feasible = proj.project(ws.q, ws.A, ws.b, ws.rownorm2, d);
        bool have_next = false;
        double next = 0.0;
        if (!feasible) {
            // The rows of the certificate, r <= 0 with n_p = A_W^T r, are
            // consistent with p once r . b_W(t) <= b_p(t); solving that for t
            // gives the level at which this contradiction disappears.
            t_lo = t;
            const std::size_t p = proj.blocking_row();
            const std::span<const double> r = proj.blocking_combination();
            const std::span<const std::size_t> active = proj.active();
            double coefficient = -ws.relaxed[p];
            double constant = ws.b0[p];
            for (std::size_t j = 0; j < active.size(); ++j) {
                coefficient += r[j] * ws.relaxed[active[j]];
                constant -= r[j] * ws.b0[active[j]];
            }
            if (coefficient < 0.0) {
                const double t_feasible = constant / coefficient;
                if (t_feasible > t_lo && t_feasible <= t_hi) {
                    have_next = true;
                    next = t_feasible;
                }
            }
        } else {
            const double dn = norm(d);
            if (std::fabs(dn - rho) <= kBallTol * rho) return t;
            if (dn < rho) {
                if (t == t_lo) return t;  // the least admissible level already reaches the ball
                t_hi = t;
                std::ranges::copy(d, best.begin());
            } else {
                t_lo = t;
            }
            // On this active set the projection is u + t w; the level with
            // |u + t w| = rho on the decreasing branch is the smaller root.
            proj.least_norm_point(ws.b0, u);
            proj.least_norm_point(ws.relaxed, w);
            const double a = dot(w, w);
            const double bq = 2.0 * dot(u, w);
            const double c = dot(u, u) - rho * rho;
            if (a > 0.0) {
                const double disc = bq * bq - 4.0 * a * c;
                if (disc >= 0.0) {
                    const double root = (-bq - std::sqrt(disc)) / (2.0 * a);
                    if (root > t_lo && root < t_hi) {
                        have_next = true;
                        next = root;
                    }
                }
            }
        }
        if (t_hi - t_lo <= kBallTol * (1.0 + t_hi)) {
            std::ranges::copy(best, d.begin());
            set_level(t_hi, ws);
            return t_hi;
        }
        if (have_next && have_candidate && next == last_candidate) have_next = false;
        if (!have_next) next = 0.5 * (t_lo + t_hi);
        have_candidate = true;
        last_candidate = next;
        t = next;
    }
    step_did_not_converge();
}

// Stage 2: minimise g . d over the polyhedron at ws.b inside the ball, least
// norm among ties. d1 is the least-norm point of the polyhedron (stage 1's
// answer), which is the step whenever it already fills the ball.
inline void objective_step(std::size_t n, std::span<const double> g, double rho,
                           std::span<const double> d1, std::span<double> d, StepWorkspace& ws) {
    const double d1n = norm(d1);
    if (d1n >= rho * (1.0 - kBallTol)) {
        const double scale = (d1n > rho) ? rho / d1n : 1.0;
        for (std::size_t i = 0; i < n; ++i) d[i] = scale * d1[i];
        return;
    }
    const double gnorm = norm(g);
    if (gnorm == 0.0) {
        std::ranges::copy(d1, d.begin());
        return;
    }
    std::vector<double>& u = ws.u;
    std::vector<double>& v = ws.v;
    std::vector<double>& best = ws.trial;
    std::vector<double>& slope = ws.slope;
    u.assign(n, 0.0);
    v.assign(n, 0.0);
    best.assign(d1.begin(), d1.end());
    ws.q.assign(n, 0.0);
    ActiveSetProjection& proj = ws.projection;

    // The parameter is s = 1 / nu: |d(s)| is nondecreasing, d(0) = d1.
    double s_lo = 0.0;
    bool have_hi = false;
    double s_hi = 0.0;
    double s = rho / gnorm;
    bool have_candidate = false;
    double last_candidate = 0.0;
    for (int probe = 0; probe < kProbeCap; ++probe) {
        for (std::size_t i = 0; i < n; ++i) ws.q[i] = -s * g[i];
        if (!proj.project(ws.q, ws.A, ws.b, ws.rownorm2, d)) step_did_not_converge();
        const double dn = norm(d);
        if (std::fabs(dn - rho) <= kBallTol * rho) return;

        proj.least_norm_point(ws.b, u);
        proj.complement(g, v);
        const double un2 = dot(u, u);
        const double vn2 = dot(v, v);
        bool have_next = false;
        double next = 0.0;
        if (dn > rho) {
            have_hi = true;
            s_hi = s;
            if (un2 < rho * rho && vn2 > 0.0) {
                const double root = std::sqrt((rho * rho - un2) / vn2);
                if (root > s_lo && root < s_hi) {
                    have_next = true;
                    next = root;
                }
            }
        } else {
            s_lo = s;
            std::ranges::copy(d, best.begin());
            if (vn2 <= (ActiveSetProjection::kDependentTol * gnorm) *
                           (ActiveSetProjection::kDependentTol * gnorm)) {
                // g lies in the active row space: d is constant on this
                // piece. The piece ends where a multiplier reaches zero;
                // if none decreases, d is the least-norm minimiser.
                const std::span<const double> lambda = proj.multipliers();
                slope.assign(lambda.size(), 0.0);
                proj.multiplier_slope(g, slope);  // lambda(s) = lambda - (s - s_now) slope
                bool terminal = true;
                double s_break = 0.0;
                bool have_break = false;
                for (std::size_t j = 0; j < lambda.size(); ++j) {
                    if (slope[j] <= 0.0) continue;
                    terminal = false;
                    const double sb = s + lambda[j] / slope[j];
                    if (!have_break || sb < s_break) {
                        have_break = true;
                        s_break = sb;
                    }
                }
                if (terminal) return;
                have_next = true;
                next = have_hi ? 0.5 * (s_break + s_hi) : 2.0 * s_break;
                if (next <= s_lo) have_next = false;
            } else if (un2 < rho * rho) {
                const double root = std::sqrt((rho * rho - un2) / vn2);
                if (root > s_lo && (!have_hi || root < s_hi)) {
                    have_next = true;
                    next = root;
                }
            }
        }
        if (have_hi && s_hi - s_lo <= kBallTol * s_hi) {
            std::ranges::copy(best, d.begin());
            return;
        }
        if (have_next && have_candidate && next == last_candidate) have_next = false;
        if (!have_next) next = have_hi ? 0.5 * (s_lo + s_hi) : 2.0 * s;
        have_candidate = true;
        last_candidate = next;
        s = next;
    }
    step_did_not_converge();
}

// The constrained step. Writes d; returns the violation level the step was
// computed for (Powell's resmax of the step).
inline double constrained_step(std::size_t n, std::size_t m, std::span<const double> xb,
                               std::span<const double> cval, std::span<const double> G,
                               std::span<const double> g, double rho, const Bounds* bounds,
                               std::span<double> d, StepWorkspace& ws) {
    build_polyhedron(n, m, xb, cval, G, bounds, 0.0, ws);
    ws.projection.reset(n);

    // The level's range: s_hi is the violation at d = 0, feasible by
    // construction; s_lo bounds every point of the ball from below,
    // constraint by constraint.
    double s_hi = 0.0, s_lo = 0.0;
    for (std::size_t k = 0; k < m; ++k) {
        double gn = 0.0;
        for (std::size_t i = 0; i < n; ++i) gn += G[k * n + i] * G[k * n + i];
        gn = std::sqrt(gn);
        s_hi = std::max(s_hi, -cval[k]);
        s_lo = std::max(s_lo, -cval[k] - rho * gn);
    }
    std::vector<double> d1(n, 0.0);
    double level = 0.0;
    if (s_hi > 0.0) {
        level = violation_level(n, rho, std::max(0.0, s_lo), s_hi, d1, ws);
    } else {
        set_level(0.0, ws);
    }
    objective_step(n, g, rho, d1, d, ws);
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
    if (bounds)
        for (std::size_t i = 0; i < n; ++i) {
            const std::optional<double>& lo = bounds->lower[i];
            const std::optional<double>& hi = bounds->upper[i];
            if (lo.has_value() && xb[i] + d[i] < *lo) d[i] = *lo - xb[i];
            if (hi.has_value() && xb[i] + d[i] > *hi) d[i] = *hi - xb[i];
        }
    return predicted_violation(n, m, cval, G, d);
}

}  // namespace flop::detail::cobyla
