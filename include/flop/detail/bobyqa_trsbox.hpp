// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// bobyqa_trsbox - the trust-region step of BOBYQA (report section 3, TRSBOX)
// =============================================================================
//
// Powell, DAMTP 2009/NA06, section 3, first half: an approximate solution
// of (1.8), minimise Q(x_k + d) subject to a <= x_k + d <= b and
// |d| <= Delta, by an active-set truncated conjugate gradient method from
// d = 0, followed, once d reaches the trust-region boundary, by moves round
// the boundary in the plane of P_I(d) and P_I(grad Q(x_k + d)).
//
//   active set   I starts as (3.3): the coordinates at a bound whose
//                gradient points out of the box. An index joins I when a
//                step stops on its bound and never leaves.
//   CG           step length min(alpha_B, alpha_Delta, alpha_Q). A bound
//                restarts the method at -P_I(grad); a full step continues
//                it with the direction conjugate to the last gradient
//                change. Stops by (3.4), with 0.01 the report's factor, or
//                when a full step reduces Q by no more than that bound.
//   boundary     stops by (3.5), with the report's 1e-4; else
//                d(theta) = d - P_I(d) + cos(theta) P_I(d) + sin(theta) s
//                (3.6) for theta in [0, pi/4], up to the first bound or the
//                first minimum of Q along the arc.
//
// The report finds theta_Q and theta_B "approximately". Here theta_B is
// exact to rounding: each coordinate is a sinusoid in theta, and the angle
// at which it reaches a bound is in closed form. theta_Q is the first zero
// of dQ/dtheta, located by sampling the derivative at kArcSamples equal
// steps of [0, pi/4] and refined by Illinois regula falsi.
//
// The CG phase is also limited to as many steps per restart as there are
// free coordinates, the number after which conjugate gradients on a
// quadratic have nothing left to do in exact arithmetic.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <vector>

#include "flop/detail/bobyqa_model.hpp"

namespace flop::detail::bobyqa {

// Section 3: the factor in (3.4) and the factor in (3.5).
constexpr double kCgStop = 0.01;
constexpr double kArcStop = 1e-4;
// The samples of dQ/dtheta along the arc (see the header).
constexpr std::size_t kArcSamples = 16;

// The workspace, in two allocations: six vectors of n doubles cut from one
// buffer, and the free flags. Not copyable, since the spans point into the
// buffer.
struct TrsboxWork {
    explicit TrsboxWork(std::size_t n)
        : buf(6 * n, 0.0),
          s(buf.data(), n),
          hs(buf.data() + n, n),
          pd(buf.data() + 2 * n, n),
          pg(buf.data() + 3 * n, n),
          hpd(buf.data() + 4 * n, n),
          x(buf.data() + 5 * n, n),
          free(n, 0) {}
    TrsboxWork(const TrsboxWork&) = delete;
    TrsboxWork& operator=(const TrsboxWork&) = delete;
    TrsboxWork(TrsboxWork&&) = delete;
    TrsboxWork& operator=(TrsboxWork&&) = delete;
    ~TrsboxWork() = default;

    std::vector<double> buf;
    std::span<double> s, hs, pd, pg, hpd, x;
    std::vector<char> free;
};

struct TrsboxResult {
    double qred = 0.0;  // Q(x_k) - Q(x_k + d)
    // The least s^T (nabla^2 Q) s / |s|^2 over the full CG steps since the
    // last restart, the set S of (6.10); have_curvature false when S is
    // empty.
    double curvature = 0.0;
    bool have_curvature = false;
};

namespace trsbox_detail {

// The first theta in (0, top] at which g changes from positive to
// nonpositive, given g(0) > 0 and g(top) <= 0: regula falsi with the
// Illinois modification (the retained end's value is halved whenever the
// same end is kept twice), which converges superlinearly. It stops when the
// bracket is within rounding of top or the new point does not land inside
// it, and returns the nonpositive end.
template <class G>
double first_root(G g, double top) {
    double a = 0.0, ga = g(0.0), b = top, gb = g(top);
    int kept = 0;
    const double tiny = std::numeric_limits<double>::epsilon() * top;
    while (b - a > tiny) {
        const double denom = ga - gb;
        if (!(denom > 0.0)) break;
        const double c = b - gb * (b - a) / denom;
        if (!(c > a && c < b)) break;
        const double gc = g(c);
        if (gc > 0.0) {
            a = c;
            ga = gc;
            if (kept == -1) gb *= 0.5;
            kept = -1;
        } else {
            b = c;
            gb = gc;
            if (kept == 1) ga *= 0.5;
            kept = 1;
        }
    }
    return b;
}

}  // namespace trsbox_detail

// d (out) is the step from x_k = model.xopt(), side (out) says which
// coordinates of x_k + d are on a bound, gd (out) is grad Q(x_k + d).
inline TrsboxResult trsbox(const Model& model, const Box& box, double delta, std::span<double> d,
                           std::span<Side> side, std::span<double> gd, TrsboxWork& w) {
    const std::size_t n = model.n;
    std::span<const double> xk = model.xopt();
    TrsboxResult out;
    std::size_t nfree = 0;
    for (std::size_t i = 0; i < n; ++i) {
        d[i] = 0.0;
        gd[i] = model.g[i];
        side[i] = Side::Free;
        if (box.at_lo(i, xk[i]) && model.g[i] >= 0.0) side[i] = Side::Lower;
        if (box.at_hi(i, xk[i]) && model.g[i] <= 0.0) side[i] = Side::Upper;
        w.free[i] = side[i] == Side::Free ? 1 : 0;
        if (w.free[i]) ++nfree;
    }
    const auto proj_norm2 = [&](std::span<const double> v) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            if (w.free[i]) s += v[i] * v[i];
        return s;
    };
    const double delta2 = delta * delta;
    double& qred = out.qred;

    // ========================================================== CG
    bool to_boundary = false;
    bool restart = true;
    std::size_t steps = 0;
    for (;;) {
        if (restart) {
            if (nfree == 0) return out;
            for (std::size_t i = 0; i < n; ++i) w.s[i] = w.free[i] ? -gd[i] : 0.0;
            out.have_curvature = false;
            steps = 0;
            restart = false;
        }
        const double ss = dot(w.s, w.s);
        if (!(ss > 0.0)) return out;
        const double gs = dot(gd, w.s);
        if (!(gs < 0.0)) return out;
        model.hess_mul(w.s, w.hs);
        const double shs = dot(w.s, w.hs);

        // alpha_Delta: |d + alpha s| = Delta, the positive root.
        const double ds = dot(d, w.s);
        const double dd = dot(d, d);
        const double slack = delta2 - dd;
        if (!(slack > 0.0)) {
            to_boundary = true;
            break;
        }
        const double alpha_delta = slack / (ds + std::sqrt(ds * ds + ss * slack));
        double alpha = alpha_delta;
        enum class Stop : std::uint8_t { Delta, Bound, Curvature } why = Stop::Delta;
        if (shs > 0.0) {
            const double aq = -gs / shs;
            if (aq < alpha) {
                alpha = aq;
                why = Stop::Curvature;
            }
        }
        std::size_t ib = n;
        for (std::size_t i = 0; i < n; ++i) {
            if (!w.free[i] || w.s[i] == 0.0) continue;
            double room;
            if (w.s[i] > 0.0) {
                if (!box.has_hi[i]) continue;
                room = box.hi[i] - xk[i] - d[i];
            } else {
                if (!box.has_lo[i]) continue;
                room = box.lo[i] - xk[i] - d[i];
            }
            double ab = room / w.s[i];
            if (ab < 0.0) ab = 0.0;
            if (ab < alpha || (ab == alpha && why == Stop::Delta && ab < alpha_delta)) {
                alpha = ab;
                ib = i;
                why = Stop::Bound;
            }
        }
        if (why == Stop::Bound && !(alpha < alpha_delta)) why = Stop::Delta;

        const double red = -alpha * gs - 0.5 * alpha * alpha * shs;
        qred += red;
        for (std::size_t i = 0; i < n; ++i) {
            d[i] += alpha * w.s[i];
            gd[i] += alpha * w.hs[i];
        }
        if (why == Stop::Delta) {
            to_boundary = true;
            break;
        }
        if (why == Stop::Bound) {
            const bool upper = w.s[ib] > 0.0;
            side[ib] = upper ? Side::Upper : Side::Lower;
            d[ib] = (upper ? box.hi[ib] : box.lo[ib]) - xk[ib];
            w.free[ib] = 0;
            --nfree;
            if (std::sqrt(proj_norm2(gd)) * delta <= kCgStop * qred) return out;
            restart = true;
            continue;
        }
        // A full projected CG step.
        const double crv = shs / ss;
        if (!out.have_curvature || crv < out.curvature) out.curvature = crv;
        out.have_curvature = true;
        const double pgn2 = proj_norm2(gd);
        if (red <= kCgStop * qred || std::sqrt(pgn2) * delta <= kCgStop * qred) return out;
        if (++steps >= nfree) return out;
        // -P_I(grad) plus the multiple of s orthogonal to the gradient change
        // alpha H s: beta = P_I(grad).Hs / s.Hs, shs > 0 on this branch.
        double pgh = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            if (w.free[i]) pgh += gd[i] * w.hs[i];
        const double beta = pgh / shs;
        for (std::size_t i = 0; i < n; ++i) w.s[i] = w.free[i] ? -gd[i] + beta * w.s[i] : 0.0;
    }
    if (!to_boundary) return out;

    // ========================================================== round the boundary
    constexpr double kQuarter = std::numbers::pi / 4.0;
    for (;;) {
        if (nfree == 0) return out;
        double pdd = 0.0, pgg = 0.0, pdg = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            w.pd[i] = w.free[i] ? d[i] : 0.0;
            w.pg[i] = w.free[i] ? gd[i] : 0.0;
            pdd += w.pd[i] * w.pd[i];
            pgg += w.pg[i] * w.pg[i];
            pdg += w.pd[i] * w.pg[i];
        }
        const double temp = pdd * pgg - pdg * pdg;
        if (temp <= kArcStop * qred * qred || !(temp > 0.0)) return out;
        const double root = std::sqrt(temp);
        for (std::size_t i = 0; i < n; ++i) w.s[i] = (pdg * w.pd[i] - pdd * w.pg[i]) / root;
        model.hess_mul(w.pd, w.hpd);
        model.hess_mul(w.s, w.hs);
        const double a1 = pdg, b1 = dot(gd, w.s);
        const double qa = dot(w.pd, w.hpd), qb = dot(w.pd, w.hs), qc = dot(w.s, w.hs);
        // f(theta) = Q(x_k + d(theta)) - Q(x_k + d) and its derivative.
        const auto f = [&](double th) {
            const double c = std::cos(th) - 1.0, sn = std::sin(th);
            return c * a1 + sn * b1 + 0.5 * c * c * qa + c * sn * qb + 0.5 * sn * sn * qc;
        };
        const auto fp = [&](double th) {
            const double c = std::cos(th), sn = std::sin(th);
            return -sn * a1 + c * b1 - (c - 1.0) * sn * qa + (c * c - c - sn * sn) * qb +
                   sn * c * qc;
        };
        if (!(b1 < 0.0)) return out;
        // theta_Q: the first sampled point where dQ/dtheta >= 0, refined.
        double theta_q = kQuarter;
        for (std::size_t k = 1; k <= kArcSamples; ++k) {
            const double th = kQuarter * static_cast<double>(k) / static_cast<double>(kArcSamples);
            if (fp(th) >= 0.0) {
                theta_q = trsbox_detail::first_root([&](double t) { return -fp(t); }, th);
                break;
            }
        }
        // theta_B: per free coordinate, x_k[i] + d[i] cos + s[i] sin, that
        // is x_k[i] + R cos(theta - phi) with R = |(d[i], s[i])| and phi =
        // atan2(s[i], d[i]), must stay in [lo, hi]. It reaches a level q
        // going down at theta = phi + acos(q / R) and going up at phi -
        // acos(q / R), modulo 2 pi; the first exit over all coordinates in
        // (0, theta] is theta_B.
        double theta = theta_q;
        std::size_t ib = n;
        bool ib_upper = false;
        constexpr double kTwoPi = 2.0 * std::numbers::pi;
        const auto wrap = [&](double a) {
            while (a < 0.0) a += kTwoPi;
            while (a >= kTwoPi) a -= kTwoPi;
            return a;
        };
        for (std::size_t i = 0; i < n; ++i) {
            if (!w.free[i]) continue;
            const double di = d[i], si = w.s[i], xi = xk[i];
            const double r = std::hypot(di, si);
            if (!(r > 0.0)) continue;
            const double phi = std::atan2(si, di);
            const double here = xi + di;
            if (box.has_lo[i]) {
                const double q = (box.lo[i] - xi) / r;
                double th = -1.0;
                if (here <= box.lo[i] && si < 0.0)
                    th = 0.0;
                else if (q > -1.0 && q < 1.0)
                    th = wrap(phi + std::acos(q));
                if (th >= 0.0 && th < theta) {
                    theta = th;
                    ib = i;
                    ib_upper = false;
                }
            }
            if (box.has_hi[i]) {
                const double q = (box.hi[i] - xi) / r;
                double th = -1.0;
                if (here >= box.hi[i] && si > 0.0)
                    th = 0.0;
                else if (q > -1.0 && q < 1.0)
                    th = wrap(phi - std::acos(q));
                if (th >= 0.0 && th < theta) {
                    theta = th;
                    ib = i;
                    ib_upper = true;
                }
            }
        }
        if (!(theta > 0.0)) {
            if (ib < n) {
                side[ib] = ib_upper ? Side::Upper : Side::Lower;
                d[ib] = (ib_upper ? box.hi[ib] : box.lo[ib]) - xk[ib];
                w.free[ib] = 0;
                --nfree;
                continue;
            }
            return out;
        }
        const double change = f(theta);
        if (!(change < 0.0)) return out;
        const double c = std::cos(theta), sn = std::sin(theta);
        for (std::size_t i = 0; i < n; ++i) {
            if (!w.free[i]) continue;
            d[i] = d[i] * c + w.s[i] * sn;
        }
        for (std::size_t i = 0; i < n; ++i) gd[i] += (c - 1.0) * w.hpd[i] + sn * w.hs[i];
        const double red = -change;
        qred += red;
        if (ib < n) {
            side[ib] = ib_upper ? Side::Upper : Side::Lower;
            d[ib] = (ib_upper ? box.hi[ib] : box.lo[ib]) - xk[ib];
            w.free[ib] = 0;
            --nfree;
            continue;
        }
        if (red <= kCgStop * qred) return out;
    }
}

}  // namespace flop::detail::bobyqa
