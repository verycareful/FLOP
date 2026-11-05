// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// bobyqa_model - BOBYQA's quadratic model and the inverse KKT matrix H
// =============================================================================
//
// M. J. D. Powell, "The BOBYQA algorithm for bound constrained optimization
// without derivatives", report DAMTP 2009/NA06, University of Cambridge,
// 2009. Equation numbers below are that report's. The updating formulas are
// Powell's from "Least Frobenius norm updating of quadratic models that
// satisfy interpolation conditions", Mathematical Programming 100, 2004,
// pp. 183-215, as restated in sections 2 and 4 of the report.
//
// Every point is held as an offset from the origin x0 (the report's x0,
// shifted now and then by section 6), so a point y_j below always means
// y_j - x0. With m interpolation points in R^n:
//
//   Model     the points, their values, the index kopt of the least value
//             (1.3), the gradient g at x_kopt (4.7), and the second
//             derivative matrix in the form (4.6): an explicit symmetric
//             n x n matrix M plus sum_l mu_l y_l y_l^T, so that a product
//             with it costs O(mn) and an update touches only mu.
//
//   Inverse   H = W^-1 of (2.7) without its (m+1)-th row and column, as
//             section 4 keeps it: the m x (m-n-1) matrix Z with
//             Omega = Z Z^T, and the (m+n) x n matrix B whose first m rows
//             are Xi^T without its first column and whose last n rows are
//             Upsilon without its first row and column.
//
// Column t of H gives the Lagrange function Lambda_t of (3.8): Omega e_t
// holds its parameters lambda_l in the second derivative sum_l lambda_l
// y_l y_l^T (4.16), and row t of B its gradient at x0 (4.15).

#pragma once

#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

namespace flop::detail::bobyqa {

inline double dot(std::span<const double> a, std::span<const double> b) noexcept {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// The pair {p(j), q(j)} of (2.3)-(2.4), 0-based, for the extra point with
// 0-based index e = j - (2n + 2) beyond the first 2n + 1. p cycles through
// the coordinates; in the l-th cycle (l = e / n + 1) q is p + l, wrapped
// into [0, n). The report's example, n = 5 and m = 20, gives {1,2}, {2,3},
// {3,4}, {4,5}, {5,1}, {1,3}, {2,4}, {3,5}, {4,1} in its 1-based numbering.
struct Pair {
    std::size_t p;
    std::size_t q;
};

constexpr Pair extra_pair(std::size_t n, std::size_t e) noexcept {
    const std::size_t p = e % n;
    const std::size_t cycle = e / n + 1;
    return {.p = p, .q = (p + cycle) % n};
}

// ============================================================================
// Inverse: Z and B, the parts of H that BOBYQA keeps
// ============================================================================
class Inverse {
public:
    Inverse(std::size_t n, std::size_t m)
        : n_(n),
          m_(m),
          nz_(m - n - 1),
          z_(m * (m - n - 1), 0.0),
          b_((m + n) * n, 0.0),
          rot_c_(m - n - 1, 1.0),
          rot_s_(m - n - 1, 0.0) {}

    [[nodiscard]] std::size_t columns() const noexcept { return nz_; }
    double& z(std::size_t i, std::size_t c) noexcept { return z_[i * nz_ + c]; }
    [[nodiscard]] double z(std::size_t i, std::size_t c) const noexcept { return z_[i * nz_ + c]; }
    double& b(std::size_t r, std::size_t k) noexcept { return b_[r * n_ + k]; }
    [[nodiscard]] double b(std::size_t r, std::size_t k) const noexcept { return b_[r * n_ + k]; }
    [[nodiscard]] std::span<const double> b_row(std::size_t r) const noexcept {
        return {b_.data() + r * n_, n_};
    }

    // H_tt = e_t^T Z Z^T e_t = |Z^T e_t|^2, nonnegative by construction (5.1).
    [[nodiscard]] double omega_diag(std::size_t t) const noexcept {
        double s = 0.0;
        for (std::size_t c = 0; c < nz_; ++c) s += z(t, c) * z(t, c);
        return s;
    }

    // out = Omega e_t, the parameters lambda of Lambda_t (4.16).
    void omega_col(std::size_t t, std::span<double> out) const noexcept {
        for (std::size_t i = 0; i < m_; ++i) {
            double s = 0.0;
            for (std::size_t c = 0; c < nz_; ++c) s += z(i, c) * z(t, c);
            out[i] = s;
        }
    }

    // Section 2, with the generality of (2.10): H and Z for the points
    // y_1 = 0, y_{i+1} = alpha_i e_i, y_{n+i+1} = beta_i e_i (when
    // n + i + 1 <= m) and, for m > 2n + 1, y_j = alpha_p e_p + alpha_q e_q
    // with {p, q} from extra_pair (2.3). The alpha_i are nonzero, and so are
    // the beta_i that are used, with alpha_i != beta_i. The elements are
    // (2.11)-(2.15); a 1-D quadratic through 0, alpha and beta gives the
    // Lagrange gradients at 0 directly: -(alpha + beta) / (alpha beta),
    // beta / (alpha (beta - alpha)) and alpha / (beta (alpha - beta)).
    void set_coordinate(std::span<const double> alpha, std::span<const double> beta) noexcept {
        for (double& v : z_) v = 0.0;
        for (double& v : b_) v = 0.0;
        const std::size_t both = (m_ - n_ - 1 < n_) ? m_ - n_ - 1 : n_;
        for (std::size_t i = 0; i < n_; ++i) {
            const double a = alpha[i];
            if (i < both) {
                const double c = beta[i];
                b(0, i) = -1.0 / a - 1.0 / c;
                b(i + 1, i) = c / (a * (c - a));
                b(n_ + i + 1, i) = a / (c * (a - c));
                // (2.14): Z's column i, three entries summing to zero with
                // zero first moment, scaled so that Z Z^T is Omega.
                z(0, i) = -std::numbers::sqrt2 / (a * c);
                z(i + 1, i) = std::numbers::sqrt2 / (a * (c - a));
                z(n_ + i + 1, i) = std::numbers::sqrt2 / (c * (a - c));
            } else {
                // (2.12) and (2.13): one point along e_i, no curvature there.
                b(0, i) = -1.0 / a;
                b(i + 1, i) = 1.0 / a;
                b(m_ + i, i) = -0.5 * a * a;
            }
        }
        // (2.15): one column per extra point.
        for (std::size_t c = n_; c < nz_; ++c) {
            const std::size_t j = n_ + c + 1;  // 0-based point index, 2n + 1 + (c - n)
            const Pair pq = extra_pair(n_, c - n_);
            const double r = 1.0 / (alpha[pq.p] * alpha[pq.q]);
            z(0, c) = r;
            z(j, c) = r;
            z(pq.p + 1, c) = -r;
            z(pq.q + 1, c) = -r;
        }
    }

    // Section 4: for replacing a point by x (an offset), with x_s the point
    // the update is centred on (s != the replaced index), the vector
    // vlag = H w = H (w - v) + e_s of (4.13), first m and last n
    // components, and the return value beta = (1/2)|x|^4 - w^T H w (4.11).
    // (w - v)_i = (1/2)((y_i.x)^2 - (y_i.x_s)^2) is formed as
    // (1/2)(y_i.d)(y_i.(x + x_s)) with d = x - x_s, which loses nothing to
    // cancellation when x is close to x_s.
    [[nodiscard]] double vlag_beta(std::span<const double> points, std::size_t s,
                                   std::span<const double> x, std::span<double> vlag,
                                   std::span<double> wv, std::span<double> zw, std::span<double> d,
                                   std::span<double> sum) const noexcept {
        std::span<const double> xs(points.data() + s * n_, n_);
        for (std::size_t k = 0; k < n_; ++k) {
            d[k] = x[k] - xs[k];
            sum[k] = x[k] + xs[k];
        }
        for (std::size_t i = 0; i < m_; ++i) {
            std::span<const double> yi(points.data() + i * n_, n_);
            wv[i] = 0.5 * dot(yi, d) * dot(yi, sum);
        }
        // Z^T wv and B^T wv by rows, so that every pass runs along memory.
        for (std::size_t c = 0; c < nz_; ++c) zw[c] = 0.0;
        for (std::size_t k = 0; k < n_; ++k) vlag[m_ + k] = 0.0;
        for (std::size_t i = 0; i < m_; ++i) {
            const double wi = wv[i];
            const double* zi = z_.data() + i * nz_;
            for (std::size_t c = 0; c < nz_; ++c) zw[c] += zi[c] * wi;
            const double* bi = b_.data() + i * n_;
            for (std::size_t k = 0; k < n_; ++k) vlag[m_ + k] += bi[k] * wi;
        }
        for (std::size_t i = 0; i < m_; ++i) {
            const double* zi = z_.data() + i * nz_;
            double t = 0.0;
            for (std::size_t c = 0; c < nz_; ++c) t += zi[c] * zw[c];
            vlag[i] = t + dot(b_row(i), d);
        }
        for (std::size_t l = 0; l < n_; ++l) {
            const double dl = d[l];
            const double* bl = b_.data() + (m_ + l) * n_;
            for (std::size_t k = 0; k < n_; ++k) vlag[m_ + k] += bl[k] * dl;
        }
        // (w - v)^T H (w - v), then (4.13): w^T H w = that + 2 w_s - v_s
        // with w_s = (1/2)(x_s.x)^2 and v_s = (1/2)|x_s|^4.
        double quad = 0.0;
        for (std::size_t i = 0; i < m_; ++i) quad += wv[i] * vlag[i];
        for (std::size_t k = 0; k < n_; ++k) quad += d[k] * vlag[m_ + k];
        vlag[s] += 1.0;
        const double xx = dot(x, x), sx = dot(xs, x), ss = dot(xs, xs);
        // (1/2)|x|^4 - (x_s.x)^2 + (1/2)|x_s|^4 - quad.
        return 0.5 * xx * xx - sx * sx + 0.5 * ss * ss - quad;
    }

    // Section 4: H becomes the inverse for the point set with point t
    // replaced, given vlag and beta from vlag_beta for the new point.
    // Requires sigma = alpha beta + tau^2 > 0, which the caller has tested
    // against (5.2). Formula (4.9) with e_t - H w replaced by
    // u = e_t - vlag, on B directly; Z by (4.14) after a rotation that
    // leaves row t with only its first entry. h, u are workspaces of
    // length m + n.
    void update(std::size_t t, std::span<const double> vlag, double beta, std::span<double> h,
                std::span<double> u) noexcept {
        // Givens rotations of Z's columns 0 and c, c = 1, 2, ..., in turn: Z
        // Z^T is unchanged. Each rotation is fixed by row t alone, so row t
        // is rotated first and the angles kept; the other rows then take all
        // the rotations in the same order, one row at a time along memory.
        double* zt_row = z_.data() + t * nz_;
        for (std::size_t c = 1; c < nz_; ++c) {
            const double zc = zt_row[c];
            if (zc == 0.0) {
                rot_c_[c] = 1.0;
                rot_s_[c] = 0.0;
                continue;
            }
            const double r = std::hypot(zt_row[0], zc);
            rot_c_[c] = zt_row[0] / r;
            rot_s_[c] = zc / r;
            zt_row[0] = r;
            zt_row[c] = 0.0;
        }
        for (std::size_t i = 0; i < m_; ++i) {
            if (i == t) continue;
            double* zi = z_.data() + i * nz_;
            double a0 = zi[0];
            for (std::size_t c = 1; c < nz_; ++c) {
                const double cs = rot_c_[c], sn = rot_s_[c];
                const double ac = zi[c];
                zi[c] = cs * ac - sn * a0;
                a0 = cs * a0 + sn * ac;
            }
            zi[0] = a0;
        }
        const double zt = z(t, 0);
        const double alpha = zt * zt;
        const double tau = vlag[t];
        const double sigma = alpha * beta + tau * tau;
        // h = H e_t: Omega e_t = zt Z(:, 0) after the rotation, then row t of B.
        for (std::size_t i = 0; i < m_; ++i) h[i] = zt * z(i, 0);
        for (std::size_t k = 0; k < n_; ++k) h[m_ + k] = b(t, k);
        for (std::size_t i = 0; i < m_ + n_; ++i) u[i] = -vlag[i];
        u[t] += 1.0;
        const double inv = 1.0 / sigma;
        for (std::size_t r = 0; r < m_ + n_; ++r) {
            for (std::size_t k = 0; k < n_; ++k) {
                const double hc = h[m_ + k], uc = u[m_ + k];
                b(r, k) +=
                    inv * (alpha * u[r] * uc - beta * h[r] * hc + tau * (h[r] * uc + u[r] * hc));
            }
        }
        const double root = std::sqrt(sigma);
        for (std::size_t i = 0; i < m_; ++i) z(i, 0) = (tau * z(i, 0) + zt * u[i]) / root;
    }

    // Section 6, (6.14)-(6.16): the origin moves by s (the report's
    // x_k - x0), with points[] the offsets from the old origin. Omega is
    // unchanged; with Gamma's column j equal to (s.(y_j - x_av))(y_j - x_av)
    // + (1/4)|s|^2 s and x_av = s / 2, Xi_red gains Gamma Omega and
    // Upsilon_red gains Gamma Xi_red^T + Xi_red Gamma^T + Gamma Omega
    // Gamma^T. In B: the first m rows gain C = Omega Gamma^T and the last n
    // gain Gamma B_old + (Gamma B_old)^T + Gamma C.
    void shift(std::span<const double> points, std::span<const double> s, std::span<double> gamma,
               std::span<double> cmat, std::span<double> zg, std::span<double> diff) {
        const double ss4 = 0.25 * dot(s, s);
        for (std::size_t j = 0; j < m_; ++j) {
            std::span<const double> yj(points.data() + j * n_, n_);
            for (std::size_t k = 0; k < n_; ++k) diff[k] = yj[k] - 0.5 * s[k];
            const double proj = dot(s, diff);
            for (std::size_t k = 0; k < n_; ++k) gamma[k * m_ + j] = proj * diff[k] + ss4 * s[k];
        }
        // C = Z (Z^T Gamma^T), m x n.
        for (std::size_t k = 0; k < n_; ++k) {
            for (std::size_t c = 0; c < nz_; ++c) {
                double t = 0.0;
                for (std::size_t j = 0; j < m_; ++j) t += z(j, c) * gamma[k * m_ + j];
                zg[c] = t;
            }
            for (std::size_t i = 0; i < m_; ++i) {
                double t = 0.0;
                for (std::size_t c = 0; c < nz_; ++c) t += z(i, c) * zg[c];
                cmat[i * n_ + k] = t;
            }
        }
        // Upsilon block, from the old B and C.
        for (std::size_t p = 0; p < n_; ++p) {
            for (std::size_t q = 0; q <= p; ++q) {
                double t = 0.0;
                for (std::size_t j = 0; j < m_; ++j) {
                    const double gp = gamma[p * m_ + j], gq = gamma[q * m_ + j];
                    t += gp * b(j, q) + b(j, p) * gq + gp * cmat[j * n_ + q];
                }
                b(m_ + p, q) += t;
                if (q != p) b(m_ + q, p) += t;
            }
        }
        for (std::size_t i = 0; i < m_; ++i)
            for (std::size_t k = 0; k < n_; ++k) b(i, k) += cmat[i * n_ + k];
    }

private:
    std::size_t n_, m_, nz_;
    std::vector<double> z_, b_;
    std::vector<double> rot_c_, rot_s_;  // the rotations of update
};

// ============================================================================
// Model: the points, their values, and the quadratic model of (4.6)-(4.7)
// ============================================================================
struct Model {
    Model(std::size_t n_, std::size_t m_)
        : n(n_),
          m(m_),
          points(m_ * n_, 0.0),
          fval(m_, 0.0),
          g(n_, 0.0),
          hq(n_ * n_, 0.0),
          mu(m_, 0.0) {}

    std::size_t n, m;
    std::vector<double> points;  // m x n offsets from the origin
    std::vector<double> fval;
    std::size_t kopt = 0;
    std::vector<double> g;   // gradient of Q at x_kopt
    std::vector<double> hq;  // M of (4.6), symmetric, full storage
    std::vector<double> mu;  // the mu_l of (4.6)

    [[nodiscard]] std::span<const double> point(std::size_t j) const noexcept {
        return {points.data() + j * n, n};
    }
    std::span<double> point(std::size_t j) noexcept { return {points.data() + j * n, n}; }
    [[nodiscard]] std::span<const double> xopt() const noexcept { return point(kopt); }
    [[nodiscard]] double fopt() const noexcept { return fval[kopt]; }

    // out = (M + sum_l mu_l y_l y_l^T) v.
    void hess_mul(std::span<const double> v, std::span<double> out) const noexcept {
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t k = 0; k < n; ++k) s += hq[i * n + k] * v[k];
            out[i] = s;
        }
        for (std::size_t l = 0; l < m; ++l) {
            if (mu[l] == 0.0) continue;
            std::span<const double> yl = point(l);
            const double c = mu[l] * dot(yl, v);
            for (std::size_t i = 0; i < n; ++i) out[i] += c * yl[i];
        }
    }

    // The diagonal element (i, i) of the second derivative matrix.
    [[nodiscard]] double hess_diag(std::size_t i) const noexcept {
        double s = hq[i * n + i];
        for (std::size_t l = 0; l < m; ++l) s += mu[l] * point(l)[i] * point(l)[i];
        return s;
    }

    // Q(x_kopt + d) - Q(x_kopt) = g.d + (1/2) d^T (nabla^2 Q) d, by (4.7).
    [[nodiscard]] double change(std::span<const double> d, std::span<double> work) const noexcept {
        hess_mul(d, work);
        return dot(g, d) + 0.5 * dot(d, work);
    }

    // mu_t y_t y_t^T moves into M, so that mu_t is zero (section 4, after
    // (4.6)).
    void absorb(std::size_t t) noexcept {
        const double c = mu[t];
        if (c == 0.0) return;
        std::span<const double> yt = point(t);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < n; ++k) hq[i * n + k] += c * yt[i] * yt[k];
        mu[t] = 0.0;
    }

    // M gains the rank-two matrix of (5.3) and (6.17), so that the second
    // derivative matrix is unchanged when every point loses s and the mu
    // are kept: v s^T + s v^T with v = sum_l mu_l y_l - (sum_l mu_l) s / 2,
    // y_l the offsets from the old origin.
    void shift_hessian(std::span<const double> s, std::span<double> v) noexcept {
        double total = 0.0;
        for (std::size_t k = 0; k < n; ++k) v[k] = 0.0;
        for (std::size_t l = 0; l < m; ++l) {
            total += mu[l];
            std::span<const double> yl = point(l);
            for (std::size_t k = 0; k < n; ++k) v[k] += mu[l] * yl[k];
        }
        for (std::size_t k = 0; k < n; ++k) v[k] -= 0.5 * total * s[k];
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < n; ++k) hq[i * n + k] += v[i] * s[k] + s[i] * v[k];
    }
};

// ============================================================================
// Box: the bounds as offsets from the origin, a - x0 and b - x0 (3.2)
// ============================================================================
// A coordinate without a bound on a side has no entry there: no sentinel.
struct Box {
    explicit Box(std::size_t n) : has_lo(n, 0), has_hi(n, 0), lo(n, 0.0), hi(n, 0.0) {}

    std::vector<char> has_lo, has_hi;
    std::vector<double> lo, hi;

    void shift(std::span<const double> s) noexcept {
        for (std::size_t i = 0; i < s.size(); ++i) {
            if (has_lo[i]) lo[i] -= s[i];
            if (has_hi[i]) hi[i] -= s[i];
        }
    }
    [[nodiscard]] bool at_lo(std::size_t i, double x) const noexcept {
        return has_lo[i] && x <= lo[i];
    }
    [[nodiscard]] bool at_hi(std::size_t i, double x) const noexcept {
        return has_hi[i] && x >= hi[i];
    }
    // x clamped into the box, coordinate i.
    [[nodiscard]] double clamp(std::size_t i, double x) const noexcept {
        if (has_lo[i] && x < lo[i]) return lo[i];
        if (has_hi[i] && x > hi[i]) return hi[i];
        return x;
    }
};

// Which bound a coordinate of a step was placed on: the step routines say so
// explicitly, so that the new point lands on the bound exactly (the
// requirement after (3.2)) and is not left a rounding error away from it.
enum class Side : signed char { Free = 0, Lower = -1, Upper = 1 };

// x = base + d, coordinate by coordinate, with a coordinate placed on a
// bound taking the bound's own value and every coordinate kept in the box.
inline void place(const Box& box, std::span<const double> base, std::span<const double> d,
                  std::span<const Side> side, std::span<double> x) noexcept {
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (side[i] == Side::Lower)
            x[i] = box.lo[i];
        else if (side[i] == Side::Upper)
            x[i] = box.hi[i];
        else
            x[i] = box.clamp(i, base[i] + d[i]);
    }
}

}  // namespace flop::detail::bobyqa
