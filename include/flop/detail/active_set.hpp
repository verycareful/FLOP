// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// active_set - exact projection onto a polyhedron by a dual active-set method
// =============================================================================
//
// Given rows A_r . d <= b_r, find the point d of the polyhedron nearest to a
// query q, that is the minimiser of |d - q|^2 subject to the rows. This is
// the dual active-set method of Goldfarb and Idnani (1983, Mathematical
// Programming 27, 1-33) for the strictly convex quadratic programme with the
// identity as its Hessian, which makes their matrix J an orthogonal basis and
// their factorisation a QR of the active rows' transpose.
//
// The state is a working set W of rows held as equalities, the multipliers
// lambda_W >= 0 of those rows, and the point d = q - A_W^T lambda_W, which is
// the minimiser subject to W alone. Each iteration takes the most violated
// row p outside W and raises its multiplier from zero: the point moves along
// z = (I - P_W) n_p, the part of the row's normal outside the active row
// space, and the active multipliers move along r = (A_W A_W^T)^-1 A_W n_p.
// The move stops where p is satisfied (p joins W) or where an active
// multiplier reaches zero first (that row leaves W and the move continues).
// A row whose normal lies in the active row space (z = 0) with no multiplier
// able to decrease (r <= 0) certifies that the polyhedron is empty: r then
// gives the nonnegative combination of active rows that contradicts p. Every
// step increases the dual objective, so the method is finite (Goldfarb and
// Idnani, theorem 3).
//
// Factorisation. A_W^T = Q_1 R with Q = [Q_1 Q_2] orthogonal n x n and R
// upper triangular |W| x |W|, so A_W A_W^T = R^T R. Adding a row appends a
// column and zeroes its tail with Givens rotations from the right; removing
// a row deletes a column and re-triangularises with rotations on the rows.
// From it: z = Q_2 Q_2^T n_p, r = R^-1 Q_1^T n_p, the least-norm point of
// the active equalities for a right-hand side c is Q_1 R^-T c_W, and the
// multipliers at a query are R^-1 (Q_1^T q - R^-T b_W). The last three are
// what a parametric caller needs to follow the solution along a line of
// queries or right-hand sides, so they are exposed.
//
// No infinity, no NaN: an unbounded primal move is a flag, a missing dual
// bound is a flag, and every division has its denominator tested.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace flop::detail::cobyla {

class ActiveSetProjection {
public:
    // A row whose normal's part outside the active row space is below this
    // fraction of the normal's length is treated as dependent on the active
    // rows: adding it would give a singular R.
    static constexpr double kDependentTol = 1e-13;
    // A row is violated when A_r d - b_r exceeds this fraction of the row's
    // scale |b_r| + |A_r| |d|; below it the row counts as satisfied, so that
    // the last bit of rounding on an active row never re-enters the loop.
    static constexpr double kRowTol = 1e-14;
    // Iteration cap per projection, a multiple of the row count: the method
    // is finite, so reaching the cap is an internal error, reported as one.
    static constexpr std::size_t kSweepsPerRow = 64;

    // Empties the working set and sizes the factors for n variables.
    void reset(std::size_t n) {
        n_ = n;
        Q_.assign(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i) Q_[i * n + i] = 1.0;
        R_.assign(n * n, 0.0);
        active_.clear();
        lambda_.clear();
        v_.assign(n, 0.0);
        r_.assign(n, 0.0);
        z_.assign(n, 0.0);
        y_.assign(n, 0.0);
        blocking_.clear();
    }

    // Projects q onto {A d <= b} (rows of A row-major, n per row; rownorm2 the
    // squared row norms, a zero-norm row is skipped as it cannot be moved)
    // and writes d. Starts from the current working set, which is kept for
    // the next call. Returns false when the polyhedron is empty; the
    // certificate is then in blocking_row() and blocking_combination().
    bool project(std::span<const double> q, std::span<const double> A, std::span<const double> b,
                 std::span<const double> rownorm2, std::span<double> d) {
        const std::size_t rows = b.size();
        // Multipliers for the current working set at this query, dropping
        // any that turn negative, so that the invariant lambda >= 0 holds
        // before the first row is added.
        for (;;) {
            multipliers_at(q, b);
            std::size_t worst = 0;
            double most_negative = 0.0;
            for (std::size_t j = 0; j < active_.size(); ++j)
                if (lambda_[j] < most_negative) {
                    most_negative = lambda_[j];
                    worst = j;
                }
            if (most_negative == 0.0) break;
            remove_row(worst);
        }
        point_at(q, b, d);

        const std::size_t cap = kSweepsPerRow * (rows + 1);
        for (std::size_t iter = 0; iter < cap; ++iter) {
            // The most violated row outside the working set.
            const double dnorm = norm(d);
            std::size_t p = rows;
            double worst = 0.0;
            for (std::size_t r = 0; r < rows; ++r) {
                if (rownorm2[r] == 0.0 || is_active(r)) continue;
                const double viol = row_dot(A, r, d) - b[r];
                const double scale = std::fabs(b[r]) + std::sqrt(rownorm2[r]) * dnorm;
                if (viol > kRowTol * scale && viol > worst) {
                    worst = viol;
                    p = r;
                }
            }
            if (p == rows) {
                point_at(q, b, d);
                return true;
            }
            // Raise the multiplier of p, dropping active rows whose
            // multiplier reaches zero on the way, until p is satisfied.
            double accumulated = 0.0;
            for (std::size_t inner = 0; inner < cap; ++inner) {
                const double zn2 = directions(A, p);  // fills r_ and z_, returns |z|^2
                const double np = std::sqrt(rownorm2[p]);
                const bool movable = zn2 > (kDependentTol * np) * (kDependentTol * np);
                const double viol = row_dot(A, p, d) - b[p];
                bool have_dual = false;
                double tau_dual = 0.0;
                std::size_t jd = 0;
                for (std::size_t j = 0; j < active_.size(); ++j) {
                    if (r_[j] <= 0.0) continue;
                    const double t = lambda_[j] / r_[j];
                    if (!have_dual || t < tau_dual) {
                        have_dual = true;
                        tau_dual = t;
                        jd = j;
                    }
                }
                if (!movable && !have_dual) {
                    blocking_row_ = p;
                    blocking_.assign(r_.begin(),
                                     r_.begin() + static_cast<std::ptrdiff_t>(active_.size()));
                    return false;
                }
                const double tau_full = movable ? std::max(viol, 0.0) / zn2 : 0.0;
                if (!movable || (have_dual && tau_dual < tau_full)) {
                    move(tau_dual, d);
                    accumulated += tau_dual;
                    remove_row(jd);
                    continue;
                }
                move(tau_full, d);
                accumulated += tau_full;
                add_row(A, p, accumulated);
                break;
            }
        }
        throw std::runtime_error("flop: active-set projection did not terminate (internal error)");
    }

    [[nodiscard]] std::span<const std::size_t> active() const { return active_; }
    [[nodiscard]] std::span<const double> multipliers() const { return lambda_; }
    [[nodiscard]] std::size_t blocking_row() const { return blocking_row_; }
    // The combination r of active rows (aligned with active()) with
    // n_p = A_W^T r and r <= 0, valid after project() returned false.
    [[nodiscard]] std::span<const double> blocking_combination() const { return blocking_; }

    // The least-norm point of {A_W d = c_W} for a full right-hand side c
    // indexed by row: u = Q_1 R^-T c_W.
    void least_norm_point(std::span<const double> c, std::span<double> u) {
        const std::size_t w = active_.size();
        for (std::size_t j = 0; j < w; ++j) y_[j] = c[active_[j]];
        solve_lower(y_, w);  // y = R^-T c_W
        for (std::size_t i = 0; i < n_; ++i) {
            double s = 0.0;
            for (std::size_t k = 0; k < w; ++k) s += Q_[i * n_ + k] * y_[k];
            u[i] = s;
        }
    }

    // The part of x outside the active row space: z = Q_2 Q_2^T x.
    void complement(std::span<const double> x, std::span<double> z) {
        const std::size_t w = active_.size();
        for (std::size_t k = 0; k < n_; ++k) {
            double s = 0.0;
            for (std::size_t i = 0; i < n_; ++i) s += Q_[i * n_ + k] * x[i];
            v_[k] = s;
        }
        for (std::size_t i = 0; i < n_; ++i) {
            double s = 0.0;
            for (std::size_t k = w; k < n_; ++k) s += Q_[i * n_ + k] * v_[k];
            z[i] = s;
        }
    }

    // How the multipliers change per unit of query along x: R^-1 Q_1^T x,
    // aligned with active().
    void multiplier_slope(std::span<const double> x, std::span<double> slope) {
        const std::size_t w = active_.size();
        for (std::size_t k = 0; k < w; ++k) {
            double s = 0.0;
            for (std::size_t i = 0; i < n_; ++i) s += Q_[i * n_ + k] * x[i];
            y_[k] = s;
        }
        solve_upper(y_, w);
        for (std::size_t k = 0; k < w; ++k) slope[k] = y_[k];
    }

private:
    [[nodiscard]] bool is_active(std::size_t r) const {
        return std::ranges::find(active_, r) != active_.end();
    }

    static double norm(std::span<const double> x) {
        double s = 0.0;
        for (double v : x) s += v * v;
        return std::sqrt(s);
    }

    [[nodiscard]] double row_dot(std::span<const double> A, std::size_t r,
                                 std::span<const double> x) const {
        double s = 0.0;
        for (std::size_t i = 0; i < n_; ++i) s += A[r * n_ + i] * x[i];
        return s;
    }

    // y[0..w) <- R^-1 y (R upper triangular, w x w).
    void solve_upper(std::vector<double>& y, std::size_t w) const {
        for (std::size_t jj = w; jj-- > 0;) {
            double s = y[jj];
            for (std::size_t k = jj + 1; k < w; ++k) s -= R_[jj * n_ + k] * y[k];
            y[jj] = s / R_[jj * n_ + jj];
        }
    }

    // y[0..w) <- R^-T y.
    void solve_lower(std::vector<double>& y, std::size_t w) const {
        for (std::size_t jj = 0; jj < w; ++jj) {
            double s = y[jj];
            for (std::size_t k = 0; k < jj; ++k) s -= R_[k * n_ + jj] * y[k];
            y[jj] = s / R_[jj * n_ + jj];
        }
    }

    // y = Q_1^T q - R^-T b_W; lambda = R^-1 y.
    void multipliers_at(std::span<const double> q, std::span<const double> b) {
        const std::size_t w = active_.size();
        for (std::size_t j = 0; j < w; ++j) y_[j] = b[active_[j]];
        solve_lower(y_, w);
        for (std::size_t k = 0; k < w; ++k) {
            double s = 0.0;
            for (std::size_t i = 0; i < n_; ++i) s += Q_[i * n_ + k] * q[i];
            y_[k] = s - y_[k];
        }
        lambda_.assign(y_.begin(), y_.begin() + static_cast<std::ptrdiff_t>(w));
        solve_upper(lambda_, w);
    }

    // d = q - Q_1 (Q_1^T q - R^-T b_W): the minimiser subject to the working
    // set as equalities, from the factors rather than from the multipliers,
    // so that the active rows hold to rounding.
    void point_at(std::span<const double> q, std::span<const double> b, std::span<double> d) {
        const std::size_t w = active_.size();
        for (std::size_t j = 0; j < w; ++j) y_[j] = b[active_[j]];
        solve_lower(y_, w);
        for (std::size_t k = 0; k < w; ++k) {
            double s = 0.0;
            for (std::size_t i = 0; i < n_; ++i) s += Q_[i * n_ + k] * q[i];
            y_[k] = s - y_[k];
        }
        for (std::size_t i = 0; i < n_; ++i) {
            double s = q[i];
            for (std::size_t k = 0; k < w; ++k) s -= Q_[i * n_ + k] * y_[k];
            d[i] = s;
        }
    }

    // For row p: v = Q^T n_p; r = R^-1 v[0..w); z = Q_2 v[w..n). Returns |z|^2.
    double directions(std::span<const double> A, std::size_t p) {
        const std::size_t w = active_.size();
        for (std::size_t k = 0; k < n_; ++k) {
            double s = 0.0;
            for (std::size_t i = 0; i < n_; ++i) s += Q_[i * n_ + k] * A[p * n_ + i];
            v_[k] = s;
        }
        for (std::size_t k = 0; k < w; ++k) r_[k] = v_[k];
        solve_upper(r_, w);
        double zn2 = 0.0;
        for (std::size_t k = w; k < n_; ++k) zn2 += v_[k] * v_[k];
        for (std::size_t i = 0; i < n_; ++i) {
            double s = 0.0;
            for (std::size_t k = w; k < n_; ++k) s += Q_[i * n_ + k] * v_[k];
            z_[i] = s;
        }
        return zn2;
    }

    // d -= tau z; lambda_W -= tau r.
    void move(double tau, std::span<double> d) {
        for (std::size_t i = 0; i < n_; ++i) d[i] -= tau * z_[i];
        for (std::size_t j = 0; j < active_.size(); ++j) lambda_[j] -= tau * r_[j];
    }

    // Applies the rotation (c, s) to columns a and b of Q: Q <- Q G^T.
    void rotate_columns(std::size_t a, std::size_t bcol, double c, double s) {
        for (std::size_t i = 0; i < n_; ++i) {
            const double qa = Q_[i * n_ + a];
            const double qb = Q_[i * n_ + bcol];
            Q_[i * n_ + a] = c * qa + s * qb;
            Q_[i * n_ + bcol] = -s * qa + c * qb;
        }
    }

    // The rotation zeroing y against x: returns (c, s) with c x + s y = hypot
    // and -s x + c y = 0.
    static void givens(double x, double y, double& c, double& s) {
        const double h = std::hypot(x, y);
        if (h == 0.0) {
            c = 1.0;
            s = 0.0;
            return;
        }
        c = x / h;
        s = y / h;
    }

    // Appends row p to the working set with multiplier lambda_p. v_ holds
    // Q^T n_p from directions(); its tail beyond w is rotated into position
    // w, which becomes the new diagonal of R.
    void add_row(std::span<const double> A, std::size_t p, double lambda_p) {
        const std::size_t w = active_.size();
        for (std::size_t k = 0; k < n_; ++k) {
            double s = 0.0;
            for (std::size_t i = 0; i < n_; ++i) s += Q_[i * n_ + k] * A[p * n_ + i];
            v_[k] = s;
        }
        for (std::size_t k = n_ - 1; k > w; --k) {
            double c = 0.0, s = 0.0;
            givens(v_[k - 1], v_[k], c, s);
            v_[k - 1] = c * v_[k - 1] + s * v_[k];
            v_[k] = 0.0;
            rotate_columns(k - 1, k, c, s);
        }
        for (std::size_t i = 0; i <= w; ++i) R_[i * n_ + w] = v_[i];
        for (std::size_t i = w + 1; i < n_; ++i) R_[i * n_ + w] = 0.0;
        active_.push_back(p);
        lambda_.push_back(lambda_p);
    }

    // Removes the row at position j of the working set: column j of R is
    // deleted and the Hessenberg tail re-triangularised.
    void remove_row(std::size_t j) {
        const std::size_t w = active_.size();
        for (std::size_t col = j; col + 1 < w; ++col)
            for (std::size_t i = 0; i < n_; ++i) R_[i * n_ + col] = R_[i * n_ + col + 1];
        for (std::size_t i = 0; i < n_; ++i) R_[i * n_ + (w - 1)] = 0.0;
        for (std::size_t col = j; col + 1 < w; ++col) {
            double c = 0.0, s = 0.0;
            givens(R_[col * n_ + col], R_[(col + 1) * n_ + col], c, s);
            for (std::size_t k = col; k + 1 < w; ++k) {
                const double top = R_[col * n_ + k];
                const double bottom = R_[(col + 1) * n_ + k];
                R_[col * n_ + k] = c * top + s * bottom;
                R_[(col + 1) * n_ + k] = -s * top + c * bottom;
            }
            rotate_columns(col, col + 1, c, s);
        }
        active_.erase(active_.begin() + static_cast<std::ptrdiff_t>(j));
        lambda_.erase(lambda_.begin() + static_cast<std::ptrdiff_t>(j));
    }

    std::size_t n_ = 0;
    std::vector<double> Q_;  // n x n, row-major; the first |W| columns span the active rows
    std::vector<double> R_;  // n x n capacity, row-major, upper triangular in its |W| x |W| corner
    std::vector<std::size_t> active_;  // rows in the working set, in factorisation order
    std::vector<double> lambda_;       // their multipliers
    std::vector<double> v_, r_, z_, y_;
    std::size_t blocking_row_ = 0;
    std::vector<double> blocking_;
};

}  // namespace flop::detail::cobyla
