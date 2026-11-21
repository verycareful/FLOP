// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// What the tests of BOBYQA's model and inverse matrix share: the matrix W of
// Powell's system (2.6), built straight from its definition (2.8) and (2.9)
// for any set of interpolation points, and inverted explicitly by Gauss-Jordan
// elimination with partial pivoting. H = W^-1 of (2.7) obtained this way owes
// nothing to the library's formulas (2.11) to (2.15), (4.9) to (4.14) and
// (6.14) to (6.17), so it is what those are held against.
//
// Points are offsets from the origin x0, m rows of n values, as the library
// holds them. H is stored in full, (m + n + 1) square, in the report's order:
// the m points, then the row of ones, then the n coordinates.
//
// The starting points of section 2 are generated here from the report's
// equations (2.10), (2.3) and (2.4), not from the library's pair function.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "flop/detail/bobyqa_model.hpp"

namespace v0121 {

using Points = std::vector<double>;  // m x n, row-major

// A small deterministic generator, so that every machine draws the same
// numbers: splitmix64, uniform in [-1, 1).
class Draw {
public:
    explicit Draw(std::uint64_t seed) : state_(seed) {}
    double next() {
        state_ += UINT64_C(0x9E3779B97F4A7C15);
        std::uint64_t z = state_;
        z = (z ^ (z >> 30U)) * UINT64_C(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27U)) * UINT64_C(0x94D049BB133111EB);
        z ^= z >> 31U;
        return 2.0 * (static_cast<double>(z >> 11U) / 9007199254740992.0) - 1.0;
    }

    // An index in [0, k).
    std::size_t below(std::size_t k) {
        const double u = 0.5 * (next() + 1.0);
        return std::min(k - 1, static_cast<std::size_t>(u * static_cast<double>(k)));
    }

private:
    std::uint64_t state_;
};

struct Dense {
    std::size_t size = 0;
    std::vector<double> a;
    explicit Dense(std::size_t k) : size(k), a(k * k, 0.0) {}
    double& operator()(std::size_t i, std::size_t j) { return a[i * size + j]; }
    [[nodiscard]] double operator()(std::size_t i, std::size_t j) const { return a[i * size + j]; }
};

// W of (2.7): A of (2.8) in the corner, Y of (2.9) below it, Y^T beside it.
inline Dense build_w(const Points& y, std::size_t n, std::size_t m) {
    Dense w(m + n + 1);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < n; ++k) s += y[i * n + k] * y[j * n + k];
            w(i, j) = 0.5 * s * s;
        }
    for (std::size_t j = 0; j < m; ++j) {
        w(m, j) = 1.0;
        w(j, m) = 1.0;
        for (std::size_t k = 0; k < n; ++k) {
            w(m + 1 + k, j) = y[j * n + k];
            w(j, m + 1 + k) = y[j * n + k];
        }
    }
    return w;
}

// The inverse by Gauss-Jordan with partial pivoting. Returns an empty matrix
// when a pivot is zero.
inline Dense invert(const Dense& w) {
    const std::size_t k = w.size;
    Dense a = w;
    Dense inv(k);
    for (std::size_t i = 0; i < k; ++i) inv(i, i) = 1.0;
    for (std::size_t col = 0; col < k; ++col) {
        std::size_t piv = col;
        for (std::size_t r = col + 1; r < k; ++r)
            if (std::fabs(a(r, col)) > std::fabs(a(piv, col))) piv = r;
        if (a(piv, col) == 0.0) return Dense(0);
        if (piv != col)
            for (std::size_t c = 0; c < k; ++c) {
                std::swap(a(piv, c), a(col, c));
                std::swap(inv(piv, c), inv(col, c));
            }
        const double d = a(col, col);
        for (std::size_t c = 0; c < k; ++c) {
            a(col, c) /= d;
            inv(col, c) /= d;
        }
        for (std::size_t r = 0; r < k; ++r) {
            if (r == col) continue;
            const double f = a(r, col);
            if (f == 0.0) continue;
            for (std::size_t c = 0; c < k; ++c) {
                a(r, c) -= f * a(col, c);
                inv(r, c) -= f * inv(col, c);
            }
        }
    }
    return inv;
}

inline Dense explicit_h(const Points& y, std::size_t n, std::size_t m) {
    return invert(build_w(y, n, m));
}

// The interpolation points of section 2 with the multipliers of (2.10): y_1 =
// x0, y_{i+1} = alpha_i e_i, y_{n+i+1} = beta_i e_i while n + i + 1 <= m, and
// for j >= 2n + 2 the point y_{p(j)+1} + y_{q(j)+1} - x0 of (2.3), with p(j)
// cycling through 1..n (2.4) and q(j) = p(j) + l or p(j) + l - n in the l-th
// cycle, whichever lies in [1, n].
inline Points coordinate_points(std::span<const double> alpha, std::span<const double> beta,
                                std::size_t m) {
    const std::size_t n = alpha.size();
    Points y(m * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        if (i + 1 < m) y[(i + 1) * n + i] = alpha[i];
        if (n + i + 1 < m) y[(n + i + 1) * n + i] = beta[i];
    }
    for (std::size_t j = 2 * n + 1; j < m; ++j) {  // 0-based; the report's j is j + 1
        const std::size_t e = j - (2 * n + 1);
        const std::size_t p = e % n;
        const std::size_t cycle = e / n + 1;
        std::size_t q = p + cycle;
        if (q >= n) q -= n;
        y[j * n + p] = alpha[p];
        y[j * n + q] = alpha[q];
    }
    return y;
}

// How far the library's Z and B are from an explicit H, scaled by the
// largest entry of H and 1: the largest error over Omega = Z Z^T, the
// m x n block B(0..m-1, .) = Xi^T without its first column, and
// B(m..m+n-1, .) = Upsilon without its first row and column.
struct HError {
    double omega = 0.0;
    double xi = 0.0;
    double upsilon = 0.0;
    [[nodiscard]] double worst() const { return std::max({omega, xi, upsilon}); }
};

inline HError compare_to_h(const flop::detail::bobyqa::Inverse& inv, const Dense& h, std::size_t n,
                           std::size_t m) {
    double scale = 1.0;
    for (const double v : h.a) scale = std::max(scale, std::fabs(v));
    HError e;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) {
            double s = 0.0;
            for (std::size_t c = 0; c < inv.columns(); ++c) s += inv.z(i, c) * inv.z(j, c);
            e.omega = std::max(e.omega, std::fabs(s - h(i, j)) / scale);
        }
    for (std::size_t j = 0; j < m; ++j)
        for (std::size_t k = 0; k < n; ++k)
            e.xi = std::max(e.xi, std::fabs(inv.b(j, k) - h(m + 1 + k, j)) / scale);
    for (std::size_t p = 0; p < n; ++p)
        for (std::size_t q = 0; q < n; ++q)
            e.upsilon =
                std::max(e.upsilon, std::fabs(inv.b(m + p, q) - h(m + 1 + p, m + 1 + q)) / scale);
    return e;
}

// The vector w of (4.10) for the offset x: w_i = (1/2)(y_i . x)^2, then 1, then x.
inline std::vector<double> w_vector(const Points& y, std::size_t n, std::size_t m,
                                    std::span<const double> x) {
    std::vector<double> w(m + n + 1);
    for (std::size_t i = 0; i < m; ++i) {
        double s = 0.0;
        for (std::size_t k = 0; k < n; ++k) s += y[i * n + k] * x[k];
        w[i] = 0.5 * s * s;
    }
    w[m] = 1.0;
    for (std::size_t k = 0; k < n; ++k) w[m + 1 + k] = x[k];
    return w;
}

inline std::vector<double> times(const Dense& h, const std::vector<double>& v) {
    std::vector<double> r(h.size, 0.0);
    for (std::size_t i = 0; i < h.size; ++i)
        for (std::size_t j = 0; j < h.size; ++j) r[i] += h(i, j) * v[j];
    return r;
}

inline double dot_all(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// The multiplier pairs of the coordinate points for a run of the starting
// procedure: alpha_i = delta and beta_i = -delta on an interior coordinate,
// delta and 2 delta on a lower bound, -delta and -2 delta on an upper one.
enum class Place : std::uint8_t { Interior, OnLower, OnUpper };

inline void multipliers(const std::vector<Place>& place, double delta, std::vector<double>& alpha,
                        std::vector<double>& beta) {
    alpha.assign(place.size(), 0.0);
    beta.assign(place.size(), 0.0);
    for (std::size_t i = 0; i < place.size(); ++i) {
        switch (place[i]) {
            case Place::Interior:
                alpha[i] = delta;
                beta[i] = -delta;
                break;
            case Place::OnLower:
                alpha[i] = delta;
                beta[i] = 2.0 * delta;
                break;
            case Place::OnUpper:
                alpha[i] = -delta;
                beta[i] = -2.0 * delta;
                break;
        }
    }
}

}  // namespace v0121
