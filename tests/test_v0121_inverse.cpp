// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: BOBYQA's inverse matrix H = W^-1, held against an explicit inverse.
//
// Powell, DAMTP 2009/NA06. The library keeps H as Omega = Z Z^T and the
// (m + n) x n matrix B (section 4), and writes it down for the starting points
// by (2.11) to (2.15), updates it by (4.9) to (4.14) with the identities
// (4.13), and moves it to a new origin by (6.14) to (6.17). Every one of those
// is compared here with the inverse of W of (2.6) to (2.9), built from its
// definition for the points at hand and inverted by Gauss-Jordan elimination
// (v0121_reference.hpp), for every m from n + 2 to (n + 1)(n + 2) / 2 with
// n <= 5 where the starting matrix is concerned.
//
// The Z of (2.14) as printed in the copy of the report this was read from has
// the first entry sqrt(2) / (alpha beta); its three entries then do not sum
// to zero, and Omega Y^T = 0 needs them to. The library's first entry is
// -sqrt(2) / (alpha beta), and the explicit inverse agrees with it.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

#include "flop/detail/bobyqa_model.hpp"
#include "v0121_problems.hpp"
#include "v0121_reference.hpp"

namespace {

using flop::detail::bobyqa::Inverse;
using v0121::Dense;
using v0121::Draw;
using v0121::Place;
using v0121::Points;

constexpr double kTol = 1e-10;
// The vectors of (4.13) are products with entries of H that reach several
// hundred after a few random replacements, so they are held to a relative error.
constexpr double kVlagTol = 1e-8;

std::size_t max_m(std::size_t n) {
    return (n + 1) * (n + 2) / 2;
}

// A mix of placements, so that no two neighbouring coordinates agree.
std::vector<Place> mixed(std::size_t n) {
    std::vector<Place> p(n);
    for (std::size_t i = 0; i < n; ++i) p[i] = static_cast<Place>(i % 3);
    return p;
}

struct PointSet {
    std::size_t n;
    std::size_t m;
    std::vector<double> alpha, beta;
    Points y;
};

PointSet start(std::size_t n, std::size_t m, const std::vector<Place>& place, double delta) {
    PointSet s{.n = n, .m = m, .alpha = {}, .beta = {}, .y = {}};
    v0121::multipliers(place, delta, s.alpha, s.beta);
    s.y = v0121::coordinate_points(s.alpha, s.beta, m);
    return s;
}

// The scratch the library's updates take.
struct Work {
    std::vector<double> vlag, wv, zw, d, sum, h, u, gamma, cmat, zg, diff;
    Work(std::size_t n, std::size_t m)
        : vlag(m + n),
          wv(m),
          zw(m - n - 1),
          d(n),
          sum(n),
          h(m + n),
          u(m + n),
          gamma(n * m),
          cmat(m * n),
          zg(m - n - 1),
          diff(n) {}
};

// Replaces point t by x in y and the library's inverse, after checking the
// vectors of (4.13) and the parameters (4.11) against the explicit H. Returns
// false when sigma is too small to be a sound step for a test of formulas.
bool replace_point(Inverse& inv, Points& y, std::size_t n, std::size_t m, std::size_t s,
                   std::size_t t, const std::vector<double>& x, Work& w, double* sigma_out) {
    const Dense h = v0121::explicit_h(y, n, m);
    const std::vector<double> wv = v0121::w_vector(y, n, m, x);
    const std::vector<double> hw = v0121::times(h, wv);
    const double beta_expected = 0.5 * std::pow(v0121::dot_all(x, x), 2) - v0121::dot_all(wv, hw);
    const double beta = inv.vlag_beta(y, s, x, w.vlag, w.wv, w.zw, w.d, w.sum);
    // (4.13): the library's vlag is H w without its (m + 1)-th component.
    for (std::size_t i = 0; i < m; ++i)
        EXPECT_NEAR(w.vlag[i], hw[i], kVlagTol * (1.0 + std::fabs(hw[i]))) << "vlag " << i;
    for (std::size_t k = 0; k < n; ++k)
        EXPECT_NEAR(w.vlag[m + k], hw[m + 1 + k], kVlagTol * (1.0 + std::fabs(hw[m + 1 + k])))
            << "vlag " << m + k;
    EXPECT_NEAR(beta, beta_expected, kVlagTol * (1.0 + std::fabs(beta_expected)));
    const double alpha = h(t, t);
    const double tau = hw[t];
    const double sigma = alpha * beta_expected + tau * tau;
    *sigma_out = sigma;
    if (!(sigma > 1e-4)) return false;
    inv.update(t, w.vlag, beta, w.h, w.u);
    std::ranges::copy(x, y.begin() + static_cast<std::ptrdiff_t>(t * n));
    return true;
}

}  // namespace

TEST(V0121Inverse, TheStartingMatrixIsTheInverseOfWForEveryPointCount) {
    // (2.11) to (2.15) against the explicit inverse, for every legal m with
    // n <= 5, with the points on and off the bounds and several radii.
    const std::vector<std::vector<Place>> kinds = {
        {}, {Place::OnLower}, {Place::OnUpper}, {Place::Interior}};
    for (std::size_t n = 1; n <= 5; ++n) {
        for (std::size_t m = n + 2; m <= max_m(n); ++m) {
            for (const double delta : {0.5, 1.0, 3.0}) {
                for (int kind = 0; kind < 5; ++kind) {
                    std::vector<Place> place(n, Place::Interior);
                    if (kind == 1) place.assign(n, Place::OnLower);
                    if (kind == 2) place.assign(n, Place::OnUpper);
                    if (kind == 3) place = mixed(n);
                    const PointSet s = start(n, m, place, delta);
                    Inverse inv(n, m);
                    inv.set_coordinate(s.alpha, s.beta);
                    const Dense h = v0121::explicit_h(s.y, n, m);
                    ASSERT_EQ(h.size, m + n + 1) << "W is singular: n " << n << " m " << m;
                    const v0121::HError e = v0121::compare_to_h(inv, h, n, m);
                    EXPECT_LT(e.worst(), kTol)
                        << "n " << n << " m " << m << " delta " << delta << " kind " << kind
                        << " (omega " << e.omega << ", xi " << e.xi << ", upsilon " << e.upsilon
                        << ")";
                }
            }
        }
    }
}

TEST(V0121Inverse, TheColumnsOfZAreTheOnesOf214And215) {
    // (2.14): for l <= min(n, m - n - 1), three nonzero entries in the l-th
    // column, at the rows 1, l + 1 and n + l + 1 (1-based). (2.15): for the
    // columns beyond n, four, at 1, n + l + 1, p + 1 and q + 1.
    const double r2 = std::numbers::sqrt2;
    for (std::size_t n = 1; n <= 5; ++n) {
        for (std::size_t m = n + 2; m <= max_m(n); ++m) {
            const PointSet s = start(n, m, mixed(n), 0.75);
            Inverse inv(n, m);
            inv.set_coordinate(s.alpha, s.beta);
            const std::size_t nz = m - n - 1;
            for (std::size_t c = 0; c < nz; ++c) {
                std::vector<double> want(m, 0.0);
                if (c < n) {
                    const double a = s.alpha[c], b = s.beta[c];
                    want[0] = -r2 / (a * b);
                    want[c + 1] = r2 / (a * (b - a));
                    want[n + c + 1] = r2 / (b * (a - b));
                } else {
                    // The pair of the (c - n + 1)-th extra point, from (2.3) and
                    // (2.4) written out again.
                    const std::size_t e = c - n;
                    const std::size_t p = e % n;
                    std::size_t q = p + e / n + 1;
                    if (q >= n) q -= n;
                    const double r = 1.0 / (s.alpha[p] * s.alpha[q]);
                    want[0] = r;
                    want[n + c + 1] = r;
                    want[p + 1] = -r;
                    want[q + 1] = -r;
                }
                for (std::size_t i = 0; i < m; ++i)
                    EXPECT_NEAR(inv.z(i, c), want[i], 1e-14 * (1.0 + std::fabs(want[i])))
                        << "n " << n << " m " << m << " column " << c << " row " << i;
            }
        }
    }
}

TEST(V0121Inverse, TheLastRowsOfZAreADiagonalMatrix) {
    // Section 2: the square matrix of the last m - n - 1 rows of Z is
    // diagonal.
    for (std::size_t n = 1; n <= 5; ++n)
        for (std::size_t m = n + 2; m <= max_m(n); ++m) {
            const PointSet s = start(n, m, mixed(n), 1.0);
            Inverse inv(n, m);
            inv.set_coordinate(s.alpha, s.beta);
            const std::size_t nz = m - n - 1;
            for (std::size_t r = 0; r < nz; ++r)
                for (std::size_t c = 0; c < nz; ++c) {
                    if (r == c) continue;
                    EXPECT_EQ(inv.z(n + 1 + r, c), 0.0) << "n " << n << " m " << m;
                }
        }
}

namespace {

// Omega Y^T = 0 (the block (1, 2) of H W = I): every row of Omega sums to
// zero and has zero first moment, that is, every column of Z does.
void expect_z_orthogonal_to_y(const Inverse& inv, const Points& y, std::size_t n, std::size_t m,
                              double tol, const char* where) {
    for (std::size_t c = 0; c < inv.columns(); ++c) {
        double sum = 0.0, scale = 0.0;
        std::vector<double> moment(n, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            sum += inv.z(i, c);
            scale = std::max(scale, std::fabs(inv.z(i, c)));
            for (std::size_t k = 0; k < n; ++k) moment[k] += inv.z(i, c) * y[i * n + k];
        }
        EXPECT_NEAR(sum, 0.0, tol * (1.0 + scale)) << where << " column " << c;
        for (std::size_t k = 0; k < n; ++k)
            EXPECT_NEAR(moment[k], 0.0, tol * (1.0 + scale)) << where << " column " << c;
    }
}

}  // namespace

TEST(V0121Inverse, TheColumnsOfZSumToZeroWithZeroFirstMoment) {
    for (std::size_t n = 1; n <= 5; ++n)
        for (std::size_t m = n + 2; m <= max_m(n); ++m) {
            const PointSet s = start(n, m, mixed(n), 0.5);
            Inverse inv(n, m);
            inv.set_coordinate(s.alpha, s.beta);
            expect_z_orthogonal_to_y(inv, s.y, n, m, 1e-12, "start");
        }
}

TEST(V0121Inverse, TheLagrangeFunctionsInterpolateAtEveryPoint) {
    // vlag of (4.13) at x = y_i is H w = H W e_i = e_i whatever the point s the
    // update is centred on: Lambda_j(y_i) = delta_ij and the gradient part is
    // zero.
    for (std::size_t n = 1; n <= 4; ++n)
        for (std::size_t m = n + 2; m <= max_m(n); m += (m == n + 2 ? 1 : 2)) {
            const PointSet st = start(n, m, mixed(n), 0.5);
            Inverse inv(n, m);
            inv.set_coordinate(st.alpha, st.beta);
            Work w(n, m);
            for (std::size_t s = 0; s < m; s += 2)
                for (std::size_t i = 0; i < m; ++i) {
                    const std::vector<double> x(
                        st.y.begin() + static_cast<std::ptrdiff_t>(i * n),
                        st.y.begin() + static_cast<std::ptrdiff_t>((i + 1) * n));
                    (void)inv.vlag_beta(st.y, s, x, w.vlag, w.wv, w.zw, w.d, w.sum);
                    for (std::size_t j = 0; j < m; ++j)
                        EXPECT_NEAR(w.vlag[j], i == j ? 1.0 : 0.0, 1e-9)
                            << "n " << n << " m " << m << " s " << s << " i " << i << " j " << j;
                    for (std::size_t k = 0; k < n; ++k) EXPECT_NEAR(w.vlag[m + k], 0.0, 1e-9);
                }
        }
}

TEST(V0121Inverse, TheUpdateIsTheInverseOfTheNewW) {
    // (4.9) to (4.14): replace one point at a time, by a point anywhere near
    // the others, and compare with the explicit inverse of the new W after
    // each. The vectors and parameters of (4.10), (4.11) and (4.13) are
    // checked on the way in replace_point, and sigma = alpha beta + tau^2 is
    // positive for every step taken.
    Draw rng(0x5EEDF10DULL);
    std::size_t steps = 0;
    for (std::size_t n = 1; n <= 4; ++n) {
        for (const std::size_t m : {n + 2, 2 * n + 1, max_m(n)}) {
            PointSet st = start(n, m, mixed(n), 0.5);
            Inverse inv(n, m);
            inv.set_coordinate(st.alpha, st.beta);
            Work w(n, m);
            for (int step = 0; step < 14; ++step) {
                const std::size_t s = rng.below(m);
                std::size_t t = rng.below(m);
                if (t == s) t = (t + 1) % m;
                std::vector<double> x(n);
                for (std::size_t k = 0; k < n; ++k) x[k] = 0.9 * rng.next();
                double sigma = 0.0;
                if (!replace_point(inv, st.y, n, m, s, t, x, w, &sigma)) continue;
                ++steps;
                const Dense h = v0121::explicit_h(st.y, n, m);
                ASSERT_EQ(h.size, m + n + 1);
                const v0121::HError e = v0121::compare_to_h(inv, h, n, m);
                EXPECT_LT(e.worst(), 1e-8)
                    << "n " << n << " m " << m << " step " << step << " t " << t << " (omega "
                    << e.omega << ", xi " << e.xi << ", upsilon " << e.upsilon << ")";
                expect_z_orthogonal_to_y(inv, st.y, n, m, 1e-8, "update");
                // Row t of Z has only its first entry beyond the rotation, so
                // Omega e_t is that entry times the first column.
                for (std::size_t c = 1; c < inv.columns(); ++c) EXPECT_EQ(inv.z(t, c), 0.0);
                std::vector<double> col(m);
                inv.omega_col_updated(t, col);
                for (std::size_t i = 0; i < m; ++i)
                    EXPECT_NEAR(col[i], h(i, t), 1e-8 * (1.0 + std::fabs(h(i, t))));
                inv.omega_col(t, col);
                for (std::size_t i = 0; i < m; ++i)
                    EXPECT_NEAR(col[i], h(i, t), 1e-8 * (1.0 + std::fabs(h(i, t))));
                std::vector<double> diag(m);
                inv.omega_diag_all(diag);
                for (std::size_t i = 0; i < m; ++i) {
                    EXPECT_NEAR(diag[i], h(i, i), 1e-8 * (1.0 + std::fabs(h(i, i))));
                    EXPECT_GE(inv.omega_diag(i), 0.0);
                }
            }
        }
    }
    EXPECT_GT(steps, 100u) << "most steps must have been sound ones";
}

TEST(V0121Inverse, TheParametersOfAStepAlongALineHaveTheClosedFormOf424) {
    // x+ = x_s + a (y_j - x_s): beta = a^2 (1 - a)^2 {(1/2)|y_j - x_s|^4 -
    // e_j^T A Omega A e_j} (4.24), and so beta <= (1/2) a^2 (1 - a)^2 |y_j - x_s|^4.
    for (std::size_t n = 1; n <= 4; ++n) {
        const std::size_t m = 2 * n + 1;
        const PointSet st = start(n, m, mixed(n), 0.5);
        Inverse inv(n, m);
        inv.set_coordinate(st.alpha, st.beta);
        const Dense h = v0121::explicit_h(st.y, n, m);
        const Dense wmat = v0121::build_w(st.y, n, m);
        Work w(n, m);
        const std::size_t s = 0;  // the origin is the point x_s, so A_sj = 0
        for (std::size_t j = 1; j < m; ++j) {
            for (const double a : {0.5, -1.5, 2.3, 0.25}) {
                std::vector<double> x(n), ysj(n);
                double nrm2 = 0.0;
                for (std::size_t k = 0; k < n; ++k) {
                    ysj[k] = st.y[j * n + k] - st.y[s * n + k];
                    nrm2 += ysj[k] * ysj[k];
                    x[k] = st.y[s * n + k] + a * ysj[k];
                }
                const double beta = inv.vlag_beta(st.y, s, x, w.vlag, w.wv, w.zw, w.d, w.sum);
                // e_j^T A Omega A e_j from the explicit blocks.
                double aoa = 0.0;
                for (std::size_t p = 0; p < m; ++p)
                    for (std::size_t q = 0; q < m; ++q) aoa += wmat(p, j) * h(p, q) * wmat(q, j);
                const double expected = a * a * (1.0 - a) * (1.0 - a) * (0.5 * nrm2 * nrm2 - aoa);
                EXPECT_NEAR(beta, expected, 1e-9 * (1.0 + std::fabs(expected)))
                    << "n " << n << " j " << j << " a " << a;
                EXPECT_LE(beta, 0.5 * a * a * (1.0 - a) * (1.0 - a) * nrm2 * nrm2 + 1e-9);
                EXPECT_GE(beta, -1e-9) << "(3.10): beta is not negative";
            }
        }
    }
}

TEST(V0121Inverse, TheShiftOfOriginIsTheInverseForTheShiftedPoints) {
    // (6.14) to (6.16), with the shift taken to a point of the set (the use of
    // section 6) and to a point that is none: Omega is untouched and the
    // result is the inverse of W for the offsets from the new origin.
    Draw rng(0xA11CE5ULL);
    for (std::size_t n = 1; n <= 4; ++n)
        for (const std::size_t m : {n + 2, 2 * n + 1, max_m(n)})
            for (int variant = 0; variant < 4; ++variant) {
                PointSet st = start(n, m, mixed(n), 0.5);
                Inverse inv(n, m);
                inv.set_coordinate(st.alpha, st.beta);
                Work w(n, m);
                // A few updates first, so that H is not the starting matrix.
                for (int step = 0; step < 6; ++step) {
                    const std::size_t s = static_cast<std::size_t>(step) % m;
                    const std::size_t t = (s + 1 + static_cast<std::size_t>(step)) % m;
                    if (t == s) continue;
                    std::vector<double> x(n);
                    for (std::size_t k = 0; k < n; ++k) x[k] = 0.9 * rng.next();
                    double sigma = 0.0;
                    (void)replace_point(inv, st.y, n, m, s, t, x, w, &sigma);
                }
                std::vector<double> shift(n);
                if (variant < 2) {
                    const std::size_t k = static_cast<std::size_t>(variant) * (m - 1) / 1;
                    for (std::size_t i = 0; i < n; ++i) shift[i] = st.y[k * n + i];
                } else {
                    for (std::size_t i = 0; i < n; ++i) shift[i] = 0.6 * rng.next();
                }
                std::vector<double> z_before;
                for (std::size_t i = 0; i < m; ++i)
                    for (std::size_t c = 0; c < inv.columns(); ++c) z_before.push_back(inv.z(i, c));
                // Omega is not touched by a shift, so its distance from the explicit
                // inverse is whatever the updates left; the shift must not add to it.
                const double omega_before =
                    v0121::compare_to_h(inv, v0121::explicit_h(st.y, n, m), n, m).omega;
                inv.shift(st.y, shift, w.gamma, w.cmat, w.zg, w.diff);
                std::size_t at = 0;
                for (std::size_t i = 0; i < m; ++i)
                    for (std::size_t c = 0; c < inv.columns(); ++c)
                        EXPECT_TRUE(v0121::same_bits(inv.z(i, c), z_before[at++]))
                            << "Omega must not change";
                for (std::size_t j = 0; j < m; ++j)
                    for (std::size_t i = 0; i < n; ++i) st.y[j * n + i] -= shift[i];
                const Dense h = v0121::explicit_h(st.y, n, m);
                ASSERT_EQ(h.size, m + n + 1);
                const v0121::HError e = v0121::compare_to_h(inv, h, n, m);
                EXPECT_LT(std::max(e.xi, e.upsilon), 1e-8)
                    << "n " << n << " m " << m << " variant " << variant << " (xi " << e.xi
                    << ", upsilon " << e.upsilon << ")";
                EXPECT_LE(e.omega, omega_before + 1e-9)
                    << "n " << n << " m " << m << " variant " << variant;
            }
}
