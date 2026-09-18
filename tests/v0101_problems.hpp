// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The test problems the 0.1.0.1 suite shares, with their optima derived
// rather than typed: every expected value below is a closed form in
// std::numbers or a rational, never a number copied from a run.
//
// Unconstrained: sphere, Rosenbrock (any n), Powell's singular function,
// Beale. Constrained: Powell's 1994 test set (problems 1 to 9 of the paper's
// numerical results; the hexagon is left out because its optimum has no
// closed form to derive here) and two Hock-Schittkowski problems with active
// inequality constraints, HS24 and HS35, both of which also carry x >= 0.

#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numbers>
#include <span>
#include <vector>

#include "flop/flop.hpp"

namespace v0101 {

// ---- unconstrained -------------------------------------------------------

// sum (x_i - c_i)^2 with c_i = 0.3 (i + 1): a centre that is nowhere near a
// coordinate axis, so no coordinate step lands on it by luck.
inline double sphere_centre(std::size_t i) {
    return 0.3 * static_cast<double>(i + 1);
}

inline double sphere(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double d = x[i] - sphere_centre(i);
        s += d * d;
    }
    return s;
}

// Rosenbrock's function in any dimension; the minimum is x = 1, f = 0.
inline double rosenbrock(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        const double a = 1.0 - x[i];
        const double b = x[i + 1] - x[i] * x[i];
        s += a * a + 100.0 * b * b;
    }
    return s;
}

// Powell's singular function: the Hessian at the minimum (the origin, f = 0)
// is singular, which is what makes it slow for every method.
inline double powell_singular(std::span<const double> x) {
    const double a = x[0] + 10.0 * x[1];
    const double b = x[2] - x[3];
    const double c = x[1] - 2.0 * x[2];
    const double d = x[0] - x[3];
    return a * a + 5.0 * b * b + c * c * c * c + 10.0 * d * d * d * d;
}

// Beale's function; the minimum is (3, 0.5), f = 0.
inline double beale(std::span<const double> x) {
    const double a = 1.5 - x[0] + x[0] * x[1];
    const double b = 2.25 - x[0] + x[0] * x[1] * x[1];
    const double c = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];
    return a * a + b * b + c * c;
}

// ---- Powell 1994, the paper's test problems --------------------------------

// A constrained problem: objective, constraints (feasible when >= 0), their
// count, the start the paper uses, the optimum and its objective value.
struct Constrained {
    const char* name;
    double (*f)(std::span<const double>);
    void (*c)(std::span<const double>, std::span<double>);
    std::size_t m;
    std::vector<double> x0;
    std::vector<double> x_opt;
    double f_opt;
};

inline void no_constraints(std::span<const double>, std::span<double>) {}

// 1. A simple quadratic: 10 (x1 + 1)^2 + x2^2.
inline double p1_f(std::span<const double> x) {
    return 10.0 * (x[0] + 1.0) * (x[0] + 1.0) + x[1] * x[1];
}

// 2. x1 x2 inside the unit circle: the minimum is -1/2 on the circle at
// x1 = -x2 = +-1/sqrt(2).
inline double p2_f(std::span<const double> x) {
    return x[0] * x[1];
}
inline void p2_c(std::span<const double> x, std::span<double> out) {
    out[0] = 1.0 - x[0] * x[0] - x[1] * x[1];
}

// 3. x1 x2 x3 inside the ellipsoid x1^2 + 2 x2^2 + 3 x3^2 <= 1. By the
// AM-GM inequality on x1^2, 2 x2^2, 3 x3^2 the product is at most
// 1 / sqrt(162) = sqrt(2) / 18 in magnitude, attained when the three are
// equal, which is x = (1/sqrt(3), 1/sqrt(6), -1/3).
inline double p3_f(std::span<const double> x) {
    return x[0] * x[1] * x[2];
}
inline void p3_c(std::span<const double> x, std::span<double> out) {
    out[0] = 1.0 - x[0] * x[0] - 2.0 * x[1] * x[1] - 3.0 * x[2] * x[2];
}

// 4. The weak Rosenbrock: (x1^2 - x2)^2 + (1 + x1)^2, minimum (-1, 1).
inline double p4_f(std::span<const double> x) {
    const double a = x[0] * x[0] - x[1];
    const double b = 1.0 + x[0];
    return a * a + b * b;
}

// 5. The intermediate Rosenbrock: 10 (x1^2 - x2)^2 + (1 + x1)^2.
inline double p5_f(std::span<const double> x) {
    const double a = x[0] * x[0] - x[1];
    const double b = 1.0 + x[0];
    return 10.0 * a * a + b * b;
}

// 6. Fletcher, equation (9.1.15): -x1 - x2 with x2 >= x1^2 and
// x1^2 + x2^2 <= 1; the minimum is (1, 1)/sqrt(2), f = -sqrt(2).
inline double p6_f(std::span<const double> x) {
    return -x[0] - x[1];
}
inline void p6_c(std::span<const double> x, std::span<double> out) {
    out[0] = x[1] - x[0] * x[0];
    out[1] = 1.0 - x[0] * x[0] - x[1] * x[1];
}

// 7. Fletcher, equation (14.4.2): minimise x3 under three constraints; the
// minimum is (0, -3, -3) with all three active, f = -3.
inline double p7_f(std::span<const double> x) {
    return x[2];
}
inline void p7_c(std::span<const double> x, std::span<double> out) {
    out[0] = 5.0 * x[0] - x[1] + x[2];
    out[1] = x[2] - x[0] * x[0] - x[1] * x[1] - 4.0 * x[1];
    out[2] = x[2] - 5.0 * x[0] - x[1];
}

// 8. Rosen-Suzuki (Hock-Schittkowski 43): minimum (0, 1, 2, -1), f = -44.
inline double p8_f(std::span<const double> x) {
    return x[0] * x[0] + x[1] * x[1] + 2.0 * x[2] * x[2] + x[3] * x[3] - 5.0 * x[0] - 5.0 * x[1] -
           21.0 * x[2] + 7.0 * x[3];
}
inline void p8_c(std::span<const double> x, std::span<double> out) {
    out[0] =
        8.0 - x[0] * x[0] - x[1] * x[1] - x[2] * x[2] - x[3] * x[3] - x[0] + x[1] - x[2] + x[3];
    out[1] = 10.0 - x[0] * x[0] - 2.0 * x[1] * x[1] - x[2] * x[2] - 2.0 * x[3] * x[3] + x[0] + x[3];
    out[2] = 5.0 - 2.0 * x[0] * x[0] - x[1] * x[1] - x[2] * x[2] - 2.0 * x[0] + x[1] + x[3];
}

// 9. Hock-Schittkowski 100, seven variables and four constraints. The
// optimum is known only numerically (to the seven digits the collection
// prints), so the check on it is looser than on the others.
inline double p9_f(std::span<const double> x) {
    const double a = x[0] - 10.0, b = x[1] - 12.0, d = x[3] - 11.0;
    return a * a + 5.0 * b * b + x[2] * x[2] * x[2] * x[2] + 3.0 * d * d +
           10.0 * x[4] * x[4] * x[4] * x[4] * x[4] * x[4] + 7.0 * x[5] * x[5] +
           x[6] * x[6] * x[6] * x[6] - 4.0 * x[5] * x[6] - 10.0 * x[5] - 8.0 * x[6];
}
inline void p9_c(std::span<const double> x, std::span<double> out) {
    out[0] = 127.0 - 2.0 * x[0] * x[0] - 3.0 * x[1] * x[1] * x[1] * x[1] - x[2] -
             4.0 * x[3] * x[3] - 5.0 * x[4];
    out[1] = 282.0 - 7.0 * x[0] - 3.0 * x[1] - 10.0 * x[2] * x[2] - x[3] + x[4];
    out[2] = 196.0 - 23.0 * x[0] - x[1] * x[1] - 6.0 * x[5] * x[5] + 8.0 * x[6];
    out[3] = -4.0 * x[0] * x[0] - x[1] * x[1] + 3.0 * x[0] * x[1] - 2.0 * x[2] * x[2] - 5.0 * x[5] +
             11.0 * x[6];
}

inline std::vector<Constrained> powell_1994_problems() {
    using std::numbers::inv_sqrt3;
    using std::numbers::sqrt2;
    const double inv_sqrt2 = 1.0 / sqrt2;
    const double inv_sqrt6 = inv_sqrt3 * inv_sqrt2;
    return {
        {.name = "1 simple quadratic",
         .f = p1_f,
         .c = no_constraints,
         .m = 0,
         .x0 = {1.0, 1.0},
         .x_opt = {-1.0, 0.0},
         .f_opt = 0.0},
        {.name = "2 unit circle",
         .f = p2_f,
         .c = p2_c,
         .m = 1,
         .x0 = {1.0, 1.0},
         .x_opt = {inv_sqrt2, -inv_sqrt2},
         .f_opt = -0.5},
        {.name = "3 ellipsoid",
         .f = p3_f,
         .c = p3_c,
         .m = 1,
         .x0 = {1.0, 1.0, 1.0},
         .x_opt = {inv_sqrt3, inv_sqrt6, -1.0 / 3.0},
         .f_opt = -sqrt2 / 18.0},
        {.name = "4 weak Rosenbrock",
         .f = p4_f,
         .c = no_constraints,
         .m = 0,
         .x0 = {1.0, 1.0},
         .x_opt = {-1.0, 1.0},
         .f_opt = 0.0},
        {.name = "5 intermediate Rosenbrock",
         .f = p5_f,
         .c = no_constraints,
         .m = 0,
         .x0 = {1.0, 1.0},
         .x_opt = {-1.0, 1.0},
         .f_opt = 0.0},
        {.name = "6 Fletcher 9.1.15",
         .f = p6_f,
         .c = p6_c,
         .m = 2,
         .x0 = {1.0, 1.0},
         .x_opt = {inv_sqrt2, inv_sqrt2},
         .f_opt = -sqrt2},
        {.name = "7 Fletcher 14.4.2",
         .f = p7_f,
         .c = p7_c,
         .m = 3,
         .x0 = {1.0, 1.0, 1.0},
         .x_opt = {0.0, -3.0, -3.0},
         .f_opt = -3.0},
        {.name = "8 Rosen-Suzuki",
         .f = p8_f,
         .c = p8_c,
         .m = 3,
         .x0 = {1.0, 1.0, 1.0, 1.0},
         .x_opt = {0.0, 1.0, 2.0, -1.0},
         .f_opt = -44.0},
        {.name = "9 Hock-Schittkowski 100",
         .f = p9_f,
         .c = p9_c,
         .m = 4,
         .x0 = {1.0, 2.0, 0.0, 4.0, 0.0, 1.0, 1.0},
         .x_opt = {2.330499, 1.951372, -0.4775414, 4.365726, -0.6244870, 1.038131, 1.594227},
         .f_opt = 680.6300573},
    };
}

// ---- Hock-Schittkowski problems with active inequalities and x >= 0 --------

// HS24: ((x1 - 3)^2 - 9) x2^3 / (27 sqrt(3)) on the triangle x1/sqrt(3) >= x2,
// x1 + sqrt(3) x2 >= 0, x1 + sqrt(3) x2 <= 6, x >= 0. The minimum is
// (3, sqrt(3)) with the first and third constraints active, f = -1.
inline double hs24_f(std::span<const double> x) {
    const double a = x[0] - 3.0;
    return (a * a - 9.0) * x[1] * x[1] * x[1] / (27.0 * std::numbers::sqrt3);
}
inline void hs24_c(std::span<const double> x, std::span<double> out) {
    out[0] = x[0] * std::numbers::inv_sqrt3 - x[1];
    out[1] = x[0] + std::numbers::sqrt3 * x[1];
    out[2] = 6.0 - x[0] - std::numbers::sqrt3 * x[1];
}

// HS35: a convex quadratic under x1 + x2 + 2 x3 <= 3 and x >= 0. The
// minimum is (4/3, 7/9, 4/9) on the constraint, f = 1/9.
inline double hs35_f(std::span<const double> x) {
    return 9.0 - 8.0 * x[0] - 6.0 * x[1] - 4.0 * x[2] + 2.0 * x[0] * x[0] + 2.0 * x[1] * x[1] +
           x[2] * x[2] + 2.0 * x[0] * x[1] + 2.0 * x[0] * x[2];
}
inline void hs35_c(std::span<const double> x, std::span<double> out) {
    out[0] = 3.0 - x[0] - x[1] - 2.0 * x[2];
}

// ---- helpers ---------------------------------------------------------------

// Non-finite values as bit patterns, written into memory the caller owns.
// Under -ffinite-math-only clang declares every double a function returns
// or takes by value free of NaN and infinity, so a helper returning one
// would hand back poison; a helper that writes the bits through a pointer
// leaves a NaN in memory, which is what the library's checks read (by
// reference, never by value). The bits pass through a volatile so the
// compiler cannot see what was written either.
constexpr std::uint64_t kQuietNaN = UINT64_C(0x7FF8000000000000);
constexpr std::uint64_t kInfinity = UINT64_C(0x7FF0000000000000);
constexpr std::uint64_t kNegInfinity = UINT64_C(0xFFF0000000000000);

inline void write_bits(double* dst, std::uint64_t bits) {
    volatile std::uint64_t opaque = bits;
    const std::uint64_t b = opaque;
    std::memcpy(dst, &b, sizeof(double));
}

inline flop::cobyla::Options options(double initial_step, double xtol_rel,
                                     std::size_t max_evaluations) {
    flop::cobyla::Options o;
    o.initial_step = initial_step;
    o.stopping.xtol_rel = xtol_rel;
    o.stopping.max_evaluations = max_evaluations;
    return o;
}

inline double max_abs_diff(std::span<const double> a, std::span<const double> b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// Bit equality, which is the only equality that means "the same trajectory"
// and the only one whose meaning -ffast-math cannot change.
inline bool same_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}
inline bool same_bits(std::span<const double> a, std::span<const double> b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!same_bits(a[i], b[i])) return false;
    return true;
}

}  // namespace v0101
