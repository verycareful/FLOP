// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// What the 0.1.1.1 Nelder-Mead suites share: the evaluation trace, the
// table objective, the exact-arithmetic check, the paper's trial points, and
// the test problems with their optima derived rather than typed.
//
// The table objective and the exact-arithmetic check are the backbone of the
// paper tests. With n a power of two (1, 2, 4, 8), x0 and initial_step on a
// coarse dyadic grid, every coefficient of both sets is dyadic (standard 1,
// 2, 1/2, 1/2; Gao and Han at n = 4: 1, 3/2, 5/8, 3/4; at n = 8: 1, 5/4,
// 11/16, 7/8) and 1/n is exact, so every trial point is computed without
// rounding for as long as its significand holds it. A test can then choose
// f at each point it expects, force any branch, any tie, any equality, and
// pin the points the method evaluates to the bit in both binaries: exact
// arithmetic is the one arithmetic reassociation cannot change. Each such
// test asserts that its points stayed on the grid, so it fails loudly if it
// leaves the exact regime instead of comparing rounded values.
//
// The problems: Gao and Han's problem (4.3), and the More, Garbow and
// Hillstrom functions of Gao and Han's Table 2 that have a closed-form
// minimiser, each at the starting point of More et al.

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

#include "flop/flop.hpp"
#include "v0101_problems.hpp"

namespace v0111 {

using v0101::kInfinity;
using v0101::kNegInfinity;
using v0101::kQuietNaN;
using v0101::max_abs_diff;
using v0101::same_bits;
using v0101::write_bits;

// ---- the evaluation trace ---------------------------------------------------

struct Point {
    std::size_t index;
    std::vector<double> x;
    double f;
};
using Trace = std::vector<Point>;

// An on_evaluation hook that copies every evaluation into t.
inline std::function<void(const flop::Evaluation&)> recorder(Trace& t) {
    return [&t](const flop::Evaluation& e) {
        t.push_back({.index = e.index, .x = {e.x.begin(), e.x.end()}, .f = e.f});
    };
}

inline flop::nelder_mead::Options options(double initial_step, std::size_t max_evaluations,
                                          bool adaptive = true) {
    flop::nelder_mead::Options o;
    o.initial_step = initial_step;
    o.stopping.max_evaluations = max_evaluations;
    o.adaptive_coefficients = adaptive;
    return o;
}

// A run and its trace.
struct Traced {
    flop::Result r;
    Trace trace;
};

template <class F>
Traced traced(F&& f, std::span<const double> x0, flop::nelder_mead::Options o) {
    Traced t;
    o.on_evaluation = recorder(t.trace);
    t.r = flop::nelder_mead::minimize(f, x0, o);
    return t;
}

template <class F>
Traced traced_batch(F&& f, std::span<const double> x0, flop::nelder_mead::Options o) {
    Traced t;
    o.on_evaluation = recorder(t.trace);
    t.r = flop::nelder_mead::minimize_batch(f, x0, o);
    return t;
}

// ---- the table objective ----------------------------------------------------

// The value a table returns for a point it does not hold. Finite, and far
// above every value a scripted test assigns, so a point the test did not
// expect ranks worst; the test also asserts misses() == 0.
constexpr double kMiss = 1.0e6;

// An objective given as a list of exact points and their values, looked up
// by bit equality.
class Table {
public:
    Table& set(std::vector<double> x, double f) {
        rows_.push_back({.x = std::move(x), .f = f});
        return *this;
    }
    double operator()(std::span<const double> x) {
        for (const Row& r : rows_)
            if (same_bits(r.x, x)) return r.f;
        ++misses_;
        return kMiss;
    }
    [[nodiscard]] std::size_t misses() const { return misses_; }

private:
    struct Row {
        std::vector<double> x;
        double f;
    };
    std::vector<Row> rows_;
    std::size_t misses_ = 0;
};

// ---- exact arithmetic -------------------------------------------------------

// x = m 2^-frac with m an integer and |x| < 2^integer, both as small as the
// value allows. Read from the bits, so no floating-point operation is
// involved in deciding whether floating-point operations were exact.
struct GridBits {
    int frac = 0;
    int integer = 0;
};

inline GridBits grid_bits(const double& x) {
    constexpr unsigned kFractionBits = 52U;
    constexpr std::uint64_t kExponentMask = 0x7FFU;
    constexpr std::uint64_t kHiddenBit = std::uint64_t{1} << kFractionBits;
    const auto bits = std::bit_cast<std::uint64_t>(x);
    const auto biased = static_cast<int>((bits >> kFractionBits) & kExponentMask);
    std::uint64_t m = bits & (kHiddenBit - 1U);
    if (biased == 0) {
        if (m == 0) return {};
        return {.frac = 1074 - std::countr_zero(m), .integer = 0};
    }
    m |= kHiddenBit;
    // x = m 2^(biased - 1075), m < 2^53.
    const int tz = std::countr_zero(m);
    return {.frac = std::max(0, 1075 - biased - tz), .integer = std::max(0, biased - 1022)};
}

// The budget a point may use: with every coordinate on a grid of 2^-F and
// below 2^I in magnitude, F + I <= 40, the centroid (a sum of at most 9 such
// numbers, then a division by n <= 8) and a trial point (a product with a
// coefficient of at most 4 fractional and 2 integer bits, then a sum) stay
// within the 53-bit significand: 40 + 4 + 3 + 4 + 2 = 53.
constexpr int kExactBudget = 40;

// The length of the longest prefix of t whose points, taken together, fit
// the budget: past it the method's arithmetic may round.
inline std::size_t exact_prefix(const Trace& t, int budget = kExactBudget) {
    int frac = 0;
    int integer = 0;
    for (std::size_t k = 0; k < t.size(); ++k) {
        for (const double& v : t[k].x) {
            const GridBits g = grid_bits(v);
            frac = std::max(frac, g.frac);
            integer = std::max(integer, g.integer);
        }
        if (frac + integer > budget) return k;
    }
    return t.size();
}

inline bool all_exact(const Trace& t, int budget = kExactBudget) {
    return exact_prefix(t, budget) == t.size();
}

// ---- the paper's points -----------------------------------------------------

using Simplex = std::vector<std::vector<double>>;  // ordered, x_1 first

// Lagarias et al., (2.12): z(tau) = xbar + tau (xbar - x_{n+1}), with xbar the
// centroid of x_1 .. x_n. Exact for the dyadic simplices the tests build.
inline std::vector<double> trial_point(const Simplex& ranked, double tau) {
    const std::size_t n = ranked.size() - 1;
    std::vector<double> z(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (std::size_t k = 0; k < n; ++k) s += ranked[k][i];
        const double xbar = s / static_cast<double>(n);
        z[i] = xbar + tau * (xbar - ranked[n][i]);
    }
    return z;
}

// Lagarias et al., section 2, step 5: v_i = x_1 + sigma (x_i - x_1).
inline std::vector<double> shrink_point(const std::vector<double>& x1,
                                        const std::vector<double>& xi, double sigma) {
    std::vector<double> v(x1.size());
    for (std::size_t i = 0; i < x1.size(); ++i) v[i] = x1[i] + sigma * (xi[i] - x1[i]);
    return v;
}

// The coefficients as the papers state them: Lagarias et al. (2.2), and Gao
// and Han (4.1) for n >= 2. Written from the formulas, not from FLOP.
struct Coefficients {
    double rho;
    double chi;
    double gamma;
    double sigma;
};

inline Coefficients standard() {
    return {.rho = 1.0, .chi = 2.0, .gamma = 0.5, .sigma = 0.5};
}

inline Coefficients gao_han(std::size_t n) {
    const auto dn = static_cast<double>(n);
    return {.rho = 1.0,
            .chi = 1.0 + 2.0 / dn,
            .gamma = 0.75 - 1.0 / (2.0 * dn),
            .sigma = 1.0 - 1.0 / dn};
}

// What FLOP documents: Gao and Han's set when adaptive and n >= 2, the
// standard set otherwise.
inline Coefficients documented(std::size_t n, bool adaptive) {
    return adaptive && n >= 2 ? gao_han(n) : standard();
}

// x0 and x0 + h e_i: the documented initial simplex without a box.
inline Simplex axis_simplex(const std::vector<double>& x0, double h) {
    Simplex s{x0};
    for (std::size_t i = 0; i < x0.size(); ++i) {
        s.push_back(x0);
        s.back()[i] += h;
    }
    return s;
}

inline std::vector<double> unit(std::size_t n, std::size_t i, double scale = 1.0) {
    std::vector<double> e(n, 0.0);
    e[i] = scale;
    return e;
}

// ---- the batch channel ------------------------------------------------------

// Wraps a scalar objective as a batch objective and records the size of
// every call.
template <class F>
struct Batched {
    F f;
    std::vector<std::size_t> sizes;
    void operator()(std::span<const std::span<const double>> xs, std::span<double> out) {
        sizes.push_back(xs.size());
        for (std::size_t i = 0; i < xs.size(); ++i) out[i] = f(xs[i]);
    }
};

template <class F>
Batched<F> batched(F f) {
    return Batched<F>{.f = std::move(f), .sizes = {}};
}

// ---- problems ---------------------------------------------------------------

using v0101::beale;
using v0101::powell_singular;
using v0101::sphere;
using v0101::sphere_centre;

// Gao and Han, problem (4.3): x'Dx + sigma (x'Bx)^2 with D = diag((1 + eps)^i),
// i = 1 .. n, and B = U'U for U the upper triangular matrix of ones, so
// x'Bx = |Ux|^2 with (Ux)_i the sum of x_j for j >= i. The minimiser is the
// origin, f = 0.
struct GaoHan43 {
    double eps;
    double sigma;
    double operator()(std::span<const double> x) const {
        double q = 0.0;
        double d = 1.0;
        for (const double v : x) {
            d *= 1.0 + eps;
            q += d * v * v;
        }
        double suffix = 0.0;
        double b = 0.0;
        for (std::size_t i = x.size(); i-- > 0;) {
            suffix += x[i];
            b += suffix * suffix;
        }
        return q + sigma * b * b;
    }
};

// More, Garbow and Hillstrom, problem (7): the helical valley. theta is
// defined for x1 > 0 and x1 < 0; on x1 = 0 it takes its limit from x1 > 0,
// +-1/4, and 0 at the origin, which the paper leaves undefined.
inline double helical_valley(std::span<const double> x) {
    using std::numbers::pi;
    double theta = 0.0;
    if (x[0] > 0.0)
        theta = std::atan(x[1] / x[0]) / (2.0 * pi);
    else if (x[0] < 0.0)
        theta = std::atan(x[1] / x[0]) / (2.0 * pi) + 0.5;
    else if (x[1] > 0.0)
        theta = 0.25;
    else if (x[1] < 0.0)
        theta = -0.25;
    const double f1 = 10.0 * (x[2] - 10.0 * theta);
    const double f2 = 10.0 * (std::sqrt(x[0] * x[0] + x[1] * x[1]) - 1.0);
    return f1 * f1 + f2 * f2 + x[2] * x[2];
}

// More et al., problem (21): the extended Rosenbrock function, n even, in
// independent pairs. The minimum is 0 at (1, ..., 1).
inline double extended_rosenbrock(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i + 1 < x.size(); i += 2) {
        const double a = 10.0 * (x[i + 1] - x[i] * x[i]);
        const double b = 1.0 - x[i];
        s += a * a + b * b;
    }
    return s;
}

// More et al., problem (14): Wood's function. The minimum is 0 at (1, 1, 1, 1).
inline double wood(std::span<const double> x) {
    const double f1 = 10.0 * (x[1] - x[0] * x[0]);
    const double f2 = 1.0 - x[0];
    const double f3 = std::sqrt(90.0) * (x[3] - x[2] * x[2]);
    const double f4 = 1.0 - x[2];
    const double f5 = std::sqrt(10.0) * (x[1] + x[3] - 2.0);
    const double f6 = (x[1] - x[3]) / std::sqrt(10.0);
    return f1 * f1 + f2 * f2 + f3 * f3 + f4 * f4 + f5 * f5 + f6 * f6;
}

// More et al., problem (25): the variably dimensioned function. The minimum
// is 0 at (1, ..., 1).
inline double variably_dimensioned(std::span<const double> x) {
    double s = 0.0;
    double sum = 0.0;
    for (std::size_t j = 0; j < x.size(); ++j) {
        const double d = x[j] - 1.0;
        s += d * d;
        sum += static_cast<double>(j + 1) * d;
    }
    return s + sum * sum + sum * sum * sum * sum;
}

// More et al., problem (35): Chebyquad with m = n. T_i is the Chebyshev
// polynomial shifted to [0, 1]; its integral over [0, 1] is 0 for odd i and
// -1/(i^2 - 1) for even i.
inline double chebyquad(std::span<const double> x) {
    const std::size_t n = x.size();
    double s = 0.0;
    for (std::size_t i = 1; i <= n; ++i) {
        double mean = 0.0;
        for (const double v : x) {
            const double y = 2.0 * v - 1.0;
            double t0 = 1.0;
            double t1 = y;
            for (std::size_t k = 1; k < i; ++k) {
                const double t2 = 2.0 * y * t1 - t0;
                t0 = t1;
                t1 = t2;
            }
            mean += t1;
        }
        mean /= static_cast<double>(n);
        const auto di = static_cast<double>(i);
        const double integral = i % 2 == 1 ? 0.0 : -1.0 / (di * di - 1.0);
        const double fi = mean - integral;
        s += fi * fi;
    }
    return s;
}

// A More, Garbow and Hillstrom problem as Gao and Han's Table 2 runs it: the
// paper's start, the closed-form minimiser (empty when it is only known up to
// a permutation of the coordinates, which sorted_minimiser then gives), and
// the final f Table 2 reports for each method at TolX = TolFun = 1e-4.
struct Mgh {
    const char* name;
    double (*f)(std::span<const double>);
    std::vector<double> x0;
    std::vector<double> x_opt;
    bool permutation_invariant;
    double table2_anms;
    double table2_snms;
};

inline std::vector<Mgh> mgh_problems() {
    const double inv_sqrt3 = std::numbers::inv_sqrt3;
    std::vector<Mgh> out;
    out.push_back({.name = "helical valley",
                   .f = helical_valley,
                   .x0 = {-1.0, 0.0, 0.0},
                   .x_opt = {1.0, 0.0, 0.0},
                   .permutation_invariant = false,
                   .table2_anms = 2.6665e-4,
                   .table2_snms = 3.5759e-4});
    out.push_back({.name = "extended rosenbrock 2",
                   .f = extended_rosenbrock,
                   .x0 = {-1.2, 1.0},
                   .x_opt = {1.0, 1.0},
                   .permutation_invariant = false,
                   .table2_anms = 8.1777e-10,
                   .table2_snms = 8.1777e-10});
    out.push_back({.name = "extended rosenbrock 4",
                   .f = extended_rosenbrock,
                   .x0 = {-1.2, 1.0, -1.2, 1.0},
                   .x_opt = {1.0, 1.0, 1.0, 1.0},
                   .permutation_invariant = false,
                   .table2_anms = 7.3907e-10,
                   .table2_snms = 2.2923e-10});
    out.push_back({.name = "powell singular",
                   .f = powell_singular,
                   .x0 = {3.0, -1.0, 0.0, 1.0},
                   .x_opt = {0.0, 0.0, 0.0, 0.0},
                   .permutation_invariant = false,
                   .table2_anms = 1.7814e-7,
                   .table2_snms = 1.3906e-6});
    out.push_back({.name = "beale",
                   .f = beale,
                   .x0 = {1.0, 1.0},
                   .x_opt = {3.0, 0.5},
                   .permutation_invariant = false,
                   .table2_anms = 1.3926e-10,
                   .table2_snms = 1.3926e-10});
    out.push_back({.name = "wood",
                   .f = wood,
                   .x0 = {-3.0, -1.0, -3.0, -1.0},
                   .x_opt = {1.0, 1.0, 1.0, 1.0},
                   .permutation_invariant = false,
                   .table2_anms = 9.1293e-9,
                   .table2_snms = 1.9448e-9});
    for (const std::size_t n : {std::size_t{4}, std::size_t{6}}) {
        std::vector<double> x0(n);
        for (std::size_t j = 0; j < n; ++j)
            x0[j] = 1.0 - static_cast<double>(j + 1) / static_cast<double>(n);
        out.push_back({.name = n == 4 ? "variably dimensioned 4" : "variably dimensioned 6",
                       .f = variably_dimensioned,
                       .x0 = x0,
                       .x_opt = std::vector<double>(n, 1.0),
                       .permutation_invariant = false,
                       .table2_anms = n == 4 ? 5.0684e-9 : 5.9536e-9,
                       .table2_snms = n == 4 ? 1.1926e-8 : 5.3381e-9});
    }
    // At n = 2 the minimiser is the two-point Gauss-Legendre rule on [0, 1],
    // (1 -+ 1/sqrt(3)) / 2, the one equal-weight rule exact for T_1 and T_2.
    out.push_back({.name = "chebyquad 2",
                   .f = chebyquad,
                   .x0 = {1.0 / 3.0, 2.0 / 3.0},
                   .x_opt = {0.5 * (1.0 - inv_sqrt3), 0.5 * (1.0 + inv_sqrt3)},
                   .permutation_invariant = true,
                   .table2_anms = 1.4277e-8,
                   .table2_snms = 1.4277e-8});
    return out;
}

}  // namespace v0111
