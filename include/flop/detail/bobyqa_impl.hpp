// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// bobyqa_impl - Powell's BOBYQA, from the 2009 report
// =============================================================================
//
// M. J. D. Powell, "The BOBYQA algorithm for bound constrained optimization
// without derivatives", report DAMTP 2009/NA06, University of Cambridge,
// 2009. Section and equation numbers below are the report's.
//
// The pieces:
//   bobyqa_model.hpp   the quadratic model (4.6)-(4.7), the inverse KKT
//                      matrix H as Z and B (section 4), its update (4.9)-
//                      (4.14) and the shift of origin (6.14)-(6.17)
//   bobyqa_trsbox.hpp  the trust-region step (section 3, first half)
//   bobyqa_altmov.hpp  the step of an alternative iteration (section 3,
//                      second half)
//   bobyqa_rescue.hpp  the choice of points in RESCUE (section 5)
// and this file: the starting procedure (section 2), the iterations and the
// control of rho and Delta (section 6), and the evaluations.
//
// Every point is an offset from the origin x0, and every evaluated point is
// formed from its offset with any coordinate on a bound set to the bound's
// own value, so that a step onto a bound is exactly on it. Nothing here uses
// infinity or NaN; an objective value is checked by its bits as it arrives.
// The solver is a template on the objective so that an evaluation inlines;
// it is instantiated in the caller's translation unit under the caller's
// floating-point model, which none of the arithmetic may depend on.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#ifdef FLOP_BOBYQA_TRACE
#include <cstdio>
#define FLOP_BOBYQA_LOG(...) std::fprintf(stderr, __VA_ARGS__)
#else
#define FLOP_BOBYQA_LOG(...) ((void)0)
#endif

#include "flop/detail/bobyqa_altmov.hpp"
#include "flop/detail/bobyqa_model.hpp"
#include "flop/detail/bobyqa_rescue.hpp"
#include "flop/detail/bobyqa_trsbox.hpp"
#include "flop/detail/evaluator.hpp"
#include "flop/detail/fp.hpp"
#include "flop/detail/tolerance.hpp"
#include "flop/options.hpp"
#include "flop/result.hpp"

namespace flop::bobyqa {

// The BOBYQA-specific settings on top of the shared ones.
struct Options : flop::Options {
    // Powell's rhoend, the final lower bound on the trust-region radius.
    // 0 derives it from the x tolerances: the larger of xtol_abs and
    // xtol_rel * initial_step.
    double final_trust_radius = 0.0;
    // m, the number of interpolation conditions, from n + 2 to
    // (n + 1)(n + 2) / 2. 0 is the report's recommendation, 2n + 1.
    std::size_t interpolation_points = 0;
    // The report requires b_i - a_i >= 2 initial_step on every coordinate
    // bounded on both sides (2.1); a narrower box is invalid input. With
    // this on, the first trust-region radius is instead the largest that
    // fits, min(initial_step, min_i (b_i - a_i) / 2).
    bool shrink_initial_step = false;
    // Which interpolation point a trust-region step replaces: the t that
    // maximises w_t sigma_t, sigma_t the denominator of the update of H. On,
    // w_t is NEWUOA's (Powell 2006, equation (7.5)), max[1, (|y_t - x*| /
    // max[0.1 Delta, rho])^6], with x* the best point after the step. Off,
    // it is the BOBYQA report's (6.1), max[1, |y_t - x_k|^2 / Delta^2].
    // NEWUOA's drops far points sooner and needs fewer evaluations: about a
    // quarter fewer than the report's weight on the report's own section 7
    // problem.
    bool newuoa_weight = true;
};

}  // namespace flop::bobyqa

namespace flop::detail::bobyqa {

// Section 6, the report's constants.
//   (6.4)   the ratio thresholds, and the factors on Delta and |d|
//   (6.4)   a Delta within kSnap of rho is rho, also after (6.5)
//   (6.5)   a short step is one under kShort rho
//   (6.6)   the schedule of rho: rhoend below kRhoNear rhoend, the
//           geometric mean up to kRhoFar rhoend, kRhoCut rho beyond
//   (6.7)   interpolation points within kFar rho are close
//   (6.12)  the alternative model, kAltModelRatio, kAltModelRuns in a row
//   (6.13)  the origin moves when |d|^2 <= kShiftRatio |x_k - x0|^2
//   (5.2)   sigma <= kSigmaRatio tau^2 calls RESCUE
//   (6.1)   after a successful step the second t is kept when its sigma
//           exceeds kSigmaRatio tau^2; the weight itself is (6.1) or
//           NEWUOA's (7.5), Options::newuoa_weight
constexpr double kRatioLow = 0.1;
constexpr double kRatioHigh = 0.7;
constexpr double kSnap = 1.5;
constexpr double kShort = 0.5;
constexpr double kShortCut = 0.1;
constexpr double kRhoNear = 16.0;
constexpr double kRhoFar = 250.0;
constexpr double kRhoCut = 0.1;
constexpr double kFar = 10.0;
constexpr double kFarDelta = 2.0;
constexpr double kAltDelta = 0.1;
constexpr double kAltModelRatio = 0.1;
constexpr std::size_t kAltModelRuns = 3;
constexpr double kShiftRatio = 1e-3;
constexpr double kSigmaRatio = 0.5;
// NEWUOA (7.5): the distance scale max[kWeightDelta Delta, rho].
constexpr double kWeightDelta = 0.1;
// (6.10): (1/8) rho^2 |s|^-2 s^T (nabla^2 Q) s.
constexpr double kCurvatureFactor = 0.125;
// (6.8): the three most recent values.
constexpr std::size_t kErrorHistory = 3;

template <class Eval>
class Solver {
public:
    Solver(Eval eval, std::span<const double> x0, const flop::bobyqa::Options& opts)
        : eval_(eval),
          n_(x0.size()),
          m_(opts.interpolation_points > 0 ? opts.interpolation_points : 2 * x0.size() + 1),
          opts_(opts),
          bounds_(opts.bounds ? &*opts.bounds : nullptr),
          model_(n_, m_),
          inv_(n_, m_),
          box_(n_),
          tw_(n_),
          aw_(n_, m_) {
        rhobeg_ = opts.initial_step;
        if (opts.shrink_initial_step && bounds_) {
            for (std::size_t i = 0; i < n_; ++i) {
                const auto& lo = bounds_->lower[i];
                const auto& hi = bounds_->upper[i];
                if (lo && hi) rhobeg_ = std::min(rhobeg_, 0.5 * *hi - 0.5 * *lo);
            }
        }
        rho_ = rhobeg_;
        delta_ = rhobeg_;
        rhoend_ =
            opts.final_trust_radius > 0.0
                ? opts.final_trust_radius
                : std::max(opts.stopping.xtol_abs, opts.stopping.xtol_rel * opts.initial_step);
        // As in COBYLA: below about machine precision of the step scale the
        // points cannot be told apart, so rho has a floor even when no x
        // tolerance was asked for; reaching the floor is RoundoffLimited,
        // reaching the caller's own rhoend is XtolReached.
        floor_ = rhobeg_ * std::numeric_limits<double>::epsilon();
        rhoend_from_caller_ = rhoend_ > floor_;
        rhoend_ = std::max(rhoend_, floor_);
        if (rhoend_ > rho_) rhoend_ = rho_;

        xbase_.assign(x0.begin(), x0.end());
        // The bounds as plain values beside the box's flags: the box itself
        // holds them as offsets from the origin, which moves.
        abs_lo_.assign(n_, 0.0);
        abs_hi_.assign(n_, 0.0);
        if (bounds_) {
            for (std::size_t i = 0; i < n_; ++i) {
                const std::optional<double>& lo = bounds_->lower[i];
                const std::optional<double>& hi = bounds_->upper[i];
                if (lo.has_value()) {
                    box_.has_lo[i] = 1;
                    abs_lo_[i] = *lo;
                }
                if (hi.has_value()) {
                    box_.has_hi[i] = 1;
                    abs_hi_[i] = *hi;
                }
            }
        }
        const std::size_t nz = m_ - n_ - 1;
        // Every scratch vector of the solver is cut from one buffer.
        const std::size_t sizes[] = {m_ + n_, m_,      nz, n_, n_, m_ + n_, m_ + n_,
                                     n_ * m_, m_ * n_, n_, n_, n_, n_,      n_,
                                     n_,      n_,      m_, n_, n_};
        std::size_t total = 0;
        for (const std::size_t k : sizes) total += k;
        scratch_.assign(total, 0.0);
        std::size_t at = 0;
        const auto carve = [&](std::size_t k) {
            const std::span<double> v(scratch_.data() + at, k);
            at += k;
            return v;
        };
        vlag_ = carve(m_ + n_);
        wv_ = carve(m_);
        zw_ = carve(nz);
        dvec_ = carve(n_);
        sum_ = carve(n_);
        hcol_ = carve(m_ + n_);
        ucol_ = carve(m_ + n_);
        gamma_ = carve(n_ * m_);
        cmat_ = carve(m_ * n_);
        work_ = carve(n_);
        work2_ = carve(n_);
        d_ = carve(n_);
        c_ = carve(n_);
        gd_ = carve(n_);
        xnew_ = carve(n_);
        x_ = carve(n_);
        lam_ = carve(m_);
        shift_ = carve(n_);
        xold_ = carve(n_);
        sides_.assign(2 * n_, Side::Free);
        side_ = std::span<Side>(sides_.data(), n_);
        cside_ = std::span<Side>(sides_.data() + n_, n_);
    }

    // The scratch spans point into this object's own buffers.
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;
    Solver(Solver&&) = delete;
    Solver& operator=(Solver&&) = delete;
    ~Solver() = default;

    Result run() {
        initialize();
        Next next = Next::Trust;
        while (!done_) {
            switch (next) {
                case Next::Trust:
                    next = trust_iteration();
                    break;
                case Next::Alternative:
                    next = alternative_iteration();
                    break;
                case Next::ReduceRho:
                    next = reduce_rho();
                    break;
            }
        }
        return finish();
    }

private:
    enum class Next : std::uint8_t { Trust, Alternative, ReduceRho };

    // ============================================================ evaluation
    bool cap_reached() {
        if (opts_.stopping.max_evaluations > 0 && evals_ >= opts_.stopping.max_evaluations) {
            stop(Status::MaxEvaluationsReached);
            return true;
        }
        return false;
    }

    [[nodiscard]] std::size_t remaining() const {
        if (opts_.stopping.max_evaluations == 0) return std::numeric_limits<std::size_t>::max();
        return opts_.stopping.max_evaluations - evals_;
    }

    void stop(Status s) {
        if (!done_) {
            done_ = true;
            status_ = s;
        }
    }

    // The point the objective sees for an offset: the origin plus the
    // offset, a coordinate on a bound of the box being the bound itself.
    void actual(std::span<const double> off, std::span<double> x) const {
        for (std::size_t i = 0; i < n_; ++i) {
            if (box_.has_lo[i] && off[i] <= box_.lo[i]) {
                x[i] = abs_lo_[i];
                continue;
            }
            if (box_.has_hi[i] && off[i] >= box_.hi[i]) {
                x[i] = abs_hi_[i];
                continue;
            }
            const double b = xbase_[i], d = off[i];
            if ((b >= 0.0) == (d >= 0.0) && std::fabs(d) > kMaxFinite - std::fabs(b))
                throw std::runtime_error(
                    "flop::bobyqa: a trial point has a coordinate beyond the largest double; the "
                    "objective may be unbounded below");
            x[i] = b + d;
            if (box_.has_lo[i] && x[i] < abs_lo_[i]) x[i] = abs_lo_[i];
            if (box_.has_hi[i] && x[i] > abs_hi_[i]) x[i] = abs_hi_[i];
        }
    }

    void record(std::span<const double> x, double f) {
        if (fp_bad(f))
            throw std::runtime_error("flop::bobyqa: the objective returned a non-finite value");
        ++evals_;
        if (opts_.on_evaluation) {
            Evaluation ev{.index = evals_ - 1, .x = x, .f = f, .constraints = {}};
            opts_.on_evaluation(ev);
        }
        if (!have_best_ || f < best_f_) {
            have_best_ = true;
            best_f_ = f;
            best_x_.assign(x.begin(), x.end());
        }
        if (opts_.stopping.stop_value && f <= *opts_.stopping.stop_value)
            stop(Status::StopValueReached);
    }

    // One evaluation at an offset. False when the cap forbids it.
    bool evaluate(std::span<const double> off, double& f) {
        if (cap_reached()) return false;
        actual(off, x_);
        f = eval_(x_);
        record(x_, f);
        return true;
    }

    // The evaluations of the offsets in rows[first..last) of pts, into
    // out[first..last), as one batch when the objective takes one and the
    // cap allows them all. False when the cap stopped them.
    bool evaluate_rows(std::span<const double> pts, std::size_t first, std::size_t last,
                       std::span<double> out) {
        const std::size_t count = last - first;
        if (count == 0) return true;
        if (Eval::has_batch && remaining() >= count) {
            std::vector<double> xs(count * n_);
            std::vector<std::span<const double>> spans(count);
            for (std::size_t r = 0; r < count; ++r) {
                actual(pts.subspan((first + r) * n_, n_),
                       std::span<double>(xs.data() + r * n_, n_));
                spans[r] = std::span<const double>(xs.data() + r * n_, n_);
            }
            std::vector<double> vals(count);
            eval_.batch(spans, vals);
            for (std::size_t r = 0; r < count; ++r) {
                record(spans[r], vals[r]);
                out[first + r] = vals[r];
            }
            return true;
        }
        for (std::size_t r = first; r < last; ++r) {
            if (!evaluate(pts.subspan(r * n_, n_), out[r])) return false;
        }
        return true;
    }

    // ===================================================== section 2: start
    void initialize() {
        // x0 moves to a bound, or to Delta inside it, when it is closer to
        // the bound than Delta; the box becomes offsets from it.
        for (std::size_t i = 0; i < n_; ++i) {
            const double lo = abs_lo_[i], hi = abs_hi_[i];
            if (box_.has_lo[i] && xbase_[i] > lo && xbase_[i] < lo + rhobeg_)
                xbase_[i] = lo + rhobeg_;
            if (box_.has_hi[i] && xbase_[i] < hi && xbase_[i] > hi - rhobeg_)
                xbase_[i] = hi - rhobeg_;
        }
        for (std::size_t i = 0; i < n_; ++i) {
            if (box_.has_lo[i]) box_.lo[i] = abs_lo_[i] - xbase_[i];
            if (box_.has_hi[i]) box_.hi[i] = abs_hi_[i] - xbase_[i];
        }
        // (2.2): alpha_i and beta_i for each coordinate.
        std::vector<double> alpha(n_), beta(n_);
        std::vector<char> interior(n_, 0);
        for (std::size_t i = 0; i < n_; ++i) {
            if (box_.has_lo[i] && box_.lo[i] >= 0.0) {
                alpha[i] = rhobeg_;
                beta[i] = 2.0 * rhobeg_;
                if (box_.has_hi[i] && beta[i] > box_.hi[i]) beta[i] = box_.hi[i];
            } else if (box_.has_hi[i] && box_.hi[i] <= 0.0) {
                alpha[i] = -rhobeg_;
                beta[i] = -2.0 * rhobeg_;
                if (box_.has_lo[i] && beta[i] < box_.lo[i]) beta[i] = box_.lo[i];
            } else {
                alpha[i] = rhobeg_;
                beta[i] = -rhobeg_;
                interior[i] = 1;
            }
        }
        std::vector<double>& pts = model_.points;
        for (std::size_t i = 0; i < n_; ++i) {
            pts[(i + 1) * n_ + i] = alpha[i];
            if (n_ + i + 1 < m_) pts[(n_ + i + 1) * n_ + i] = beta[i];
        }
        const std::size_t first = std::min(m_, 2 * n_ + 1);
        if (!evaluate_rows(pts, 0, first, model_.fval) || done_) return;
        if (m_ > 2 * n_ + 1) {
            // The bias towards lower values: swap the pair where the second
            // point is the lower one, on interior coordinates.
            for (std::size_t i = 0; i < n_; ++i) {
                if (!interior[i] || !(model_.fval[n_ + i + 1] < model_.fval[i + 1])) continue;
                std::swap(model_.fval[n_ + i + 1], model_.fval[i + 1]);
                std::swap(pts[(n_ + i + 1) * n_ + i], pts[(i + 1) * n_ + i]);
                std::swap(alpha[i], beta[i]);
            }
            for (std::size_t j = 2 * n_ + 1; j < m_; ++j) {
                const Pair pq = extra_pair(n_, j - 2 * n_ - 1);
                pts[j * n_ + pq.p] = alpha[pq.p];
                pts[j * n_ + pq.q] = alpha[pq.q];
            }
            if (!evaluate_rows(pts, first, m_, model_.fval) || done_) return;
        }
        // The first model (2.5): gradient and diagonal from the coordinate
        // points, off-diagonal pairs from the extra points, the rest zero.
        const double f0 = model_.fval[0];
        const std::size_t both = std::min(n_, m_ - n_ - 1);
        std::vector<double>& hq = model_.hq;
        for (std::size_t i = 0; i < n_; ++i) {
            const double a = alpha[i];
            const double fa = (model_.fval[i + 1] - f0) / a;
            if (i < both) {
                const double b = beta[i];
                const double fb = (model_.fval[n_ + i + 1] - f0) / b;
                const double hii = 2.0 * (fa - fb) / (a - b);
                hq[i * n_ + i] = hii;
                model_.g[i] = fa - 0.5 * hii * a;
            } else {
                model_.g[i] = fa;
            }
        }
        for (std::size_t j = 2 * n_ + 1; j < m_; ++j) {
            const Pair pq = extra_pair(n_, j - 2 * n_ - 1);
            const double ap = alpha[pq.p], aq = alpha[pq.q];
            const double rest = model_.fval[j] - f0 - model_.g[pq.p] * ap - model_.g[pq.q] * aq -
                                0.5 * hq[pq.p * n_ + pq.p] * ap * ap -
                                0.5 * hq[pq.q * n_ + pq.q] * aq * aq;
            const double hpq = rest / (ap * aq);
            hq[pq.p * n_ + pq.q] = hpq;
            hq[pq.q * n_ + pq.p] = hpq;
        }
        inv_.set_coordinate(alpha, beta);
        std::size_t kopt = 0;
        for (std::size_t j = 1; j < m_; ++j)
            if (model_.fval[j] < model_.fval[kopt]) kopt = j;
        model_.kopt = kopt;
        if (kopt != 0) {
            model_.hess_mul(model_.point(kopt), work_);
            for (std::size_t i = 0; i < n_; ++i) model_.g[i] += work_[i];
        }
    }

    // ====================================================== shared pieces
    [[nodiscard]] double max_distance(std::span<const double> centre) const {
        double best = 0.0;
        for (std::size_t j = 0; j < m_; ++j) {
            double s = 0.0;
            std::span<const double> y = model_.point(j);
            for (std::size_t i = 0; i < n_; ++i) s += (y[i] - centre[i]) * (y[i] - centre[i]);
            best = std::max(best, s);
        }
        return std::sqrt(best);
    }

    // (6.1): the t != kopt maximising w_t sigma_t, sigma_t = H_tt beta +
    // vlag_t^2, w_t from Options::newuoa_weight around centre.
    [[nodiscard]] std::size_t select_t(std::span<const double> centre, double beta) const {
        std::size_t best = m_;
        double best_w = 0.0;
        const double scale = opts_.newuoa_weight ? std::max(kWeightDelta * delta_, rho_) : delta_;
        const double scale2 = scale * scale;
        for (std::size_t j = 0; j < m_; ++j) {
            if (j == model_.kopt) continue;
            double dist2 = 0.0;
            std::span<const double> y = model_.point(j);
            for (std::size_t i = 0; i < n_; ++i) dist2 += (y[i] - centre[i]) * (y[i] - centre[i]);
            const double r2 = dist2 / scale2;
            const double weight = std::max(1.0, opts_.newuoa_weight ? r2 * r2 * r2 : r2);
            const double sigma = inv_.omega_diag(j) * beta + vlag_[j] * vlag_[j];
            const double val = weight * sigma;
            if (best == m_ || val > best_w) {
                best = j;
                best_w = val;
            }
        }
        return best;
    }

    [[nodiscard]] double sigma_of(std::size_t t, double beta) const {
        return inv_.omega_diag(t) * beta + vlag_[t] * vlag_[t];
    }

    // Section 6, (6.13)-(6.17): the origin moves to x_k.
    void shift_origin() {
        std::span<const double> xk = model_.xopt();
        std::copy(xk.begin(), xk.end(), shift_.begin());
        const std::span<const double> s = shift_;
        inv_.shift(model_.points, s, gamma_, cmat_, zw_, work_);
        model_.shift_hessian(s, work_);
        for (std::size_t j = 0; j < m_; ++j)
            for (std::size_t i = 0; i < n_; ++i) model_.points[j * n_ + i] -= s[i];
        box_.shift(s);
        for (std::size_t i = 0; i < n_; ++i) xbase_[i] += s[i];
    }

    // Section 4: point t becomes xnew with value f, H already updated for
    // it. diff = f - Q(xnew).
    void update_model(std::size_t t, double f, double diff) {
        std::copy(model_.xopt().begin(), model_.xopt().end(), xold_.begin());
        const std::span<const double> xold = xold_;
        const bool improved = f < model_.fopt();
        model_.absorb(t);
        for (std::size_t i = 0; i < n_; ++i) model_.points[t * n_ + i] = xnew_[i];
        model_.fval[t] = f;
        inv_.omega_col(t, lam_);
        for (std::size_t l = 0; l < m_; ++l) {
            lam_[l] *= diff;
            model_.mu[l] += lam_[l];
        }
        // (4.8): the gradient at the old x_k.
        for (std::size_t i = 0; i < n_; ++i) model_.g[i] += diff * inv_.b(t, i);
        for (std::size_t l = 0; l < m_; ++l) {
            std::span<const double> yl = model_.point(l);
            const double c = lam_[l] * dot(yl, xold);
            for (std::size_t i = 0; i < n_; ++i) model_.g[i] += c * yl[i];
        }
        if (improved) {
            for (std::size_t i = 0; i < n_; ++i) work2_[i] = xnew_[i] - xold[i];
            model_.hess_mul(work2_, work_);
            for (std::size_t i = 0; i < n_; ++i) model_.g[i] += work_[i];
            model_.kopt = t;
        }
    }

#ifdef FLOP_BOBYQA_TRACE
    // The largest |Q(y_j) - F(y_j)| over the points, relative to the
    // spread of the values: zero to rounding while the model interpolates.
    void trace_interpolation(const char* where) {
        std::vector<double> dj(n_), hd(n_);
        double worst = 0.0, spread = 0.0;
        for (std::size_t j = 0; j < m_; ++j) {
            for (std::size_t i = 0; i < n_; ++i) dj[i] = model_.point(j)[i] - model_.xopt()[i];
            const double q = model_.fopt() + model_.change(dj, hd);
            worst = std::max(worst, std::fabs(q - model_.fval[j]));
            spread = std::max(spread, std::fabs(model_.fval[j] - model_.fopt()));
        }
        FLOP_BOBYQA_LOG("[bobyqa] interp %s %.2e\n", where, spread > 0.0 ? worst / spread : worst);
    }
#endif

    void push_error(double err, double dnorm) {
        errors_[error_next_] = err;
        error_steps_[error_next_] = dnorm;
        error_next_ = (error_next_ + 1) % kErrorHistory;
        if (error_count_ < kErrorHistory) ++error_count_;
    }

    // (6.8): epsilon_max is usable when the last three steps were evaluated
    // since the last RESCUE and none was longer than rho.
    bool usable_error(double& eps) const {
        if (error_count_ < kErrorHistory) return false;
        eps = 0.0;
        for (std::size_t k = 0; k < kErrorHistory; ++k) {
            if (error_steps_[k] > rho_) return false;
            eps = std::max(eps, errors_[k]);
        }
        return true;
    }

    [[nodiscard]] bool ftol_met(double before, double after) const {
        const Stopping& s = opts_.stopping;
        return (s.ftol_abs > 0.0 && difference_within_abs(before, after, s.ftol_abs)) ||
               (s.ftol_rel > 0.0 && difference_within_rel(before, after, s.ftol_rel));
    }

    // The projection of (6.12) at x_k: components into the box only.
    [[nodiscard]] double projected_norm2(std::span<const double> g) const {
        std::span<const double> xk = model_.xopt();
        double s = 0.0;
        for (std::size_t i = 0; i < n_; ++i) {
            double gi = g[i];
            if (box_.at_lo(i, xk[i])) gi = std::min(0.0, gi);
            if (box_.at_hi(i, xk[i])) gi = std::max(0.0, gi);
            s += gi * gi;
        }
        return s;
    }

    // (6.12): the least-Frobenius-norm model of the current values, which
    // replaces Q after kAltModelRuns consecutive trust-region iterations on
    // which its projected gradient is the much smaller one.
    void alternative_model_test() {
        const double fo = model_.fopt();
        for (std::size_t c = 0; c < inv_.columns(); ++c) {
            double t = 0.0;
            for (std::size_t j = 0; j < m_; ++j) t += inv_.z(j, c) * (model_.fval[j] - fo);
            zw_[c] = t;
        }
        for (std::size_t j = 0; j < m_; ++j) {
            double t = 0.0;
            for (std::size_t c = 0; c < inv_.columns(); ++c) t += inv_.z(j, c) * zw_[c];
            lam_[j] = t;
        }
        std::span<const double> xk = model_.xopt();
        for (std::size_t i = 0; i < n_; ++i) {
            double t = 0.0;
            for (std::size_t j = 0; j < m_; ++j) t += inv_.b(j, i) * (model_.fval[j] - fo);
            work_[i] = t;
        }
        for (std::size_t l = 0; l < m_; ++l) {
            std::span<const double> yl = model_.point(l);
            const double c = lam_[l] * dot(yl, xk);
            for (std::size_t i = 0; i < n_; ++i) work_[i] += c * yl[i];
        }
        if (projected_norm2(work_) <= kAltModelRatio * projected_norm2(model_.g)) {
            ++alt_model_runs_;
        } else {
            alt_model_runs_ = 0;
        }
        if (alt_model_runs_ >= kAltModelRuns) {
            for (double& v : model_.hq) v = 0.0;
            for (std::size_t l = 0; l < m_; ++l) model_.mu[l] = lam_[l];
            for (std::size_t i = 0; i < n_; ++i) model_.g[i] = work_[i];
            alt_model_runs_ = 0;
        }
    }

    // ============================================================== RESCUE
    // Section 5. Returns true when the interpolation points changed (T is
    // not empty).
    bool rescue() {
        // The origin moves to x_k; H is rebuilt, so only the model moves.
        std::vector<double> s(model_.xopt().begin(), model_.xopt().end());
        model_.shift_hessian(s, work_);
        for (std::size_t j = 0; j < m_; ++j)
            for (std::size_t i = 0; i < n_; ++i) model_.points[j * n_ + i] -= s[i];
        box_.shift(s);
        for (std::size_t i = 0; i < n_; ++i) xbase_[i] += s[i];

        RescuePlan plan = rescue_plan(model_, box_, delta_, inv_);
        FLOP_BOBYQA_LOG("[bobyqa] rescue ev=%zu\n", evals_);
        std::vector<char> kept(m_, 0);
        for (std::size_t j = 0; j < m_; ++j)
            if (plan.origin[j] != kNewPoint) kept[plan.origin[j]] = 1;
        for (std::size_t i = 0; i < m_; ++i)
            if (!kept[i]) model_.absorb(i);
        std::vector<double> fval(m_, 0.0), mu(m_, 0.0);
        std::vector<std::size_t> fresh;
        for (std::size_t j = 0; j < m_; ++j) {
            if (plan.origin[j] == kNewPoint) {
                fresh.push_back(j);
                continue;
            }
            fval[j] = model_.fval[plan.origin[j]];
            mu[j] = model_.mu[plan.origin[j]];
        }
        model_.points = plan.points;
        model_.fval = fval;
        model_.mu = mu;
        model_.kopt = 0;
        error_count_ = 0;
        if (fresh.empty()) return false;

        // The new values, in order of slot, then (5.12) for each in turn.
        std::vector<double> vals(m_, 0.0);
        for (std::size_t k = 0; k < fresh.size(); ++k) {
            if (done_) return true;
            // Consecutive slots go out together.
            std::size_t first = fresh[k], last = first + 1;
            while (k + 1 < fresh.size() && fresh[k + 1] == last) {
                ++last;
                ++k;
            }
            if (!evaluate_rows(model_.points, first, last, vals)) return true;
        }
        for (const std::size_t t : fresh) {
            std::span<const double> y = model_.point(t);
            const double diff = vals[t] - model_.fopt() - model_.change(y, work_);
            inv_.omega_col(t, lam_);
            for (std::size_t l = 0; l < m_; ++l) model_.mu[l] += diff * lam_[l];
            for (std::size_t i = 0; i < n_; ++i) model_.g[i] += diff * inv_.b(t, i);
            model_.fval[t] = vals[t];
        }
        std::size_t kopt = 0;
        for (std::size_t j = 1; j < m_; ++j)
            if (model_.fval[j] < model_.fval[kopt]) kopt = j;
        if (kopt != 0) {
            model_.hess_mul(model_.point(kopt), work_);
            for (std::size_t i = 0; i < n_; ++i) model_.g[i] += work_[i];
            model_.kopt = kopt;
        }
        return true;
    }

    // ================================================== the two iterations
    // Section 3 and 6: a trust-region iteration.
    Next trust_iteration() {
        bool rescued = false, last_rescue_changed = false;
        for (;;) {
            if (done_) return Next::Trust;
            const TrsboxResult tr = trsbox(model_, box_, delta_, d_, side_, gd_, tw_);
            place(box_, model_.xopt(), d_, side_, xnew_);
            std::span<const double> xk = model_.xopt();
            double d2 = 0.0;
            for (std::size_t i = 0; i < n_; ++i) {
                dvec_[i] = xnew_[i] - xk[i];
                d2 += dvec_[i] * dvec_[i];
            }
            const double dlen = std::sqrt(d2);
            const double dnorm = std::min(delta_, dlen);
            if (dnorm < kShort * rho_) {
                const double dist = max_distance(xk);
                delta_ = std::min(kShortCut * delta_, 0.5 * dist);
                if (delta_ <= kSnap * rho_) delta_ = rho_;
                pending_short_ = d2 > 0.0;
                FLOP_BOBYQA_LOG("[bobyqa] short ev=%zu rho=%.2e delta=%.2e |d|=%.2e dist=%.2e\n",
                                evals_, rho_, delta_, dlen, dist);
                if (dist <= kFar * rho_) return Next::ReduceRho;
                double eps = 0.0;
                if (usable_error(eps) && model_adequate(eps, tr)) return Next::ReduceRho;
                pending_short_ = false;
                return Next::Alternative;
            }
            pending_short_ = false;
            double xk2 = 0.0;
            for (std::size_t i = 0; i < n_; ++i) xk2 += xk[i] * xk[i];
            if (d2 <= kShiftRatio * xk2) {
                shift_origin();
                for (std::size_t i = 0; i < n_; ++i) xnew_[i] -= shift_[i];
            }
            const double beta =
                inv_.vlag_beta(model_.points, model_.kopt, xnew_, vlag_, wv_, zw_, work_, sum_);
            std::size_t t = select_t(model_.xopt(), beta);
            const double tau = vlag_[t];
            if (sigma_of(t, beta) <= kSigmaRatio * tau * tau) {
                if (rescued && !last_rescue_changed) {
                    stop(Status::RoundoffLimited);
                    return Next::Trust;
                }
                last_rescue_changed = rescue();
                rescued = true;
                continue;
            }
            double f = 0.0;
            if (!evaluate(xnew_, f)) return Next::Trust;
            const double fo = model_.fopt();
            const double change = model_.change(dvec_, work_);
            const double qred = -change;
            const double diff = f - fo - change;
            push_error(std::fabs(diff), dnorm);
            if (!(qred > 0.0)) {
                stop(Status::RoundoffLimited);
                return Next::Trust;
            }
            const double ratio = (fo - f) / qred;
            FLOP_BOBYQA_LOG("[bobyqa] tr  ev=%zu rho=%.2e delta=%.2e |d|=%.2e ratio=%.3f f=%.6e\n",
                            evals_, rho_, delta_, dlen, ratio, f);
            // (6.4).
            const double old_delta = delta_;
            if (ratio <= kRatioLow)
                delta_ = std::min(0.5 * old_delta, dlen);
            else if (ratio <= kRatioHigh)
                delta_ = std::max(0.5 * old_delta, dlen);
            else
                delta_ = std::max(0.5 * old_delta, 2.0 * dlen);
            if (delta_ <= kSnap * rho_) delta_ = rho_;
            const bool improved = f < fo;
            if (improved) {
                // The second choice of t, centred on the new point (6.1).
                const double keep = delta_;
                delta_ = old_delta;
                const std::size_t t2 = select_t(xnew_, beta);
                delta_ = keep;
                if (sigma_of(t2, beta) > kSigmaRatio * vlag_[t2] * vlag_[t2]) t = t2;
            }
            inv_.update(t, vlag_, beta, hcol_, ucol_);
            update_model(t, f, diff);
#ifdef FLOP_BOBYQA_TRACE
            trace_interpolation("tr");
#endif
            if (done_) return Next::Trust;
            if (improved && ftol_met(fo, f)) {
                stop(Status::FtolReached);
                return Next::Trust;
            }
            alternative_model_test();
            if (ratio >= kRatioLow) return Next::Trust;
            const double dist = max_distance(model_.xopt());
            if (dist > std::max(kFarDelta * delta_, kFar * rho_)) return Next::Alternative;
            if (improved || std::max(dlen, delta_) > rho_) return Next::Trust;
            return Next::ReduceRho;
        }
    }

    // (6.10) and (6.11): the model is good enough to give up rho.
    [[nodiscard]] bool model_adequate(double eps, const TrsboxResult& tr) const {
        FLOP_BOBYQA_LOG("[bobyqa] adequate? eps=%.3e crv=%d %.3e\n", eps, tr.have_curvature ? 1 : 0,
                        tr.curvature);
        if (tr.have_curvature && eps > kCurvatureFactor * rho_ * rho_ * tr.curvature) return false;
        for (std::size_t i = 0; i < n_; ++i) {
            double v = 0.0;
            if (box_.at_lo(i, xnew_[i]))
                v = rho_;
            else if (box_.at_hi(i, xnew_[i]))
                v = -rho_;
            else
                continue;
            const double first = v * gd_[i];
            const double second = first + 0.5 * v * v * model_.hess_diag(i);
            if (eps > std::max(first, second)) {
                FLOP_BOBYQA_LOG("[bobyqa]   bound %zu fails: %.3e %.3e\n", i, first, second);
                return false;
            }
        }
        return true;
    }

    // Section 3 and 6: an alternative iteration.
    Next alternative_iteration() {
        bool rescued = false;
        for (;;) {
            if (done_) return Next::Trust;
            std::span<const double> xk = model_.xopt();
            // (3.7): the point furthest from x_k.
            std::size_t t = m_;
            double far2 = 0.0;
            for (std::size_t j = 0; j < m_; ++j) {
                if (j == model_.kopt) continue;
                double s = 0.0;
                std::span<const double> y = model_.point(j);
                for (std::size_t i = 0; i < n_; ++i) s += (y[i] - xk[i]) * (y[i] - xk[i]);
                if (t == m_ || s > far2) {
                    t = j;
                    far2 = s;
                }
            }
            const double dist = std::sqrt(far2);
            double dalt = delta_;
            if (dist < kFar * delta_) dalt = std::max(kAltDelta * dist, rho_);
            const double cauchy2 = altmov(model_, inv_, box_, t, dalt, d_, side_, c_, cside_, aw_);
            // (6.13) applies to this step as to a trust-region one: every
            // step of length at least rho / 2. Both candidate steps are
            // displacements from x_k, so they survive the shift unchanged.
            const double dd = dot(d_, d_);
            if (dd >= 0.25 * rho_ * rho_ && dd <= kShiftRatio * dot(xk, xk)) shift_origin();
            place(box_, xk, d_, side_, xnew_);
            double beta =
                inv_.vlag_beta(model_.points, model_.kopt, xnew_, vlag_, wv_, zw_, work_, sum_);
            double sigma = sigma_of(t, beta);
            if (cauchy2 > sigma) {
                place(box_, xk, c_, cside_, xnew_);
                beta =
                    inv_.vlag_beta(model_.points, model_.kopt, xnew_, vlag_, wv_, zw_, work_, sum_);
                sigma = sigma_of(t, beta);
            }
            const double tau = vlag_[t];
            if (sigma <= kSigmaRatio * tau * tau) {
                if (rescued) {
                    stop(Status::RoundoffLimited);
                    return Next::Trust;
                }
                rescued = true;
                if (rescue()) return Next::Trust;
                continue;
            }
            double dlen2 = 0.0;
            for (std::size_t i = 0; i < n_; ++i) {
                dvec_[i] = xnew_[i] - xk[i];
                dlen2 += dvec_[i] * dvec_[i];
            }
            double f = 0.0;
            if (!evaluate(xnew_, f)) return Next::Trust;
            const double fo = model_.fopt();
            const double diff = f - fo - model_.change(dvec_, work_);
            // The step's length as its trust region bounds it: a step to the
            // boundary can come out an ulp longer than the radius.
            push_error(std::fabs(diff), std::min(dalt, std::sqrt(dlen2)));
            const bool improved = f < fo;
            FLOP_BOBYQA_LOG("[bobyqa] alt ev=%zu rho=%.2e delta=%.2e |d|=%.2e cauchy=%d f=%.6e\n",
                            evals_, rho_, delta_, std::sqrt(dlen2), cauchy2 > sigma ? 1 : 0, f);
            inv_.update(t, vlag_, beta, hcol_, ucol_);
            update_model(t, f, diff);
#ifdef FLOP_BOBYQA_TRACE
            trace_interpolation("alt");
#endif
            if (!done_ && improved && ftol_met(fo, f)) stop(Status::FtolReached);
            return Next::Trust;
        }
    }

    // (6.6): the next rho, or the end.
    Next reduce_rho() {
        if (rho_ <= rhoend_) {
            // A last short step: the report evaluates it, so that the
            // result can be x_k + d when that is lower.
            if (pending_short_) {
                double f = 0.0;
                evaluate(xnew_, f);
            }
            stop(rhoend_from_caller_ ? Status::XtolReached : Status::RoundoffLimited);
            return Next::Trust;
        }
        pending_short_ = false;
        FLOP_BOBYQA_LOG("[bobyqa] rho ev=%zu rho=%.2e\n", evals_, rho_);
        const double old = rho_;
        if (rho_ <= kRhoNear * rhoend_)
            rho_ = rhoend_;
        else if (rho_ <= kRhoFar * rhoend_)
            rho_ = std::sqrt(rho_ * rhoend_);
        else
            rho_ = kRhoCut * rho_;
        delta_ = std::max(0.5 * old, rho_);
        return Next::Trust;
    }

    Result finish() {
        Result r;
        r.x = best_x_;
        r.f = best_f_;
        r.evaluations = evals_;
        r.status = status_;
        r.final_radius = rho_;
        return r;
    }

    Eval eval_;
    std::size_t n_, m_;
    const flop::bobyqa::Options& opts_;
    const Bounds* bounds_;
    Model model_;
    Inverse inv_;
    Box box_;
    TrsboxWork tw_;
    AltmovWork aw_;

    double rhobeg_ = 0.0, rho_ = 0.0, rhoend_ = 0.0, floor_ = 0.0, delta_ = 0.0;
    bool rhoend_from_caller_ = false;
    bool pending_short_ = false;
    bool done_ = false;
    Status status_ = Status::RoundoffLimited;
    std::size_t evals_ = 0;
    bool have_best_ = false;
    double best_f_ = 0.0;
    std::vector<double> best_x_;
    std::size_t alt_model_runs_ = 0;
    double errors_[kErrorHistory] = {};
    double error_steps_[kErrorHistory] = {};
    std::size_t error_next_ = 0, error_count_ = 0;

    std::vector<double> xbase_, abs_lo_, abs_hi_, scratch_;
    std::span<double> vlag_, wv_, zw_, dvec_, sum_, hcol_, ucol_, gamma_, cmat_;
    std::span<double> work_, work2_, d_, c_, gd_, xnew_, x_, lam_, shift_, xold_;
    std::vector<Side> sides_;
    std::span<Side> side_, cside_;
};

}  // namespace flop::detail::bobyqa
