// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// cobyla_impl - Powell's COBYLA, from the 1994 paper
// =============================================================================
//
// M. J. D. Powell, "A direct search optimization method that models the
// objective and constraint functions by linear interpolation", in Advances in
// Optimization and Numerical Analysis (S. Gomez and J.-P. Hennart, eds.),
// Kluwer, 1994, pp. 51-67. Section numbers below refer to that paper.
//
// The state is Powell's: a simplex of n + 1 points held as a base point and n
// displacement vectors (the columns of S), the inverse S^-1 held row by row,
// and the objective and constraint values at every vertex. The linear models
// come from the simplex, the trust-region step from cobyla_step.hpp, and the
// two simplex updates below are the only ways the simplex ever changes:
//
//   replace_vertex(j, d)  column j of S becomes d. With w = S^-1 d, the rows
//                         of the inverse become r_j / w_j and r_i - w_i r_j / w_j,
//                         a rank-one update that needs w_j != 0.
//   move_base(j)          the base moves to vertex j. Every column loses s_j,
//                         column j becomes -s_j, and row j of the inverse
//                         becomes minus the sum of all rows; the other rows do
//                         not change.
//
// Both are O(n^2). Nothing here recomputes an inverse.
//
// The method carries no random state, uses no infinity and no NaN, and every
// objective value is checked by its bits the moment it lands in memory. The
// solver is a template on the objective so that an evaluation inlines into
// the loop; it is instantiated once by the facade in the library and by any
// caller of the template entry points in the caller's own translation unit,
// under the caller's own floating-point model, which is why none of the
// arithmetic here may depend on that model.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#ifdef FLOP_COBYLA_TRACE
#include <cstdio>
#endif

#include "flop/concepts.hpp"
#include "flop/detail/cobyla_step.hpp"
#include "flop/detail/fp.hpp"
#include "flop/options.hpp"
#include "flop/result.hpp"

namespace flop::cobyla {

// The COBYLA-specific settings on top of the shared ones.
struct Options : flop::Options {
    // Powell's rhoend, the trust-region radius at which the method stops.
    // 0 derives it from the x tolerances: the larger of xtol_abs and
    // xtol_rel * initial_step.
    double final_trust_radius = 0.0;
};

}  // namespace flop::cobyla

namespace flop::detail::cobyla {

// Section 2: the four constants of the method.
//   alpha  a vertex closer than alpha * rho to its opposite face is too close
//   beta   an edge longer than beta * rho is too long
//   gamma  a geometry step places the new vertex gamma * rho from the face
//   delta  an edge longer than delta * rho is preferred for replacement
constexpr double kAlpha = 0.25;
constexpr double kBeta = 2.1;
constexpr double kGamma = 0.5;
constexpr double kDelta = 1.1;

// A problem with no constraints, so the unconstrained entry points share one
// solver with the constrained ones.
struct NoConstraints {
    void operator()(std::span<const double>, std::span<double>) const noexcept {}
};

// The two objective channels behind one interface. The solver asks for a
// batch only for its initial simplex, and only when the objective offers one.
template <ScalarObjective F>
struct ScalarEvaluator {
    F& f;
    static constexpr bool has_batch = false;
    double operator()(std::span<const double> x) { return static_cast<double>(f(x)); }
    void batch(std::span<const std::span<const double>> xs, std::span<double> out) {
        for (std::size_t i = 0; i < xs.size(); ++i) out[i] = static_cast<double>(f(xs[i]));
    }
};

template <BatchObjective F>
struct BatchEvaluator {
    F& f;
    static constexpr bool has_batch = true;
    double operator()(std::span<const double> x) {
        double out = 0.0;
        std::span<const double> one[1] = {x};
        f(std::span<const std::span<const double>>(one, 1), std::span<double>(&out, 1));
        return out;
    }
    void batch(std::span<const std::span<const double>> xs, std::span<double> out) { f(xs, out); }
};

template <class Eval, ConstraintFunction C>
class Solver {
public:
    Solver(Eval eval, C& c, std::size_t m, std::span<const double> x0,
           const flop::cobyla::Options& opts)
        : eval_(eval),
          c_(c),
          n_(x0.size()),
          m_(m),
          opts_(opts),
          bounds_(opts.bounds ? &*opts.bounds : nullptr) {
        rho_ = opts.initial_step;
        rhoend_ =
            opts.final_trust_radius > 0.0
                ? opts.final_trust_radius
                : std::max(opts.stopping.xtol_abs, opts.stopping.xtol_rel * opts.initial_step);
        // Below roughly machine precision of the step scale the simplex cannot
        // be told from a point, so the radius has a floor even when no x
        // tolerance was asked for; reaching that floor is RoundoffLimited,
        // reaching the caller's own rhoend is XtolReached.
        floor_ = opts.initial_step * std::numeric_limits<double>::epsilon();
        rhoend_from_caller_ = rhoend_ > floor_;
        rhoend_ = std::max(rhoend_, floor_);
        if (rhoend_ > rho_) rhoend_ = rho_;

        base_.assign(x0.begin(), x0.end());
        sim_.assign(n_ * n_, 0.0);
        simi_.assign(n_ * n_, 0.0);
        for (std::size_t i = 0; i < n_; ++i) {
            sim_[i * n_ + i] = rho_;
            simi_[i * n_ + i] = 1.0 / rho_;
        }
        fval_.assign(n_ + 1, 0.0);
        cval_.assign((n_ + 1) * std::max<std::size_t>(m_, 1), 0.0);
        viol_.assign(n_ + 1, 0.0);
        x_.assign(n_, 0.0);
        cbuf_.assign(std::max<std::size_t>(m_, 1), 0.0);
        gf_.assign(n_, 0.0);
        gc_.assign(m_ * n_, 0.0);
        d_.assign(n_, 0.0);
        w_.assign(n_, 0.0);
        vsig_.assign(n_, 0.0);
        veta_.assign(n_, 0.0);
    }

    Result run() {
        initial_simplex();
        while (!done_) iterate();
        return finish();
    }

private:
    // ================================================================== eval
    // One objective evaluation at x, with the constraint values into c and
    // the violation into viol. False when the cap forbids it.
    bool evaluate(std::span<const double> x, double& f, std::span<double> c, double& viol) {
        if (cap_reached()) return false;
        const double value = eval_(x);
        record(x, value, c, f, viol);
        return true;
    }

    bool cap_reached() {
        if (opts_.stopping.max_evaluations > 0 && evals_ >= opts_.stopping.max_evaluations) {
            stop(Status::MaxEvaluationsReached);
            return true;
        }
        return false;
    }

    // Everything that happens once a value exists: the bit check, the
    // constraints, the count, the trace and the stop-value test.
    void record(std::span<const double> x, const double& value, std::span<double> c, double& f,
                double& viol) {
        f = value;
        if (fp_bad(f))
            throw std::runtime_error("flop::cobyla: the objective returned a non-finite value");
        viol = 0.0;
        if (m_ > 0) {
            c_(x, c);
            for (std::size_t k = 0; k < m_; ++k) {
                if (fp_bad(c[k]))
                    throw std::runtime_error(
                        "flop::cobyla: a constraint returned a non-finite value");
                viol = std::max(viol, -c[k]);
            }
        }
        ++evals_;
        if (opts_.on_evaluation) {
            Evaluation ev{evals_ - 1, x, f, std::span<const double>(c.data(), m_)};
            opts_.on_evaluation(ev);
        }
        if (opts_.stopping.stop_value && f <= *opts_.stopping.stop_value && !done_) {
            stop_at_point_ = true;
            stop_x_.assign(x.begin(), x.end());
            stop_f_ = f;
            stop_viol_ = viol;
            stop(Status::StopValueReached);
        }
    }

    void stop(Status s) {
        if (!done_) {
            done_ = true;
            status_ = s;
        }
    }

    // ================================================================ simplex
    std::span<double> col(std::size_t j) { return {sim_.data() + j * n_, n_}; }
    std::span<double> row(std::size_t j) { return {simi_.data() + j * n_, n_}; }
    std::span<double> cvals(std::size_t j) {
        return {cval_.data() + j * std::max<std::size_t>(m_, 1), m_};
    }
    [[nodiscard]] double merit(std::size_t j) const { return fval_[j] + mu_ * viol_[j]; }
    [[nodiscard]] std::size_t base_index() const { return n_; }

    // Column j of S becomes d (a displacement from the base). Requires
    // w_j = r_j . d to be non-zero; the caller checks.
    void replace_vertex(std::size_t j, std::span<const double> d) {
        for (std::size_t i = 0; i < n_; ++i) w_[i] = dot(row(i), d);
        const double wj = w_[j];
        auto rj = row(j);
        for (std::size_t i = 0; i < n_; ++i) rj[i] /= wj;
        for (std::size_t i = 0; i < n_; ++i) {
            if (i == j) continue;
            auto ri = row(i);
            const double wi = w_[i];
            for (std::size_t k = 0; k < n_; ++k) ri[k] -= wi * rj[k];
        }
        auto sj = col(j);
        for (std::size_t i = 0; i < n_; ++i) sj[i] = d[i];
    }

    // The base moves to vertex j and vertex j takes the old base.
    void move_base(std::size_t j) {
        auto sj = col(j);
        for (std::size_t i = 0; i < n_; ++i) base_[i] += sj[i];
        for (std::size_t k = 0; k < n_; ++k) {
            if (k == j) continue;
            auto sk = col(k);
            for (std::size_t i = 0; i < n_; ++i) sk[i] -= sj[i];
        }
        for (std::size_t i = 0; i < n_; ++i) sj[i] = -sj[i];
        auto rj = row(j);
        for (std::size_t i = 0; i < n_; ++i) w_[i] = 0.0;
        for (std::size_t k = 0; k < n_; ++k) {
            auto rk = row(k);
            for (std::size_t i = 0; i < n_; ++i) w_[i] += rk[i];
        }
        for (std::size_t i = 0; i < n_; ++i) rj[i] = -w_[i];
        std::swap(fval_[j], fval_[n_]);
        std::swap(viol_[j], viol_[n_]);
        auto cj = cvals(j), cb = cvals(n_);
        for (std::size_t k = 0; k < m_; ++k) std::swap(cj[k], cb[k]);
    }

    void store_vertex(std::size_t j, double f, std::span<const double> c, double viol) {
        fval_[j] = f;
        viol_[j] = viol;
        auto cj = cvals(j);
        for (std::size_t k = 0; k < m_; ++k) cj[k] = c[k];
    }

    // The displacement of initial vertex i along coordinate i: +rho when the
    // box allows it, else -rho, else whichever side has more room, shrunk to
    // that room. Without bounds it is always +rho.
    [[nodiscard]] double initial_offset(std::size_t i) const {
        if (!bounds_) return rho_;
        const std::optional<double>& hi = bounds_->upper[i];
        const std::optional<double>& lo = bounds_->lower[i];
        const double up = hi.has_value() ? *hi - base_[i] : rho_;
        const double down = lo.has_value() ? base_[i] - *lo : rho_;
        if (up >= rho_) return rho_;
        if (down >= rho_) return -rho_;
        return up >= down ? up : -down;
    }

    // Section 2: the n + 1 initial points, x0 and x0 displaced along each
    // coordinate. Sequentially the base moves to any vertex that improves on
    // it as the simplex is built; from a batch, all n + 1 go out in one call
    // and the best becomes the base afterwards. A cap below n + 1 forces the
    // sequential path so the cap is honoured exactly.
    void initial_simplex() {
        double f = 0.0, v = 0.0;
        const bool batch = Eval::has_batch && (opts_.stopping.max_evaluations == 0 ||
                                               opts_.stopping.max_evaluations >= n_ + 1);
        if (batch) {
            std::vector<double> points((n_ + 1) * n_);
            std::vector<std::span<const double>> xs(n_ + 1);
            std::vector<double> out(n_ + 1);
            for (std::size_t j = 0; j <= n_; ++j) {
                for (std::size_t i = 0; i < n_; ++i) points[j * n_ + i] = base_[i];
                if (j < n_) points[j * n_ + j] += initial_offset(j);
                xs[j] = std::span<const double>(points.data() + j * n_, n_);
            }
            eval_.batch(xs, out);
            for (std::size_t j = 0; j <= n_; ++j) {
                record(xs[j], out[j], cbuf_, f, v);
                store_vertex(j, f, cbuf_, v);
            }
            for (std::size_t j = 0; j < n_; ++j) {
                for (std::size_t i = 0; i < n_; ++i) d_[i] = points[j * n_ + i] - base_[i];
                replace_vertex(j, d_);
            }
            if (done_) return;
            std::size_t best = n_;
            for (std::size_t j = 0; j < n_; ++j)
                if (merit(j) < merit(best)) best = j;
            if (best != n_) move_base(best);
            return;
        }
        if (!evaluate(base_, f, cbuf_, v)) return;
        store_vertex(n_, f, cbuf_, v);
        if (done_) return;
        for (std::size_t j = 0; j < n_; ++j) {
            for (std::size_t i = 0; i < n_; ++i) x_[i] = base_[i];
            x_[j] += initial_offset(j);
            if (!evaluate(x_, f, cbuf_, v)) return;
            for (std::size_t i = 0; i < n_; ++i) d_[i] = x_[i] - base_[i];
            replace_vertex(j, d_);
            store_vertex(j, f, cbuf_, v);
            if (done_) return;
            if (merit(j) < merit(n_)) move_base(j);
        }
    }

    // ============================================================= iteration
    // Section 2: the linear models from the simplex, and the geometry
    // measures. vsig_j is the distance from vertex j to the face through the
    // other n points, 1 / |r_j|; veta_j is the edge length |s_j|.
    bool build_models() {
        for (std::size_t i = 0; i < n_; ++i) gf_[i] = 0.0;
        for (std::size_t k = 0; k < m_ * n_; ++k) gc_[k] = 0.0;
        for (std::size_t j = 0; j < n_; ++j) {
            auto rj = row(j);
            const double df = fval_[j] - fval_[n_];
            for (std::size_t i = 0; i < n_; ++i) gf_[i] += df * rj[i];
            auto cj = cvals(j), cb = cvals(n_);
            for (std::size_t k = 0; k < m_; ++k) {
                const double dc = cj[k] - cb[k];
                for (std::size_t i = 0; i < n_; ++i) gc_[k * n_ + i] += dc * rj[i];
            }
            const double rn = norm(rj);
            if (rn == 0.0) return false;
            vsig_[j] = 1.0 / rn;
            veta_[j] = norm(col(j));
        }
        return true;
    }

    [[nodiscard]] bool acceptable() const {
        for (std::size_t j = 0; j < n_; ++j)
            if (vsig_[j] < kAlpha * rho_ || veta_[j] > kBeta * rho_) return false;
        return true;
    }

    // The linearised merit change for a displacement d from the base.
    [[nodiscard]] double linear_merit(std::span<const double> d) const {
        double lin = dot(gf_, d);
        if (m_ > 0) {
            double worst = 0.0;
            auto cb = cvals_const(n_);
            for (std::size_t k = 0; k < m_; ++k) {
                double v = cb[k];
                for (std::size_t i = 0; i < n_; ++i) v += gc_[k * n_ + i] * d[i];
                worst = std::max(worst, -v);
            }
            lin += mu_ * worst;
        }
        return lin;
    }
    [[nodiscard]] std::span<const double> cvals_const(std::size_t j) const {
        return {cval_.data() + j * std::max<std::size_t>(m_, 1), m_};
    }

    // The largest factor t in (0, 1] with base + t d inside the box.
    [[nodiscard]] double box_factor(std::span<const double> d) const {
        if (!bounds_) return 1.0;
        double t = 1.0;
        for (std::size_t i = 0; i < n_; ++i) {
            const std::optional<double>& hi = bounds_->upper[i];
            const std::optional<double>& lo = bounds_->lower[i];
            if (d[i] > 0.0 && hi.has_value()) t = std::min(t, (*hi - base_[i]) / d[i]);
            if (d[i] < 0.0 && lo.has_value()) t = std::min(t, (*lo - base_[i]) / d[i]);
        }
        return std::max(t, 0.0);
    }

    // Section 2: improve the simplex. The vertex to move is the one at the end
    // of the longest edge when an edge is too long, else the one closest to
    // its opposite face. It is moved to gamma * rho from that face along the
    // face normal r_j, on the side where the linearised merit is smaller.
    // With a box, a side that leaves it is scaled back to the box, and the
    // side allowing the longer step wins a tie of merit.
    bool geometry_step() {
        std::size_t jdrop = 0;
        double worst_eta = 0.0;
        for (std::size_t j = 0; j < n_; ++j)
            if (veta_[j] > kBeta * rho_ && veta_[j] > worst_eta) {
                worst_eta = veta_[j];
                jdrop = j;
            }
        if (worst_eta == 0.0) {
            jdrop = 0;
            for (std::size_t j = 1; j < n_; ++j)
                if (vsig_[j] < vsig_[jdrop]) jdrop = j;
        }
        auto rj = row(jdrop);
        const double scale = kGamma * rho_ * vsig_[jdrop];  // gamma rho / |r_j|
        for (std::size_t i = 0; i < n_; ++i) d_[i] = scale * rj[i];
        const double t_plus = box_factor(d_);
        for (std::size_t i = 0; i < n_; ++i) w_[i] = -d_[i];
        const double t_minus = box_factor(w_);
        for (std::size_t i = 0; i < n_; ++i) w_[i] *= t_minus;
        for (std::size_t i = 0; i < n_; ++i) d_[i] *= t_plus;
        // The side with the smaller linearised merit, except that a side the
        // box has shortened to nothing is never taken over one it has not.
        const double m_plus = linear_merit(d_), m_minus = linear_merit(w_);
        bool take_minus = (m_minus < m_plus) || (m_minus == m_plus && t_minus > t_plus);
        if (t_plus == 0.0 && t_minus > 0.0) take_minus = true;
        if (t_minus == 0.0 && t_plus > 0.0) take_minus = false;
        if (take_minus)
            for (std::size_t i = 0; i < n_; ++i) d_[i] = w_[i];
        // Both sides blocked by the box: the base sits in a corner, the
        // simplex cannot be repaired there, and only the trust-region step
        // and the radius can still say something.
        if (norm(d_) == 0.0) {
            geometry_blocked_ = true;
            return false;
        }
        geometry_blocked_ = false;
        for (std::size_t i = 0; i < n_; ++i) x_[i] = base_[i] + d_[i];
        double f = 0.0, v = 0.0;
        if (!evaluate(x_, f, cbuf_, v)) return true;
        if (dot(rj, d_) == 0.0) {
            stop(Status::RoundoffLimited);
            return true;
        }
        replace_vertex(jdrop, d_);
        store_vertex(jdrop, f, cbuf_, v);
        if (done_) return true;
        if (merit(jdrop) < merit(n_)) move_base(jdrop);
        return true;
    }

    // Section 2: halve the trust region, and let the penalty parameter fall
    // when the simplex says it is larger than the constraints need.
    void reduce_rho() {
        if (rho_ <= rhoend_) {
            stop(rhoend_from_caller_ ? Status::XtolReached : Status::RoundoffLimited);
            return;
        }
        rho_ *= 0.5;
        if (rho_ <= 1.5 * rhoend_) rho_ = rhoend_;
        geometry_blocked_ = false;
        if (mu_ > 0.0) {
            bool have_denom = false;
            double denom = 0.0;
            for (std::size_t k = 0; k < m_; ++k) {
                double cmin = cvals(0)[k], cmax = cmin;
                for (std::size_t j = 0; j <= n_; ++j) {
                    cmin = std::min(cmin, cvals(j)[k]);
                    cmax = std::max(cmax, cvals(j)[k]);
                }
                if (cmin < 0.5 * cmax) {
                    const double temp = std::max(cmax, 0.0) - cmin;
                    denom = have_denom ? std::min(denom, temp) : temp;
                    have_denom = true;
                }
            }
            if (!have_denom) {
                mu_ = 0.0;
            } else {
                double fmin = fval_[0], fmax = fval_[0];
                for (std::size_t j = 0; j <= n_; ++j) {
                    fmin = std::min(fmin, fval_[j]);
                    fmax = std::max(fmax, fval_[j]);
                }
                if (fmax - fmin < mu_ * denom) mu_ = (fmax - fmin) / denom;
            }
        }
    }

    // Section 2, the branch taken when a trust-region step fails or is too
    // short: repair the simplex first if it is not acceptable, else shrink
    // the trust region.
    void after_failed_step() {
        if (!acceptable_ && !geometry_blocked_) {
            allow_geometry_ = true;
            return;
        }
        reduce_rho();
    }

    // Section 2 and 3: one trust-region iteration.
    void trust_region_iteration() {
        const double viol_pred = trust_region_step(n_, m_, base_, cvals_const(n_), gc_, gf_, rho_,
                                                   bounds_, d_, ws_, fixed_);
        const double dnorm = norm(d_);
        if (dnorm < 0.5 * rho_) {
            after_failed_step();
            return;
        }
        const double prerec = viol_[n_] - viol_pred;  // predicted fall in the violation
        const double lin_f = dot(gf_, d_);
        double prerem = mu_ * prerec - lin_f;  // predicted fall in the merit
        if (prerem <= 0.0) {
            if (prerec > 0.0) {
                // The penalty is too small for this step to count as progress:
                // raise it past the value that makes the prediction positive,
                // then make sure the base is still the vertex of least merit.
                const double barmu = lin_f / prerec;
                if (mu_ < 1.5 * barmu) {
                    mu_ = 2.0 * barmu;
                    prerem = mu_ * prerec - lin_f;
                    std::size_t best = n_;
                    for (std::size_t j = 0; j < n_; ++j)
                        if (merit(j) < merit(best)) best = j;
                    if (best != n_) {
                        move_base(best);
                        return;
                    }
                }
            }
            if (prerem <= 0.0) {
                after_failed_step();
                return;
            }
        }

        // From here the point is a trust-region point: no geometry repair
        // until a step fails.
        allow_geometry_ = false;
        for (std::size_t i = 0; i < n_; ++i) x_[i] = base_[i] + d_[i];
        double f = 0.0, v = 0.0;
        if (!evaluate(x_, f, cbuf_, v)) return;
        if (done_) return;

        double trured = merit(n_) - (f + mu_ * v);
        if (mu_ == 0.0 && f == fval_[n_]) {
            // A flat objective: judge the step on the violation alone.
            prerem = prerec;
            trured = viol_[n_] - v;
        }

        // Which vertex the new point replaces. With w = S^-1 d, |w_j| is how
        // much of the new point lies along the direction only vertex j
        // supplies; a failed step may only replace a vertex it beats by that
        // measure (|w_j| > 1), a successful one the vertex it is most aligned
        // with. Then, among vertices whose replacement keeps the simplex
        // acceptable, prefer the one at the end of the longest edge.
        for (std::size_t i = 0; i < n_; ++i) w_[i] = dot(row(i), d_);
        double best_w = (trured > 0.0) ? 0.0 : 1.0;
        bool found = false;
        std::size_t jdrop = 0;
        for (std::size_t j = 0; j < n_; ++j) {
            const double aw = std::fabs(w_[j]);
            if (aw > best_w) {
                best_w = aw;
                jdrop = j;
                found = true;
            }
        }
        double edgmax = kDelta * rho_;
        for (std::size_t j = 0; j < n_; ++j) {
            const double sigbar = std::fabs(w_[j]) * vsig_[j];
            if (sigbar < kAlpha * rho_ && sigbar < vsig_[j]) continue;
            double edge;
            if (trured > 0.0) {
                auto sj = col(j);
                double e2 = 0.0;
                for (std::size_t i = 0; i < n_; ++i) e2 += (d_[i] - sj[i]) * (d_[i] - sj[i]);
                edge = std::sqrt(e2);
            } else {
                edge = veta_[j];
            }
            if (edge > edgmax) {
                edgmax = edge;
                jdrop = j;
                found = true;
            }
        }
        bool replaced = false;
        if (found && w_[jdrop] != 0.0) {
            replaced = true;
            const double f_before = fval_[n_];
            replace_vertex(jdrop, d_);
            store_vertex(jdrop, f, cbuf_, v);
            if (merit(jdrop) < merit(n_)) {
                move_base(jdrop);
                const double fall = f_before - fval_[n_];
                const Stopping& s = opts_.stopping;
                if ((s.ftol_abs > 0.0 && fall <= s.ftol_abs) ||
                    (s.ftol_rel > 0.0 && fall <= s.ftol_rel * std::fabs(fval_[n_]))) {
                    stop(Status::FtolReached);
                    return;
                }
            }
        }
        if (replaced && trured > 0.0 && trured >= 0.1 * prerem) return;
        after_failed_step();
    }

    // Section 2: one iteration. A trust-region step is the default; a
    // geometry step happens only when the previous trust-region step failed
    // and the simplex is not acceptable, and it is followed by a trust-region
    // step whatever the repaired simplex looks like. One repair between two
    // steps is the whole budget: repairing until every test passes would walk
    // the base along the face normals instead of the model, since each
    // repair point that improves the merit moves the base and lengthens the
    // edges behind it again.
    void iterate() {
        if (!build_models()) {
            stop(Status::RoundoffLimited);
            return;
        }
        acceptable_ = acceptable();
#ifdef FLOP_COBYLA_TRACE
        trace_iteration();
#endif
        if (!acceptable_ && allow_geometry_) {
            allow_geometry_ = false;
            if (geometry_step()) return;
        }
        trust_region_iteration();
    }

#ifdef FLOP_COBYLA_TRACE
    // Compiled in only with -DFLOP_COBYLA_TRACE: one line per iteration on
    // stderr with what the acceptability test saw. A debugging aid, never on
    // in a build that ships.
    void trace_iteration() const {
        double min_sig = vsig_[0], max_eta = veta_[0];
        std::size_t arg_sig = 0, arg_eta = 0, n_sig = 0, n_eta = 0;
        for (std::size_t j = 0; j < n_; ++j) {
            if (vsig_[j] < min_sig) {
                min_sig = vsig_[j];
                arg_sig = j;
            }
            if (veta_[j] > max_eta) {
                max_eta = veta_[j];
                arg_eta = j;
            }
            if (vsig_[j] < kAlpha * rho_) ++n_sig;
            if (veta_[j] > kBeta * rho_) ++n_eta;
        }
        std::fprintf(stderr,
                     "[cobyla] evals=%zu rho=%.3g mu=%.3g acceptable=%d allow_geometry=%d "
                     "min_vsig/rho=%.3f (j=%zu, %zu below alpha) max_veta/rho=%.3f (j=%zu, %zu "
                     "above beta) f_base=%.6g\n",
                     evals_, rho_, mu_, acceptable_ ? 1 : 0, allow_geometry_ ? 1 : 0,
                     min_sig / rho_, arg_sig, n_sig, max_eta / rho_, arg_eta, n_eta, fval_[n_]);
    }
#endif

    Result finish() {
        Result r;
        if (stop_at_point_) {
            r.x = stop_x_;
            r.f = stop_f_;
            r.max_constraint_violation = stop_viol_;
        } else {
            r.x = base_;
            r.f = fval_[n_];
            r.max_constraint_violation = viol_[n_];
        }
        r.evaluations = evals_;
        r.status = status_;
        r.final_trust_radius = rho_;
        return r;
    }

    Eval eval_;
    C& c_;
    std::size_t n_, m_;
    const flop::cobyla::Options& opts_;
    const Bounds* bounds_;

    double rho_ = 0.0, rhoend_ = 0.0, floor_ = 0.0, mu_ = 0.0;
    bool rhoend_from_caller_ = false;
    bool allow_geometry_ = false;    // Powell's IBRNCH, inverted: set only by a failed step
    bool acceptable_ = true;         // the simplex as it stood at the top of this iteration
    bool geometry_blocked_ = false;  // the last geometry attempt had nowhere to go
    bool done_ = false;
    Status status_ = Status::RoundoffLimited;
    bool stop_at_point_ = false;
    std::vector<double> stop_x_;
    double stop_f_ = 0.0, stop_viol_ = 0.0;
    std::size_t evals_ = 0;

    std::vector<double> base_, sim_, simi_, fval_, cval_, viol_;
    std::vector<double> x_, cbuf_, gf_, gc_, d_, w_, vsig_, veta_;
    StepWorkspace ws_;
    std::vector<char> fixed_;
};

}  // namespace flop::detail::cobyla
