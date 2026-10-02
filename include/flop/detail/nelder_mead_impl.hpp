// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// nelder_mead_impl - the Nelder-Mead simplex method
// =============================================================================
//
// J. A. Nelder and R. Mead, "A simplex method for function minimization",
// The Computer Journal 7(4), 1965, pp. 308-313: the method.
//
// J. C. Lagarias, J. A. Reeds, M. H. Wright and P. E. Wright, "Convergence
// properties of the Nelder-Mead simplex method in low dimensions", SIAM
// Journal on Optimization 9(1), 1998, pp. 112-147: section 2 is the precise
// statement implemented here, one iteration in five steps (order, reflect,
// expand, contract outside or inside, shrink) with its acceptance tests and
// its two tie-breaking rules.
//
// F. Gao and L. Han, "Implementing the Nelder-Mead simplex algorithm with
// adaptive parameters", Computational Optimization and Applications 51(1),
// 2012, pp. 259-277: the coefficients that depend on the dimension.
//
// The state is n + 1 vertices in fixed storage slots with their objective
// values, a rank -> slot index (rank 0 is the best vertex), the sum of all
// vertices, and per coordinate the least and greatest value any vertex has
// there. A vertex never moves in memory; ordering moves indices. One
// nonshrink iteration replaces the worst vertex and costs O(n) beyond its
// evaluations:
//
//   centroid   x-bar = (sum - x_worst) / n, from the running sum. The sum is
//              updated by the one vertex that changed and rebuilt from the
//              vertices every n + 1 replacements, which bounds the rounding
//              it accumulates to that of n + 1 updates.
//   ranking    the new vertex is placed by binary search over the n vertices
//              kept, and the indices behind it shift by one.
//   extent     per coordinate, the least and greatest vertex value and the
//              slot holding each. The new vertex updates them in O(1) per
//              coordinate; only a coordinate whose extreme was held by the
//              discarded vertex, and is not taken over by the new one, is
//              rescanned, in O(n).
//
// The x tolerances act on the radius r, the largest coordinate distance from
// a vertex to the best one, which costs O(n^2) to compute. The extents give
// the diameter D, the largest coordinate range, in O(n), and D / 2 <= r <= D
// because the best vertex lies inside every range. So D <= t proves r <= t
// and D > 2t proves r > t; the exact radius is computed only between the
// two, which is the last few iterations before a stop. Both implications
// survive rounding: each is one subtraction, rounded monotonically, against
// the same subtraction or half of it.
//
// A shrink is O(n^2) and is the only step whose n evaluations are
// independent, so it goes out as one batch when the objective takes one.
//
// Box bounds: no paper defines Nelder-Mead on a box. Every trial point is
// projected onto the box, so every evaluation is inside it and a bound can
// be reached exactly. A simplex whose every vertex sits on the same bound in
// some coordinate has lost that dimension and cannot leave the face, so
// before a tolerance stop is reported the face is tested: one point displaced
// off it by the simplex radius. If that point is better the minimum is not
// on the face, and the method starts again from it with a fresh simplex.
//
// No random state, no infinity, no NaN. Every objective value is checked by
// its bits the moment it lands in memory. The arithmetic is affine
// combinations and comparisons; none of it depends on the floating-point
// model for anything but rounding.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "flop/detail/box.hpp"
#include "flop/detail/evaluator.hpp"
#include "flop/detail/fp.hpp"
#include "flop/options.hpp"
#include "flop/result.hpp"

namespace flop::nelder_mead {

// The Nelder-Mead settings on top of the shared ones.
struct Options : flop::Options {
    // Gao and Han's coefficients, which depend on the dimension n: reflection
    // 1, expansion 1 + 2/n, contraction 3/4 - 1/(2n), shrink 1 - 1/n. At
    // n = 2 they are the standard ones. Off, and always at n = 1, where the
    // adaptive shrink coefficient would be zero: the standard 1, 2, 1/2, 1/2.
    bool adaptive_coefficients = true;
};

}  // namespace flop::nelder_mead

namespace flop::detail::nelder_mead {

// Lagarias et al., section 2: reflection rho > 0, expansion chi > 1 with
// chi > rho, contraction 0 < gamma < 1, shrink 0 < sigma < 1.
struct Coefficients {
    double rho;
    double chi;
    double gamma;
    double sigma;
};

// Nelder and Mead's values, the standard choice in Lagarias et al.
constexpr Coefficients kStandard{.rho = 1.0, .chi = 2.0, .gamma = 0.5, .sigma = 0.5};

// Gao and Han, section 3, for n >= 2; the standard values otherwise.
constexpr Coefficients coefficients(std::size_t n, bool adaptive) noexcept {
    if (!adaptive || n < 2) return kStandard;
    const auto dn = static_cast<double>(n);
    return {.rho = 1.0, .chi = 1.0 + 2.0 / dn, .gamma = 0.75 - 0.5 / dn, .sigma = 1.0 - 1.0 / dn};
}

template <class Eval>
class Solver {
public:
    Solver(Eval eval, std::span<const double> x0, const flop::nelder_mead::Options& opts)
        : eval_(eval),
          n_(x0.size()),
          opts_(opts),
          bounds_(opts.bounds ? &*opts.bounds : nullptr),
          coef_(coefficients(x0.size(), opts.adaptive_coefficients)),
          inv_n_(1.0 / static_cast<double>(x0.size())) {
        verts_.assign((n_ + 1) * n_, 0.0);
        fval_.assign(n_ + 1, 0.0);
        order_.assign(n_ + 1, 0);
        lo_.assign(n_, 0.0);
        hi_.assign(n_, 0.0);
        arglo_.assign(n_, 0);
        arghi_.assign(n_, 0);
        sum_.assign(n_, 0.0);
        dir_.assign(n_, 0.0);
        xbar_.assign(n_, 0.0);
        xr_.assign(n_, 0.0);
        xt_.assign(n_, 0.0);
        fbuf_.assign(n_ + 1, 0.0);
        spans_.resize(n_ + 1);
        best_x_.assign(x0.begin(), x0.end());
        for (std::size_t i = 0; i < n_; ++i) verts_[i] = x0[i];
    }

    Result run() {
        build_simplex(false);
        while (!done_) iterate();
        return finish();
    }

private:
    // ================================================================== eval
    std::span<double> vertex(std::size_t slot) { return {verts_.data() + slot * n_, n_}; }
    [[nodiscard]] std::span<const double> vertex(std::size_t slot) const {
        return {verts_.data() + slot * n_, n_};
    }

    bool cap_reached() {
        if (opts_.stopping.max_evaluations > 0 && evals_ >= opts_.stopping.max_evaluations) {
            stop(Status::MaxEvaluationsReached);
            return true;
        }
        return false;
    }

    [[nodiscard]] std::size_t evaluations_left() const {
        const std::size_t cap = opts_.stopping.max_evaluations;
        return cap == 0 ? std::numeric_limits<std::size_t>::max() : cap - evals_;
    }

    // One evaluation at x into f. False when the cap forbids it.
    bool evaluate(std::span<const double> x, double& f) {
        if (cap_reached()) return false;
        f = eval_(x);
        record(x, f);
        return true;
    }

    // Everything that happens once a value is in memory: the bit check, the
    // count, the trace, the best point seen and the stop-value test.
    void record(std::span<const double> x, const double& f) {
        if (fp_bad(f))
            throw std::runtime_error(
                "flop::nelder_mead: the objective returned a non-finite value");
        ++evals_;
        if (opts_.on_evaluation) {
            Evaluation ev{.index = evals_ - 1, .x = x, .f = f, .constraints = {}};
            opts_.on_evaluation(ev);
        }
        if (!have_best_ || f < best_f_) {
            std::copy(x.begin(), x.end(), best_x_.begin());
            best_f_ = f;
            have_best_ = true;
        }
        if (opts_.stopping.stop_value && f <= *opts_.stopping.stop_value)
            stop(Status::StopValueReached);
    }

    void stop(Status s) {
        if (!done_) {
            done_ = true;
            status_ = s;
        }
    }

    // Evaluates the vertices in slots[0 .. k) into their fval_ entries: one
    // batch when the objective takes one and the cap allows all k, else one
    // at a time until the cap. False when the run stopped part way.
    bool evaluate_slots(std::span<const std::size_t> slots) {
        const std::size_t k = slots.size();
        if (Eval::has_batch && k > 1 && evaluations_left() >= k) {
            for (std::size_t j = 0; j < k; ++j) spans_[j] = vertex(slots[j]);
            const std::span<double> out(fbuf_.data(), k);
            eval_.batch(std::span<const std::span<const double>>(spans_.data(), k), out);
            for (std::size_t j = 0; j < k; ++j) {
                record(spans_[j], out[j]);
                fval_[slots[j]] = out[j];
            }
            return !done_;
        }
        for (std::size_t j = 0; j < k; ++j) {
            if (!evaluate(vertex(slots[j]), fval_[slots[j]])) return false;
            if (done_) return false;
        }
        return true;
    }

    // ================================================================ simplex
    // The vertex in slot 0 and n more, slot i + 1 displaced from it along
    // coordinate i by initial_step, or to the other side or by less where the
    // box requires (box.hpp). Slot 0 is evaluated first, so evaluation index
    // 0 is x0 on both channels. With base_known, slot 0 already holds an
    // evaluated point (a restart) and only the n displaced ones are new.
    void build_simplex(bool base_known) {
        const std::span<const double> base = vertex(0);
        for (std::size_t j = 1; j <= n_; ++j) {
            auto v = vertex(j);
            std::copy(base.begin(), base.end(), v.begin());
            v[j - 1] += axis_offset(bounds_, base, j - 1, opts_.initial_step);
        }
        for (std::size_t j = 0; j <= n_; ++j) order_[j] = j;
        const std::span<const std::size_t> slots(order_.data() + (base_known ? 1 : 0),
                                                 base_known ? n_ : n_ + 1);
        if (!evaluate_slots(slots)) return;
        rank_all();
    }

    // Lagarias et al., section 2: the ordering of a simplex whose vertices
    // all changed rank (the initial simplex, a restart, a shrink). A stable
    // sort keeps the incoming order on ties, and the incoming order has the
    // vertex that was best first, which is the paper's one rule for a shrink:
    // if a new vertex ties with x_1, x_1 stays first.
    void rank_all() {
        std::stable_sort(order_.begin(), order_.end(),
                         [this](std::size_t a, std::size_t b) { return fval_[a] < fval_[b]; });
        rebuild_sum();
        rebuild_extents();
    }

    void rebuild_sum() {
        std::fill(sum_.begin(), sum_.end(), 0.0);
        for (std::size_t s = 0; s <= n_; ++s) {
            const auto v = vertex(s);
            for (std::size_t i = 0; i < n_; ++i) sum_[i] += v[i];
        }
        updates_since_sum_ = 0;
    }

    [[nodiscard]] double distance_to_best(std::size_t slot) const {
        const auto v = vertex(slot);
        const auto b = vertex(order_[0]);
        double d = 0.0;
        for (std::size_t i = 0; i < n_; ++i) d = std::max(d, std::fabs(v[i] - b[i]));
        return d;
    }

    // The exact radius, O(n^2).
    [[nodiscard]] double exact_radius() const {
        double r = 0.0;
        for (std::size_t s = 0; s <= n_; ++s) r = std::max(r, distance_to_best(s));
        return r;
    }

    void rescan_extent(std::size_t i) {
        lo_[i] = hi_[i] = verts_[i];
        arglo_[i] = arghi_[i] = 0;
        for (std::size_t s = 1; s <= n_; ++s) {
            const double v = verts_[s * n_ + i];
            if (v < lo_[i]) {
                lo_[i] = v;
                arglo_[i] = s;
            }
            if (v > hi_[i]) {
                hi_[i] = v;
                arghi_[i] = s;
            }
        }
    }

    void rebuild_extents() {
        for (std::size_t i = 0; i < n_; ++i) rescan_extent(i);
    }

    // Slot `slot` now holds a new vertex. The extremes it takes over or
    // keeps cost nothing; one it held and lost is found by a rescan.
    void update_extents(std::size_t slot) {
        const auto v = vertex(slot);
        for (std::size_t i = 0; i < n_; ++i) {
            const double x = v[i];
            bool rescan = false;
            if (x <= lo_[i]) {
                lo_[i] = x;
                arglo_[i] = slot;
            } else if (arglo_[i] == slot) {
                rescan = true;
            }
            if (x >= hi_[i]) {
                hi_[i] = x;
                arghi_[i] = slot;
            } else if (arghi_[i] == slot) {
                rescan = true;
            }
            if (rescan) rescan_extent(i);
        }
    }

    [[nodiscard]] double diameter() const {
        double d = 0.0;
        for (std::size_t i = 0; i < n_; ++i) d = std::max(d, hi_[i] - lo_[i]);
        return d;
    }

    // Lagarias et al., section 2, after a nonshrink step: the worst vertex is
    // discarded and the accepted point v takes rank j, the first rank among
    // the n kept vertices whose value is strictly above f(v). On a tie with a
    // kept vertex, the kept vertex ranks first.
    void accept(std::span<const double> x, double f) {
        const std::size_t slot = order_[n_];
        auto w = vertex(slot);
        for (std::size_t i = 0; i < n_; ++i) {
            sum_[i] += x[i] - w[i];
            w[i] = x[i];
        }
        fval_[slot] = f;
        const auto kept_end = order_.begin() + static_cast<std::ptrdiff_t>(n_);
        const auto pos = std::upper_bound(order_.begin(), kept_end, f,
                                          [this](double v, std::size_t s) { return v < fval_[s]; });
        std::move_backward(pos, kept_end, kept_end + 1);
        *pos = slot;
        if (++updates_since_sum_ > n_) rebuild_sum();
        update_extents(slot);
    }

    // Lagarias et al., section 2, step 5: every vertex but the best moves to
    // x_1 + sigma (x_i - x_1), and the n new points are evaluated.
    void shrink() {
        const auto b = vertex(order_[0]);
        for (std::size_t r = 1; r <= n_; ++r) {
            auto v = vertex(order_[r]);
            for (std::size_t i = 0; i < n_; ++i) v[i] = b[i] + coef_.sigma * (v[i] - b[i]);
        }
        if (!evaluate_slots(std::span<const std::size_t>(order_.data() + 1, n_))) return;
        rank_all();
    }

    // ============================================================ iteration
    // The trial point xbar + t * dir_, projected onto the box, into out.
    void trial(double t, std::span<double> out) const {
        for (std::size_t i = 0; i < n_; ++i) out[i] = xbar_[i] + t * dir_[i];
        if (bounds_) project_onto_box(*bounds_, out);
    }

    // Lagarias et al., section 2, steps 2 to 5. The four trial points are
    // all on the line through the centroid of the best n and the worst
    // vertex, xbar + t (xbar - x_{n+1}), at t = rho (reflection), rho chi
    // (expansion), rho gamma (outside contraction) and -gamma (inside
    // contraction), which is the paper's closed form of each.
    void iterate() {
        if (should_stop()) return;
        const std::size_t worst = order_[n_];
        const double f1 = fval_[order_[0]];
        const double fn = fval_[order_[n_ - 1]];  // at n = 1, the best vertex
        const double fw = fval_[worst];
        const auto xw = vertex(worst);
        for (std::size_t i = 0; i < n_; ++i) {
            xbar_[i] = (sum_[i] - xw[i]) * inv_n_;
            dir_[i] = xbar_[i] - xw[i];
        }

        double fr = 0.0;
        trial(coef_.rho, xr_);
        if (!evaluate(xr_, fr) || done_) return;
        if (f1 <= fr && fr < fn) {
            accept(xr_, fr);
            return;
        }
        if (fr < f1) {
            // Expand, and keep whichever of the two points is better: the
            // paper's rule compares f_e with f_r, not with f_1.
            double fe = 0.0;
            trial(coef_.rho * coef_.chi, xt_);
            if (!evaluate(xt_, fe) || done_) return;
            if (fe < fr)
                accept(xt_, fe);
            else
                accept(xr_, fr);
            return;
        }
        double fc = 0.0;
        if (fr < fw) {
            trial(coef_.rho * coef_.gamma, xt_);
            if (!evaluate(xt_, fc) || done_) return;
            if (fc <= fr) {
                accept(xt_, fc);
                return;
            }
        } else {
            trial(-coef_.gamma, xt_);
            if (!evaluate(xt_, fc) || done_) return;
            if (fc < fw) {
                accept(xt_, fc);
                return;
            }
        }
        shrink();
    }

    // ============================================================ stopping
    // The stopping tests, on the ordered simplex at the top of an iteration.
    // The radius is the largest coordinate distance from a vertex to the best
    // one; the spread is f_{n+1} - f_1. The radius has a floor at the
    // precision of the coordinates: below it the vertices cannot be told
    // apart, and reaching it without a caller's tolerance is RoundoffLimited.
    bool should_stop() {
        const double diam = diameter();
        bool have_radius = false;
        // r <= t, deciding from the diameter where it can (see the header).
        auto radius_within = [&](double t) {
            if (diam <= t) return true;
            if (diam > 2.0 * t) return false;
            if (!have_radius) {
                radius_ = exact_radius();
                have_radius = true;
            }
            return radius_ <= t;
        };
        const Stopping& s = opts_.stopping;
        const double f1 = fval_[order_[0]];
        const double spread = fval_[order_[n_]] - f1;
        std::optional<Status> verdict;
        if ((s.xtol_abs > 0.0 && radius_within(s.xtol_abs)) ||
            (s.xtol_rel > 0.0 && radius_within(s.xtol_rel * opts_.initial_step)))
            verdict = Status::XtolReached;
        else if ((s.ftol_abs > 0.0 && spread <= s.ftol_abs) ||
                 (s.ftol_rel > 0.0 && spread <= s.ftol_rel * std::fabs(f1)))
            verdict = Status::FtolReached;
        else if (radius_within(precision_floor()))
            verdict = Status::RoundoffLimited;
        if (!verdict) return false;
        radius_ = exact_radius();
        if (*verdict != Status::RoundoffLimited && bounds_ && left_a_face()) return true;
        stop(*verdict);
        return true;
    }

    [[nodiscard]] double precision_floor() const {
        double scale = opts_.initial_step;
        for (const double v : vertex(order_[0])) scale = std::max(scale, std::fabs(v));
        return scale * std::numeric_limits<double>::epsilon();
    }

    // The face test before a tolerance stop on a box. For each coordinate in
    // which every vertex sits on the same bound, one point: the best vertex
    // moved off that bound by the radius (by less where the other bound is
    // closer). True when the run did something instead of stopping: it
    // found a better point off a face and restarted from it, or the cap or
    // the stop value ended it during the test. False when every face the
    // simplex lies on held, and the stop stands.
    bool left_a_face() {
        const std::size_t best = order_[0];
        const double h = std::max(radius_, precision_floor());
        for (std::size_t i = 0; i < n_; ++i) {
            const double xi = vertex(best)[i];
            const std::optional<double>& lo = bounds_->lower[i];
            const std::optional<double>& hi = bounds_->upper[i];
            const bool at_lo = lo.has_value() && xi == *lo;
            const bool at_hi = hi.has_value() && xi == *hi;
            if (!at_lo && !at_hi) continue;
            bool flat = true;
            for (std::size_t s = 0; s <= n_ && flat; ++s) flat = vertex(s)[i] == xi;
            if (!flat) continue;
            const auto b = vertex(best);
            std::copy(b.begin(), b.end(), xt_.begin());
            if (at_lo)
                xt_[i] = hi.has_value() ? std::min(xi + h, *hi) : xi + h;
            else
                xt_[i] = lo.has_value() ? std::max(xi - h, *lo) : xi - h;
            double f = 0.0;
            if (!evaluate(xt_, f) || done_) return true;
            if (f < fval_[best]) {
                auto v0 = vertex(0);
                std::copy(xt_.begin(), xt_.end(), v0.begin());
                fval_[0] = f;
                build_simplex(true);
                return true;
            }
        }
        return false;
    }

    Result finish() {
        Result r;
        r.x = best_x_;
        r.f = best_f_;
        r.evaluations = evals_;
        r.status = status_;
        r.final_radius = exact_radius();
        r.max_constraint_violation = 0.0;
        return r;
    }

    Eval eval_;
    std::size_t n_;
    const flop::nelder_mead::Options& opts_;
    const Bounds* bounds_;
    Coefficients coef_;
    double inv_n_;

    std::vector<double> verts_;               // slot s holds verts_[s * n, (s + 1) * n)
    std::vector<double> fval_;                // f at each slot
    std::vector<std::size_t> order_;          // rank -> slot, rank 0 best
    std::vector<double> lo_, hi_;             // per coordinate, the least and greatest vertex value
    std::vector<std::size_t> arglo_, arghi_;  // the slot holding each
    std::vector<double> sum_;                 // sum of all n + 1 vertices
    std::size_t updates_since_sum_ = 0;
    std::vector<double> dir_, xbar_, xr_, xt_, fbuf_;
    std::vector<std::span<const double>> spans_;

    std::vector<double> best_x_;
    double best_f_ = 0.0;
    bool have_best_ = false;
    double radius_ = 0.0;  // the exact radius, as of the last stopping test that needed it
    std::size_t evals_ = 0;
    bool done_ = false;
    Status status_ = Status::RoundoffLimited;
};

}  // namespace flop::detail::nelder_mead
