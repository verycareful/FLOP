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
// A shrink is O(n^2), and its new points are independent, so they go out as
// one batch when the objective takes one; so do the initial simplex, a
// restart's simplex and each rung of the ladder below.
//
// Box bounds: no paper defines Nelder-Mead on a box. Every trial point is
// projected onto the box, so every evaluation is inside it and a bound can
// be reached exactly. Projection can also flatten the simplex against a
// face or collapse it onto a corner, where it stops without being near a
// minimiser, so once the box has acted on the run (a trial point projected,
// or an initial vertex moved by a bound) the poll ladder runs before any
// stop (see ladder()): coordinate polls at the initial step and every
// halving of it down to the scale of the stop. A better poll point restarts
// the method from it. A box that never binds changes nothing.
//
// No random state, no infinity, no NaN. Every objective value is checked by
// its bits the moment it lands in memory, and every point is checked against
// a range (check_range) inside which none of the arithmetic can overflow.
// The arithmetic is affine combinations and comparisons; none of it depends
// on the floating-point model for anything but rounding.

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
#include "flop/detail/tolerance.hpp"
#include "flop/detail/validate.hpp"
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

// Gao and Han, section 4.1, formula (4.1), for n >= 2; the standard values
// otherwise.
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
          inv_n_(1.0 / static_cast<double>(x0.size())),
          range_limit_(nelder_mead_range_limit(x0.size())) {
        verts_.assign((n_ + 1) * n_, 0.0);
        fval_.assign(n_ + 1, 0.0);
        order_.assign(n_ + 1, 0);
        moved_.assign(n_, 0);
        lo_.assign(n_, 0.0);
        hi_.assign(n_, 0.0);
        arglo_.assign(n_, 0);
        arghi_.assign(n_, 0);
        sum_.assign(n_, 0.0);
        dir_.assign(n_, 0.0);
        xbar_.assign(n_, 0.0);
        xr_.assign(n_, 0.0);
        xt_.assign(n_, 0.0);
        // n + 1 points for the initial simplex, 2n for a rung of the ladder.
        fbuf_.assign(std::max(n_ + 1, 2 * n_), 0.0);
        spans_.resize(std::max(n_ + 1, 2 * n_));
        poll_.assign(2 * n_ * n_, 0.0);
        best_x_.assign(x0.begin(), x0.end());
        complete_radius_ = opts.initial_step;
        for (std::size_t i = 0; i < n_; ++i) verts_[i] = x0[i];
    }

    Result run() {
        build_simplex(false, opts_.initial_step);
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

    // No point is evaluated with a coordinate of magnitude above
    // range_limit_ = max / (n + 5) (validate.hpp). Every vertex is an
    // evaluated point, so every vertex is inside that range, and then the
    // running sum ((n + 1) vertices, plus one update of at most 2 limits),
    // the centroid, a trial point (at most 5 limits: the centroid plus 2 times
    // a direction of at most 2) and a shrunk vertex (at most 3) are all
    // finite. A point outside it ends the run instead: the objective is most
    // likely unbounded below along a coordinate without a bound.
    void check_range(std::span<const double> x) const {
        for (const double& v : x)
            if (std::fabs(v) > range_limit_) out_of_range();
    }

    [[noreturn]] static void out_of_range() {
        throw std::runtime_error(
            "flop::nelder_mead: a point to evaluate has a coordinate beyond DBL_MAX / (n + 5), the "
            "range in which the method's arithmetic stays finite; the objective may be unbounded "
            "below");
    }

    // One evaluation at x into f. False when the cap forbids it.
    bool evaluate(std::span<const double> x, double& f) {
        check_range(x);
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
            std::ranges::copy(x, best_x_.begin());
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

    // Evaluates the k independent points in spans_[0 .. k) into fbuf_[0 ..
    // k): one batch when the objective takes one and the cap allows all k,
    // else one at a time until the cap. Every point is range-checked before
    // any is evaluated, so both channels throw at the same moment. True when
    // every one of them was evaluated, which a batch always is, even when one
    // of its points met stop_value; the run may have ended either way.
    bool evaluate_spans(std::size_t k) {
        for (std::size_t j = 0; j < k; ++j) check_range(spans_[j]);
        if (Eval::has_batch && k > 1 && evaluations_left() >= k) {
            const std::span<double> out(fbuf_.data(), k);
            eval_.batch(std::span<const std::span<const double>>(spans_.data(), k), out);
            for (std::size_t j = 0; j < k; ++j) record(spans_[j], out[j]);
            return true;
        }
        for (std::size_t j = 0; j < k; ++j) {
            if (!evaluate(spans_[j], fbuf_[j])) return false;
            if (done_ && j + 1 < k) return false;
        }
        return true;
    }

    // The vertices in slots, evaluated as above into their fval_ entries.
    bool evaluate_slots(std::span<const std::size_t> slots) {
        const std::size_t k = slots.size();
        for (std::size_t j = 0; j < k; ++j) spans_[j] = vertex(slots[j]);
        if (!evaluate_spans(k)) return false;
        for (std::size_t j = 0; j < k; ++j) fval_[slots[j]] = fbuf_[j];
        return true;
    }

    // The simplex, whose radius is r, is about to have vertices placed that
    // are not yet evaluated. r is what Result::final_radius reports if the
    // run ends before they all are.
    void leave_complete_simplex(double r) {
        complete_radius_ = r;
        complete_ = false;
    }

    // Every vertex has its value: rank them, and the simplex is the one
    // final_radius describes.
    void enter_complete_simplex() {
        rank_all();
        complete_ = true;
    }

    // ================================================================ simplex
    // The vertex in slot 0 and n more, slot i + 1 displaced from it along
    // coordinate i by the step h, or to the other side or to a bound where
    // the box requires (box.hpp). Slot 0 is evaluated first, so evaluation
    // index 0 is x0 on both channels. With base_known, slot 0 already holds
    // an evaluated point (a restart) and only the n displaced ones are new.
    // The caller guarantees |x_i| + h <= max for the base (validation for
    // x0, step_fits for a restart).
    void build_simplex(bool base_known, double h) {
        const std::span<const double> base = vertex(0);
        for (std::size_t j = 1; j <= n_; ++j) {
            auto v = vertex(j);
            std::ranges::copy(base, v.begin());
            v[j - 1] = axis_vertex(bounds_, base, j - 1, h);
            if (v[j - 1] != base[j - 1] + h) box_acted_ = true;
        }
        for (std::size_t j = 0; j <= n_; ++j) order_[j] = j;
        const std::span<const std::size_t> slots(order_.data() + (base_known ? 1 : 0),
                                                 base_known ? n_ : n_ + 1);
        if (evaluate_slots(slots)) enter_complete_simplex();
    }

    // Lagarias et al., section 2: the ordering of a simplex whose vertices
    // all changed rank (the initial simplex, a restart, a shrink). A stable
    // sort keeps the incoming order on ties, and the incoming order has the
    // vertex that was best first, which is the paper's one rule for a shrink:
    // if a new vertex ties with x_1, x_1 stays first.
    void rank_all() {
        std::ranges::stable_sort(
            order_, [this](std::size_t a, std::size_t b) { return fval_[a] < fval_[b]; });
        rebuild_sum();
        rebuild_extents();
    }

    void rebuild_sum() {
        std::ranges::fill(sum_, 0.0);
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
    // x_1 + sigma (x_i - x_1), and the new points are evaluated.
    //
    // In rounded arithmetic a vertex need not move: one d units in the last
    // place from x_1 lands back on itself whenever sigma d rounds to d, which
    // for sigma = 1 - 1/n is every d below about n / 2. A coordinate that
    // compares equal keeps its bits, and a vertex none of whose coordinates
    // changed keeps its value without being evaluated again, so only the
    // vertices that moved are evaluated. A shrink that moves no vertex would
    // leave the simplex exactly as it was, and the method would repeat it
    // forever; that is the precision of the arithmetic reached, and the run
    // stops RoundoffLimited without evaluating anything.
    void shrink() {
        const double r = exact_radius();
        const auto b = vertex(order_[0]);
        std::size_t moved = 0;
        for (std::size_t k = 1; k <= n_; ++k) {
            auto v = vertex(order_[k]);
            bool changed = false;
            for (std::size_t i = 0; i < n_; ++i) {
                const double x = b[i] + coef_.sigma * (v[i] - b[i]);
                if (x != v[i]) {
                    v[i] = x;
                    changed = true;
                }
            }
            if (changed) moved_[moved++] = order_[k];
        }
        if (moved == 0) {
            conclude(Status::RoundoffLimited);
            return;
        }
        leave_complete_simplex(r);
        if (evaluate_slots(std::span<const std::size_t>(moved_.data(), moved)))
            enter_complete_simplex();
    }

    // ============================================================ iteration
    // The trial point xbar + t * dir_, projected onto the box, into out.
    void trial(double t, std::span<double> out) {
        for (std::size_t i = 0; i < n_; ++i) out[i] = xbar_[i] + t * dir_[i];
        if (bounds_ && project_onto_box(*bounds_, out)) box_acted_ = true;
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
        const double& f1 = fval_[order_[0]];
        const double& fw = fval_[order_[n_]];
        std::optional<Status> verdict;
        if ((s.xtol_abs > 0.0 && radius_within(s.xtol_abs)) ||
            (s.xtol_rel > 0.0 && radius_within(s.xtol_rel * opts_.initial_step)))
            verdict = Status::XtolReached;
        else if ((s.ftol_abs > 0.0 && difference_within_abs(fw, f1, s.ftol_abs)) ||
                 (s.ftol_rel > 0.0 && difference_within_rel(fw, f1, s.ftol_rel)))
            verdict = Status::FtolReached;
        else if (radius_within(precision_floor()))
            verdict = Status::RoundoffLimited;
        if (!verdict) return false;
        conclude(*verdict);
        return true;
    }

    // Every stop verdict ends here, whichever test reached it. Once the box
    // has acted on the run, the ladder runs first and may restart the method
    // instead. Until then the run is the unbounded method exactly, a box that
    // never binds changes nothing, and the stop needs no more than it would
    // without one.
    void conclude(Status s) {
        radius_ = exact_radius();
        if (box_acted_ && ladder(ladder_bottom(s))) return;
        stop(s);
    }

    // The smallest rung of the ladder: the scale at which the stop was
    // reached. For XtolReached the x tolerance the caller accepts (the larger
    // of the two when both are set); otherwise the radius, or the precision
    // floor where the radius is below it.
    [[nodiscard]] double ladder_bottom(Status s) const {
        if (s == Status::XtolReached) {
            const Stopping& st = opts_.stopping;
            double t = 0.0;
            if (st.xtol_abs > 0.0) t = st.xtol_abs;
            if (st.xtol_rel > 0.0) t = std::max(t, st.xtol_rel * opts_.initial_step);
            return t;
        }
        return std::max(radius_, precision_floor());
    }

    [[nodiscard]] double precision_floor() const {
        double scale = opts_.initial_step;
        for (const double v : vertex(order_[0])) scale = std::max(scale, std::fabs(v));
        return scale * std::numeric_limits<double>::epsilon();
    }

    // ============================================================== ladder
    // The poll ladder, before any stop once the box has acted. Projection can
    // flatten the simplex against a face, or collapse it onto a corner,
    // where the method can no longer move; a stop reported there need not
    // be near a minimiser. The ladder is the poll step of coordinate search
    // on a box in R. M. Lewis and V. Torczon, "Pattern search algorithms for
    // bound constrained minimization", SIAM Journal on Optimization 9(4),
    // 1999, pp. 1082-1099: the directions +-e_i (section 6.1, B = I and
    // M = I, the diagonal core pattern section 3.5 requires near a bound),
    // the step halved after a failed poll (Fig. 3.3 with theta = 1/2), and
    // the best point of the whole poll taken (the strong hypothesis, Fig.
    // 4.1). Two differences: their poll leaves out a point outside the box
    // (Fig. 3.1), where the ladder clips it onto the bound, which only adds
    // candidates, since every point of their poll inside the box is still
    // polled; and here the polls certify a stop rather than drive the search,
    // so an improvement restarts Nelder-Mead and a stop stands only when no
    // rung down to the stop's scale improves on the best vertex.
    //
    // Rung by rung, D = initial_step, D / 2, ... while D >= bottom: the 2n
    // points b + D e_i and b - D e_i, b the best vertex, each clipped to the
    // box, in coordinate order and + before -; a point clipped back onto b is
    // b and is skipped. The whole rung is evaluated, as one call of up to 2n
    // points with a batch objective, so both channels evaluate the same
    // points. If the first point of least value in the rung is strictly
    // below f(b), the method starts again from it with a fresh simplex of
    // size D. Each restart starts from a strictly lower value, so there are
    // finitely many.
    //
    // True when the run did something instead of stopping: it restarted, or
    // the cap or the stop value ended it during the ladder. False when every
    // rung failed, and the stop stands.
    bool ladder(double bottom) {
        const std::size_t best = order_[0];
        const std::span<const double> b = vertex(best);
        const double fb = fval_[best];
        double d = opts_.initial_step;
        while (d >= bottom) {
            const std::size_t k = fill_rung(b, d);
            if (k > 0) {
                if (!evaluate_spans(k) || done_) return true;
                std::size_t arg = k;
                for (std::size_t j = 0; j < k; ++j)
                    if (fbuf_[j] < fb && (arg == k || fbuf_[j] < fbuf_[arg])) arg = j;
                if (arg != k) {
                    restart_from(spans_[arg], fbuf_[arg], d);
                    return true;
                }
            }
            d *= 0.5;
        }
        return false;
    }

    // The rung at step d around b into poll_, with spans_[0 .. k) pointing at
    // its points; returns k.
    std::size_t fill_rung(std::span<const double> b, double d) {
        std::size_t k = 0;
        for (std::size_t i = 0; i < n_; ++i) {
            for (const double sign : {1.0, -1.0}) {
                double c = 0.0;
                if (!poll_coordinate(b[i], sign, d, i, c)) continue;
                const std::span<double> p(poll_.data() + k * n_, n_);
                std::ranges::copy(b, p.begin());
                p[i] = c;
                spans_[k] = p;
                ++k;
            }
        }
        return k;
    }

    // Coordinate i of a poll point, b_i + sign d clipped to the box, into c.
    // False when that is b_i itself. The sum leaves the finite range only on
    // the side of b_i's sign and only when d > max - |b_i|; it is then past
    // any bound on that side, so the clipped coordinate is that bound, and
    // without one the point is out of range.
    bool poll_coordinate(const double& bi, double sign, double d, std::size_t i, double& c) const {
        const std::optional<double>& bound = sign > 0.0 ? bounds_->upper[i] : bounds_->lower[i];
        const bool away = sign > 0.0 ? bi >= 0.0 : bi <= 0.0;
        if (away && d > kMaxFinite - std::fabs(bi)) {
            if (!bound.has_value()) out_of_range();
            c = *bound;
        } else {
            c = bi + sign * d;
            if (bound.has_value() && (sign > 0.0 ? c > *bound : c < *bound)) c = *bound;
        }
        return c != bi;
    }

    // A restart from the evaluated point x, value f, with a fresh simplex of
    // size h. The simplex computes x_i + h and x_i - h (box.hpp), so both
    // must be finite.
    void restart_from(std::span<const double> x, double f, double h) {
        for (const double& v : x)
            if (h > kMaxFinite - std::fabs(v)) out_of_range();
        leave_complete_simplex(radius_);
        std::ranges::copy(x, vertex(0).begin());
        fval_[0] = f;
        build_simplex(true, h);
    }

    Result finish() {
        Result r;
        r.x = best_x_;
        r.f = best_f_;
        r.evaluations = evals_;
        r.status = status_;
        r.final_radius = complete_ ? exact_radius() : complete_radius_;
        r.max_constraint_violation = 0.0;
        return r;
    }

    Eval eval_;
    std::size_t n_;
    const flop::nelder_mead::Options& opts_;
    const Bounds* bounds_;
    Coefficients coef_;
    double inv_n_;
    double range_limit_;  // no evaluated coordinate is larger in magnitude

    std::vector<double> verts_;               // slot s holds verts_[s * n, (s + 1) * n)
    std::vector<double> fval_;                // f at each slot
    std::vector<std::size_t> order_;          // rank -> slot, rank 0 best
    std::vector<double> lo_, hi_;             // per coordinate, the least and greatest vertex value
    std::vector<std::size_t> arglo_, arghi_;  // the slot holding each
    std::vector<double> sum_;                 // sum of all n + 1 vertices
    std::vector<std::size_t> moved_;          // the slots a shrink moved
    std::size_t updates_since_sum_ = 0;
    std::vector<double> dir_, xbar_, xr_, xt_, fbuf_;
    std::vector<double> poll_;  // a rung of the ladder: up to 2n points of n coordinates
    std::vector<std::span<const double>> spans_;

    std::vector<double> best_x_;
    double best_f_ = 0.0;
    bool have_best_ = false;
    double radius_ = 0.0;  // the exact radius, as of the last stopping test that needed it
    // The box has acted on the run: projection moved a trial point, or an
    // initial vertex is not x + h. Set once, never cleared.
    bool box_acted_ = false;
    // Result::final_radius: the radius of the last simplex whose every vertex
    // was evaluated, or initial_step while the first one is incomplete.
    bool complete_ = false;
    double complete_radius_ = 0.0;
    std::size_t evals_ = 0;
    bool done_ = false;
    Status status_ = Status::RoundoffLimited;
};

}  // namespace flop::detail::nelder_mead
