// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// A direct transcription of Algorithm NM, Lagarias, Reeds, Wright and
// Wright, SIAM J. Optim. 9(1), 1998, section 2.1, pp. 115-117, written for
// the test suite and nothing else. It is slow on purpose: the simplex is
// kept as an ordered list of points with their values, the centroid is
// summed from scratch every iteration, the radius is computed exactly at
// every stopping test, and the two tie-breaking rules are applied as the
// paper states them rather than through any index structure. What it shares
// with FLOP is only what the paper leaves open and FLOP documents
// (docs/algorithms/nelder-mead.md, "Deviations from the paper"): the initial
// simplex, the projection onto a box, the stopping rules, the shrink that
// moves no vertex, and the poll ladder.
//
// Where the 0.1.1.1 suite settled what FLOP must do, the reference does
// that, and a test that compares FLOP with it on such a path is red until
// 0.1.1.2:
//   - the face test runs before every stop verdict, RoundoffLimited
//     included;
//   - Result::final_radius is the radius of the last simplex whose every
//     vertex was evaluated, and initial_step while the first one is
//     incomplete;
//   - an initial vertex whose step is shrunk to the room left in the box
//     lands on the bound itself.
//
// The nonshrink ordering rule is printed on p. 116 as
// j = max{ l | f(v) < f(x_{l+1}) }, 0 <= l <= n, which taken literally is
// always n: every accepted point is strictly better than x_{n+1}. The text
// above it ("the highest possible index consistent with the relation
// f(x_1) <= ... <= f(x_{n+1})") and the worked example on p. 118 ((1, 2, 2,
// 3, 3) with f(v) = 2 gives (1, 2, 2, 2, 3), x_4 = v) both define the
// smallest such l, so the new vertex goes after every kept vertex of equal
// value. That is the reading transcribed here.
//
// Every evaluation goes through an oracle, std::optional<double>(x, scale),
// called with the point the reference wants and the magnitude of the
// coordinates it was computed from. An oracle that is the objective runs the
// reference on its own. An oracle that replays a FLOP trace (Audit, below)
// checks each point against FLOP's, then hands back FLOP's point and value,
// so a long run is compared step by step without the two trajectories
// drifting apart by rounding; nullopt ends the run as "ran out".

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "flop/flop.hpp"
#include "v0111_problems.hpp"

namespace v0111 {

// The step of Algorithm NM in which an iteration terminated (section 2.1),
// or a restart from the poll ladder.
enum class Termination : std::uint8_t {
    Reflect = 2,
    Expand = 3,
    Contract = 4,
    Shrink = 5,
    Restart = 6
};

struct Vertex {
    std::vector<double> x;
    double f = 0.0;
};
using Ordered = std::vector<Vertex>;  // x_1 first

// One completed iteration, with the ordered simplex before and after it.
struct Step {
    Termination termination;
    double tau = 0.0;  // the accepted trial point's coefficient in (2.12), 0 for a shrink
    Ordered before;
    Ordered after;
    std::size_t first_evaluation = 0;  // index of the iteration's first evaluation
    std::size_t evaluations = 0;
};

struct ReferenceRun {
    flop::Result result;
    bool ran_out = false;
    std::vector<Step> steps;         // filled when recording
    std::vector<double> top_radius;  // the exact radius at every stopping test
    std::size_t shrinks = 0;
    std::size_t still_shrinks = 0;  // shrinks that moved no vertex, each a stop verdict
    std::size_t restarts = 0;
};

class Reference {
public:
    Reference(std::span<const double> x0, const flop::nelder_mead::Options& opts,
              bool record = false)
        : x0_(x0.begin(), x0.end()),
          n_(x0.size()),
          opts_(opts),
          coef_(documented(x0.size(), opts.adaptive_coefficients)),
          record_(record) {}

    template <class Oracle>
    ReferenceRun run(Oracle& oracle) {
        run_ = ReferenceRun{};
        evals_ = 0;
        ended_ = false;
        have_best_ = false;
        complete_radius_.reset();
        box_acted_ = false;
        status_ = flop::Status::RoundoffLimited;
        if (build(oracle, x0_, std::nullopt, opts_.initial_step)) {
            while (!ended_) {
                if (stop_test(oracle)) continue;
                if (ended_) break;
                iterate(oracle);
            }
        }
        run_.result.x = best_x_;
        run_.result.f = best_f_;
        run_.result.evaluations = evals_;
        run_.result.status = status_;
        run_.result.final_radius = complete_radius_ ? *complete_radius_ : opts_.initial_step;
        run_.result.max_constraint_violation = 0.0;
        return run_;
    }

private:
    // ============================================================ evaluation
    // False when the run ended before or at this evaluation: the cap, the
    // stop value, or an oracle that ran out.
    template <class Oracle>
    bool evaluate(Oracle& oracle, std::vector<double>& x, double& f) {
        const std::size_t cap = opts_.stopping.max_evaluations;
        if (cap > 0 && evals_ >= cap) {
            end(flop::Status::MaxEvaluationsReached);
            return false;
        }
        const std::optional<double> v = oracle(std::span<double>(x), scale_of(x));
        if (!v) {
            run_.ran_out = true;
            ended_ = true;
            return false;
        }
        f = *v;
        ++evals_;
        if (!have_best_ || f < best_f_) {
            best_x_ = x;
            best_f_ = f;
            have_best_ = true;
        }
        if (opts_.stopping.stop_value && f <= *opts_.stopping.stop_value) {
            end(flop::Status::StopValueReached);
            return false;
        }
        return true;
    }

    void end(flop::Status s) {
        if (!ended_) {
            ended_ = true;
            status_ = s;
        }
    }

    // The magnitude the next point was computed from: the largest coordinate
    // of any vertex or of the point itself.
    [[nodiscard]] double scale_of(const std::vector<double>& x) const {
        double s = 0.0;
        for (const double v : x) s = std::max(s, std::fabs(v));
        for (const Vertex& v : simplex_)
            for (const double c : v.x) s = std::max(s, std::fabs(c));
        return s;
    }

    // ========================================================== the simplex
    // The documented initial vertex along coordinate i for step h: base + h
    // when that point is in the box, else base - h when that one is, else
    // the bound on the side with more room.
    [[nodiscard]] double offset_vertex(const std::vector<double>& base, std::size_t i,
                                       double h) const {
        const double up = base[i] + h;
        if (!opts_.bounds) return up;
        const std::optional<double>& lo = opts_.bounds->lower[i];
        const std::optional<double>& hi = opts_.bounds->upper[i];
        if (!hi.has_value() || up <= *hi) return up;
        const double down = base[i] - h;
        if (!lo.has_value() || down >= *lo) return down;
        return *hi - base[i] >= base[i] - *lo ? *hi : *lo;
    }

    // The initial simplex, or a restart's: base and the n displaced points
    // at step h, evaluated base first (unless known) and then in coordinate
    // order, and ordered stably, so ties keep construction order.
    template <class Oracle>
    bool build(Oracle& oracle, const std::vector<double>& base, std::optional<double> base_f,
               double h) {
        Ordered fresh;
        fresh.push_back({.x = base, .f = base_f.value_or(0.0)});
        if (!base_f && !evaluate(oracle, fresh[0].x, fresh[0].f)) return false;
        for (std::size_t i = 0; i < n_; ++i) {
            Vertex v{.x = base, .f = 0.0};
            v.x[i] = offset_vertex(base, i, h);
            if (v.x[i] != base[i] + h) box_acted_ = true;
            if (!evaluate(oracle, v.x, v.f)) return false;
            fresh.push_back(std::move(v));
        }
        std::ranges::stable_sort(fresh, [](const Vertex& a, const Vertex& b) { return a.f < b.f; });
        simplex_ = std::move(fresh);
        return true;
    }

    [[nodiscard]] double radius() const {
        double r = 0.0;
        for (const Vertex& v : simplex_)
            for (std::size_t i = 0; i < n_; ++i)
                r = std::max(r, std::fabs(v.x[i] - simplex_[0].x[i]));
        return r;
    }

    [[nodiscard]] double precision_floor() const {
        double scale = opts_.initial_step;
        for (const double v : simplex_[0].x) scale = std::max(scale, std::fabs(v));
        return scale * std::numeric_limits<double>::epsilon();
    }

    // ============================================================= stopping
    // The documented stopping rules at the top of an iteration, on the exact
    // radius. True when the run did something other than iterate: it
    // stopped, or the ladder restarted it.
    template <class Oracle>
    bool stop_test(Oracle& oracle) {
        const double r = radius();
        complete_radius_ = r;
        run_.top_radius.push_back(r);
        const flop::Stopping& s = opts_.stopping;
        const double f1 = simplex_.front().f;
        const double spread = simplex_.back().f - f1;
        std::optional<flop::Status> verdict;
        if ((s.xtol_abs > 0.0 && r <= s.xtol_abs) ||
            (s.xtol_rel > 0.0 && r <= s.xtol_rel * opts_.initial_step))
            verdict = flop::Status::XtolReached;
        else if ((s.ftol_abs > 0.0 && spread <= s.ftol_abs) ||
                 (s.ftol_rel > 0.0 && spread <= s.ftol_rel * std::fabs(f1)))
            verdict = flop::Status::FtolReached;
        else if (r <= precision_floor())
            verdict = flop::Status::RoundoffLimited;
        if (!verdict) return false;
        conclude(oracle, *verdict, r);
        return true;
    }

    // A stop verdict, from the tests above or from a shrink that moved no
    // vertex: once the box has acted on the run (a projected trial point, an
    // initial vertex other than base + h), the ladder first, then the stop.
    template <class Oracle>
    void conclude(Oracle& oracle, flop::Status verdict, double r) {
        if (box_acted_ && opts_.bounds && ladder(oracle, *opts_.bounds, bottom(verdict, r))) return;
        end(verdict);
    }

    // The smallest rung: the caller's x tolerance for XtolReached (the larger
    // of the two when both are set), else max(radius, floor).
    [[nodiscard]] double bottom(flop::Status verdict, double r) const {
        if (verdict == flop::Status::XtolReached) {
            const flop::Stopping& s = opts_.stopping;
            double t = 0.0;
            if (s.xtol_abs > 0.0) t = s.xtol_abs;
            if (s.xtol_rel > 0.0) t = std::max(t, s.xtol_rel * opts_.initial_step);
            return t;
        }
        return std::max(r, precision_floor());
    }

    // The poll ladder: for d = initial_step, d / 2, ... while d >= bottom,
    // the points x_1 +- d e_i clipped to the box, coordinate by coordinate
    // and + before -, a point clipped back onto x_1 left out; the whole rung
    // is evaluated, and the first point of least value restarts the method
    // with a fresh simplex of size d when it is strictly below f(x_1). True
    // when the run restarted or ended here.
    template <class Oracle>
    bool ladder(Oracle& oracle, const flop::Bounds& bounds, double bottom) {
        const Vertex best = simplex_.front();
        double d = opts_.initial_step;
        while (d >= bottom) {
            std::vector<Vertex> rung;
            for (std::size_t i = 0; i < n_; ++i) {
                for (const double sign : {1.0, -1.0}) {
                    Vertex p = best;
                    p.x[i] = best.x[i] + sign * d;
                    const std::optional<double>& bound =
                        sign > 0.0 ? bounds.upper[i] : bounds.lower[i];
                    if (bound.has_value() && (sign > 0.0 ? p.x[i] > *bound : p.x[i] < *bound))
                        p.x[i] = *bound;
                    if (p.x[i] == best.x[i]) continue;
                    rung.push_back(std::move(p));
                }
            }
            if (!rung.empty()) {
                const std::size_t first = evals_;
                for (Vertex& p : rung)
                    if (!evaluate(oracle, p.x, p.f)) return true;
                std::optional<std::size_t> arg;
                for (std::size_t j = 0; j < rung.size(); ++j)
                    if (rung[j].f < best.f && (!arg || rung[j].f < rung[*arg].f)) arg = j;
                if (arg) {
                    const Ordered before = simplex_;
                    if (!build(oracle, rung[*arg].x, rung[*arg].f, d)) return true;
                    ++run_.restarts;
                    note(Termination::Restart, 0.0, before, first);
                    return true;
                }
            }
            d *= 0.5;
        }
        return false;
    }

    // ============================================================ iteration
    void project(std::vector<double>& x) {
        if (!opts_.bounds) return;
        for (std::size_t i = 0; i < n_; ++i) {
            const std::optional<double>& lo = opts_.bounds->lower[i];
            const std::optional<double>& hi = opts_.bounds->upper[i];
            if (lo.has_value() && x[i] < *lo) {
                x[i] = *lo;
                box_acted_ = true;
            }
            if (hi.has_value() && x[i] > *hi) {
                x[i] = *hi;
                box_acted_ = true;
            }
        }
    }

    // (2.12), summed from scratch: xbar = sum_{i <= n} x_i / n.
    [[nodiscard]] std::vector<double> z(double tau) {
        std::vector<double> out(n_);
        for (std::size_t i = 0; i < n_; ++i) {
            double s = 0.0;
            for (std::size_t k = 0; k < n_; ++k) s += simplex_[k].x[i];
            const double xbar = s / static_cast<double>(n_);
            out[i] = xbar + tau * (xbar - simplex_[n_].x[i]);
        }
        project(out);
        return out;
    }

    // Steps 2 to 5 of Algorithm NM, as printed.
    template <class Oracle>
    void iterate(Oracle& oracle) {
        const Ordered before = simplex_;
        const std::size_t first = evals_;
        const double f1 = simplex_[0].f;
        const double fn = simplex_[n_ - 1].f;
        const double fn1 = simplex_[n_].f;
        const double rho = coef_.rho, chi = coef_.chi, gamma = coef_.gamma;

        // 2. Reflect.
        Vertex r{.x = z(rho), .f = 0.0};
        if (!evaluate(oracle, r.x, r.f)) return;
        if (f1 <= r.f && r.f < fn) {
            accept(std::move(r), Termination::Reflect, rho, before, first);
            return;
        }
        // 3. Expand.
        if (r.f < f1) {
            Vertex e{.x = z(rho * chi), .f = 0.0};
            if (!evaluate(oracle, e.x, e.f)) return;
            if (e.f < r.f)
                accept(std::move(e), Termination::Expand, rho * chi, before, first);
            else
                accept(std::move(r), Termination::Expand, rho, before, first);
            return;
        }
        // 4. Contract: fr >= fn here.
        if (fn <= r.f && r.f < fn1) {
            // a. Outside.
            Vertex c{.x = z(rho * gamma), .f = 0.0};
            if (!evaluate(oracle, c.x, c.f)) return;
            if (c.f <= r.f) {
                accept(std::move(c), Termination::Contract, rho * gamma, before, first);
                return;
            }
        } else {
            // b. Inside: fr >= f_{n+1}.
            Vertex cc{.x = z(-gamma), .f = 0.0};
            if (!evaluate(oracle, cc.x, cc.f)) return;
            if (cc.f < fn1) {
                accept(std::move(cc), Termination::Contract, -gamma, before, first);
                return;
            }
        }
        // 5. Shrink.
        shrink(oracle, before, first);
    }

    // The nonshrink ordering rule, p. 116, in the reading of p. 118: x_{n+1}
    // is discarded and v takes position j + 1 for the smallest l = j with
    // f(v) < f(x_{l+1}); every other vertex keeps its relative order.
    void accept(Vertex v, Termination t, double tau, const Ordered& before, std::size_t first) {
        std::size_t j = n_;
        for (std::size_t l = 0; l <= n_; ++l) {
            if (v.f < before[l].f) {
                j = l;
                break;
            }
        }
        Ordered next(before.begin(), before.begin() + static_cast<std::ptrdiff_t>(n_));
        next.insert(next.begin() + static_cast<std::ptrdiff_t>(j), std::move(v));
        simplex_ = std::move(next);
        note(t, tau, before, first);
    }

    // Step 5 and the shrink ordering rule, pp. 116-117: v_i = x_1 + sigma
    // (x_i - x_1) for i = 2 .. n + 1, evaluated in rank order; x_1 stays
    // first when a new point ties with it, and beyond that the new points
    // keep the order their originals had (FLOP's documented choice of
    // "whatever rule is used to define the original ordering"). FLOP's
    // documented deviations: a vertex the rounded shrink leaves where it was
    // keeps its value and is not evaluated again, and a shrink that moves no
    // vertex is the precision of the arithmetic reached, a RoundoffLimited
    // verdict.
    template <class Oracle>
    void shrink(Oracle& oracle, const Ordered& before, std::size_t first) {
        Ordered fresh;
        std::vector<bool> moved;
        for (std::size_t k = 1; k <= n_; ++k) {
            Vertex v{.x = std::vector<double>(n_), .f = before[k].f};
            bool changed = false;
            for (std::size_t i = 0; i < n_; ++i) {
                const double x = before[0].x[i] + coef_.sigma * (before[k].x[i] - before[0].x[i]);
                v.x[i] = x == before[k].x[i] ? before[k].x[i] : x;
                changed = changed || x != before[k].x[i];
            }
            moved.push_back(changed);
            fresh.push_back(std::move(v));
        }
        if (std::ranges::none_of(moved, [](bool b) { return b; })) {
            ++run_.still_shrinks;
            conclude(oracle, flop::Status::RoundoffLimited, radius());
            return;
        }
        for (std::size_t k = 0; k < n_; ++k)
            if (moved[k] && !evaluate(oracle, fresh[k].x, fresh[k].f)) return;
        std::ranges::stable_sort(fresh, [](const Vertex& a, const Vertex& b) { return a.f < b.f; });
        const auto ahead =
            std::ranges::count_if(fresh, [&](const Vertex& v) { return v.f < before[0].f; });
        fresh.insert(fresh.begin() + ahead, before[0]);
        simplex_ = std::move(fresh);
        ++run_.shrinks;
        note(Termination::Shrink, 0.0, before, first);
    }

    void note(Termination t, double tau, const Ordered& before, std::size_t first) {
        if (!record_) return;
        run_.steps.push_back({.termination = t,
                              .tau = tau,
                              .before = before,
                              .after = simplex_,
                              .first_evaluation = first,
                              .evaluations = evals_ - first});
    }

    std::vector<double> x0_;
    std::size_t n_;
    const flop::nelder_mead::Options& opts_;
    Coefficients coef_;
    bool record_;

    Ordered simplex_;
    bool box_acted_ = false;
    std::optional<double> complete_radius_;
    ReferenceRun run_;
    std::size_t evals_ = 0;
    bool ended_ = false;
    flop::Status status_ = flop::Status::RoundoffLimited;
    std::vector<double> best_x_;
    double best_f_ = 0.0;
    bool have_best_ = false;
};

// ---- oracles ----------------------------------------------------------------

// Runs the reference on an objective of its own.
template <class F>
struct Direct {
    F& f;
    Trace trace;
    std::optional<double> operator()(std::span<double> x, double /*scale*/) {
        const double v = f(std::span<const double>(x));
        trace.push_back({.index = trace.size(), .x = {x.begin(), x.end()}, .f = v});
        return v;
    }
};

template <class F>
Direct<F> direct(F& f) {
    return Direct<F>{.f = f, .trace = {}};
}

// Replays a FLOP trace. Each point the reference asks for is compared with
// FLOP's next evaluation, bit for bit (tolerance 0) or to tolerance * (n + 2)
// * epsilon * scale per coordinate, where scale is the magnitude the point
// was computed from; then FLOP's point and value are handed back, so the
// reference continues from exactly what FLOP evaluated.
class Audit {
public:
    // The per-coordinate allowance, in units of (n + 2) epsilon times the
    // scale: the running sum FLOP keeps is rebuilt every n + 1 replacements,
    // so its rounding is that of O(n) additions of coordinates of the
    // scale's size, divided by n and carried through a coefficient of at
    // most 3 (1 + rho chi).
    static constexpr double kRounding = 16.0;

    Audit(const Trace& t, bool exact) : t_(t), exact_(exact) {}

    std::optional<double> operator()(std::span<double> x, double scale) {
        if (k_ == t_.size()) return std::nullopt;
        const Point& p = t_[k_];
        if (!mismatch_ && !agrees(x, p.x, scale)) {
            mismatch_ = k_;
            std::ostringstream os;
            os.precision(17);
            os << "evaluation " << k_ << ": the reference wants (";
            for (std::size_t i = 0; i < x.size(); ++i) os << (i ? ", " : "") << x[i];
            os << "), FLOP evaluated (";
            for (std::size_t i = 0; i < p.x.size(); ++i) os << (i ? ", " : "") << p.x[i];
            os << ")";
            what_ = os.str();
        }
        std::ranges::copy(p.x, x.begin());
        ++k_;
        return p.f;
    }

    [[nodiscard]] bool clean() const { return !mismatch_; }
    [[nodiscard]] std::size_t consumed() const { return k_; }
    [[nodiscard]] const std::string& what() const { return what_; }

private:
    [[nodiscard]] bool agrees(std::span<const double> a, std::span<const double> b,
                              double scale) const {
        if (a.size() != b.size()) return false;
        if (exact_) return same_bits(a, b);
        const double allowed = kRounding * static_cast<double>(a.size() + 2) *
                               std::numeric_limits<double>::epsilon() * scale;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (std::fabs(a[i] - b[i]) > allowed) return false;
        return true;
    }

    const Trace& t_;
    bool exact_;
    std::size_t k_ = 0;
    std::optional<std::size_t> mismatch_;
    std::string what_;
};

// The verdict of an audit: FLOP's whole trace was what the reference asked
// for, nothing more and nothing less (points_agree, same_length); the two
// runs ended the same way at the same best point (same_outcome); and they
// report the same final radius (same_radius), a field of its own because a
// run that ends part way through a shrink or a restart reports the radius
// of the simplex before it, which the tests on those paths name directly.
struct AuditReport {
    bool points_agree;
    bool same_length;
    bool same_outcome;
    bool same_radius;
    std::string detail;
};

inline AuditReport audit(std::span<const double> x0, const flop::nelder_mead::Options& opts,
                         const Trace& trace, const flop::Result& r, bool exact,
                         ReferenceRun* out = nullptr, bool record = false) {
    Reference ref(x0, opts, record);
    Audit oracle(trace, exact);
    ReferenceRun run = ref.run(oracle);
    AuditReport rep{.points_agree = oracle.clean(),
                    .same_length = !run.ran_out && oracle.consumed() == trace.size(),
                    .same_outcome = run.result.status == r.status &&
                                    run.result.evaluations == r.evaluations &&
                                    same_bits(run.result.x, r.x) && same_bits(run.result.f, r.f),
                    .same_radius = same_bits(run.result.final_radius, r.final_radius),
                    .detail = oracle.what()};
    if (!rep.same_length) {
        std::ostringstream os;
        os << " reference consumed " << oracle.consumed() << " of " << trace.size()
           << " evaluations" << (run.ran_out ? " and wanted more" : "");
        rep.detail += os.str();
    }
    if (!rep.same_outcome || !rep.same_radius) {
        std::ostringstream os;
        os.precision(17);
        os << " reference status " << flop::to_string(run.result.status) << " after "
           << run.result.evaluations << ", radius " << run.result.final_radius << "; FLOP "
           << flop::to_string(r.status) << " after " << r.evaluations << ", radius "
           << r.final_radius;
        rep.detail += os.str();
    }
    if (out) *out = std::move(run);
    return rep;
}

}  // namespace v0111
