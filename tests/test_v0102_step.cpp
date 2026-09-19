// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.0.2: the constrained trust-region subproblem, held to closed forms.
//
// Powell (1994, section 2, page 54) defines the step x* as the minimiser of
// the linearised objective subject to the ball and the linearised
// constraints, least norm among ties; when the ball and the constraints
// contradict, x* minimises the greatest violation within the ball, spends
// any remaining freedom on the objective, then takes the least norm. Every
// problem below is small enough that the definition can be solved by hand,
// so the solver is held to the closed form to a tolerance that is rounding,
// not method error. The 0.1.0.1 suite found that with constraints and a box
// together the step left the box and could stop at the unconstrained
// minimum; the box cases here pin the promise directly at the step level.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "flop/bounds.hpp"
#include "flop/detail/cobyla_step.hpp"
#include "v0101_problems.hpp"

namespace {

namespace cb = flop::detail::cobyla;

// Rounding, not method error: the closed forms below are a handful of
// operations, and the solver's active-set algebra on two or five variables
// is not much more.
constexpr double kExact = 1e-13;

// One subproblem: the linear models at the base xb (constraint k is
// cval[k] + G_k . d >= 0), the objective gradient g, the radius, and an
// optional box.
struct Subproblem {
    std::size_t n = 0;
    std::size_t m = 0;
    std::vector<double> xb;
    std::vector<double> cval;
    std::vector<double> G;  // row-major, m rows of n
    std::vector<double> g;
    double rho = 1.0;
    std::optional<flop::Bounds> bounds;
};

struct Step {
    std::vector<double> d;
    double predicted_violation = 0.0;
};

Step solve(const Subproblem& p, cb::StepWorkspace& ws) {
    Step s;
    s.d.assign(p.n, 0.0);
    std::vector<char> fixed;
    s.predicted_violation = cb::trust_region_step(p.n, p.m, p.xb, p.cval, p.G, p.g, p.rho,
                                                  p.bounds ? &*p.bounds : nullptr, s.d, ws, fixed);
    return s;
}

Step solve(const Subproblem& p) {
    cb::StepWorkspace ws;
    return solve(p, ws);
}

void expect_step(const Step& s, std::span<const double> expected, double tol = kExact) {
    ASSERT_EQ(s.d.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(s.d[i], expected[i], tol) << "coordinate " << i;
}

double norm(std::span<const double> d) {
    double s = 0.0;
    for (double v : d) s += v * v;
    return std::sqrt(s);
}

// A two-variable problem at the origin with one constraint.
Subproblem two_d(std::vector<double> g, double cval, std::vector<double> G, double rho) {
    Subproblem p;
    p.n = 2;
    p.m = 1;
    p.xb = {0.0, 0.0};
    p.cval = {cval};
    p.G = std::move(G);
    p.g = std::move(g);
    p.rho = rho;
    return p;
}

// ---- the objective against the ball and one constraint --------------------

// min d_1 + d_2 subject to d_2 >= -a and |d| <= 1, with a below 1/sqrt2 so
// the constraint is active: d_2 = -a, and the rest of the radius goes to
// d_1 = -sqrt(1 - a^2).
TEST(V0102Step, AnActiveHalfspaceAndTheBall) {
    const double a = 0.25;
    const Step s = solve(two_d({1.0, 1.0}, a, {0.0, 1.0}, 1.0));
    const std::vector<double> expected = {-std::sqrt(1.0 - a * a), -a};
    expect_step(s, expected);
    EXPECT_NEAR(s.predicted_violation, 0.0, kExact);
}

// min d_2 subject to d_2 >= -a and |d| <= 1: every point of the chord
// d_2 = -a minimises the objective, and the definition picks the one of least
// norm, (0, -a), which lies inside the ball.
TEST(V0102Step, ALineOfMinimisersGivesTheLeastNormOne) {
    const double a = 0.4;
    const Step s = solve(two_d({0.0, 1.0}, a, {0.0, 1.0}, 1.0));
    const std::vector<double> expected = {0.0, -a};
    expect_step(s, expected);
    EXPECT_NEAR(s.predicted_violation, 0.0, kExact);
}

// The constraint is satisfied at the origin with room to spare and the
// steepest-descent step to the ball does not reach it: the constrained
// step is the unconstrained one, -rho g / |g|.
TEST(V0102Step, AnInactiveConstraintGivesTheSteepestDescentStep) {
    const Step s = solve(two_d({3.0, 4.0}, 10.0, {0.0, 1.0}, 0.5));
    const std::vector<double> expected = {-0.3, -0.4};
    expect_step(s, expected);
}

// A zero gradient with a feasible origin: no direction is preferred, so the
// least-norm point of the feasible set is the origin itself.
TEST(V0102Step, AZeroGradientAtAFeasibleOriginStaysPut) {
    const Step s = solve(two_d({0.0, 0.0}, 0.3, {1.0, 0.0}, 1.0));
    const std::vector<double> expected = {0.0, 0.0};
    expect_step(s, expected);
}

// ---- stage 1: an infeasible origin -----------------------------------------

// -1 + d_1 >= 0 cannot be met inside a ball of radius 1/2: the best the
// ball can do is d = (1/2, 0), leaving a violation of 1/2, which the step
// reports as its predicted violation.
TEST(V0102Step, AnInfeasibleOriginMinimisesTheViolationFirst) {
    const Step s = solve(two_d({0.0, 1.0}, -1.0, {1.0, 0.0}, 0.5));
    const std::vector<double> expected = {0.5, 0.0};
    expect_step(s, expected);
    EXPECT_NEAR(s.predicted_violation, 0.5, kExact);
}

// -0.3 + d_1 >= 0 is reachable inside the unit ball, so the violation drops
// to zero and the freedom left, d_1 >= 0.3, goes to the objective d_2:
// d = (0.3, -sqrt(1 - 0.09)).
TEST(V0102Step, AReachableConstraintLeavesFreedomForTheObjective) {
    const Step s = solve(two_d({0.0, 1.0}, -0.3, {1.0, 0.0}, 1.0));
    const std::vector<double> expected = {0.3, -std::sqrt(1.0 - 0.09)};
    expect_step(s, expected);
    EXPECT_NEAR(s.predicted_violation, 0.0, kExact);
}

// A zero gradient with an infeasible origin: stage 1 alone decides, and the
// least-norm point of the level set is the foot of the perpendicular.
TEST(V0102Step, AZeroGradientAtAnInfeasibleOriginTakesTheShortestRepair) {
    const Step s = solve(two_d({0.0, 0.0}, -0.3, {1.0, 0.0}, 1.0));
    const std::vector<double> expected = {0.3, 0.0};
    expect_step(s, expected);
}

// d_1 >= 0.6 and d_1 <= -0.6 contradict each other: the greatest violation
// max(0.6 - d_1, 0.6 + d_1) is least at d_1 = 0, level 0.6, and the
// objective d_2 then runs to the ball: d = (0, -1).
TEST(V0102Step, ContradictoryConstraintsSplitTheDifference) {
    Subproblem p;
    p.n = 2;
    p.m = 2;
    p.xb = {0.0, 0.0};
    p.cval = {-0.6, -0.6};
    p.G = {1.0, 0.0, -1.0, 0.0};
    p.g = {0.0, 1.0};
    p.rho = 1.0;
    const Step s = solve(p);
    const std::vector<double> expected = {0.0, -1.0};
    expect_step(s, expected);
    EXPECT_NEAR(s.predicted_violation, 0.6, kExact);
}

// A constraint whose linear model has a zero gradient cannot be repaired by
// any step; its violation sets the level, and the other constraint, relaxed
// to that level, still binds the objective: 0.3 + d_1 >= -0.4 gives
// d_1 >= -0.7, and min d_1 lands there with d_2 = 0 by least norm.
TEST(V0102Step, AZeroRowSetsTheLevelAndTheRestStillBinds) {
    Subproblem p;
    p.n = 2;
    p.m = 2;
    p.xb = {0.0, 0.0};
    p.cval = {-0.4, 0.3};
    p.G = {0.0, 0.0, 1.0, 0.0};
    p.g = {1.0, 0.0};
    p.rho = 1.0;
    const Step s = solve(p);
    const std::vector<double> expected = {-0.7, 0.0};
    expect_step(s, expected);
    EXPECT_NEAR(s.predicted_violation, 0.4, kExact);
}

// ---- degenerate constraint sets --------------------------------------------

// The same constraint twice must give the step the single one gives: a
// second row parallel to an active one carries no new information and must
// not be allowed to break the active-set algebra.
TEST(V0102Step, ADuplicatedConstraintChangesNothing) {
    const double a = 0.25;
    const Step single = solve(two_d({1.0, 1.0}, a, {0.0, 1.0}, 1.0));
    Subproblem p = two_d({1.0, 1.0}, a, {0.0, 1.0}, 1.0);
    p.m = 2;
    p.cval = {a, a};
    p.G = {0.0, 1.0, 0.0, 1.0};
    const Step twice = solve(p);
    expect_step(twice, single.d);
}

// Two constraints and the ball meet at a corner inside the ball: d_1 >= -a
// and d_2 >= -b with a^2 + b^2 < 1, objective d_1 + d_2, so the corner
// (-a, -b) is the unique minimiser and the ball is slack.
TEST(V0102Step, TwoActiveConstraintsMeetInsideTheBall) {
    Subproblem p;
    p.n = 2;
    p.m = 2;
    p.xb = {0.0, 0.0};
    p.cval = {0.3, 0.4};
    p.G = {1.0, 0.0, 0.0, 1.0};
    p.g = {1.0, 1.0};
    p.rho = 1.0;
    const Step s = solve(p);
    const std::vector<double> expected = {-0.3, -0.4};
    expect_step(s, expected);
}

// ---- a box together with constraints ---------------------------------------

// The 0.1.0.1 defect at the step level. Box d_1 >= -0.2 (lower bound on the
// first variable), constraint d_2 >= -0.5, objective d_1 + d_2, unit ball:
// the corner (-0.2, -0.5) has norm below 1, so it is the step, and the
// base plus the step is inside the box.
TEST(V0102Step, ABoxRowAndAConstraintTogether) {
    Subproblem p = two_d({1.0, 1.0}, 0.5, {0.0, 1.0}, 1.0);
    const std::vector<double> lo = {-0.2, -1.0};
    const std::vector<double> hi = {1.0, 1.0};
    p.bounds = flop::Bounds::box(lo, hi);
    const Step s = solve(p);
    const std::vector<double> expected = {-0.2, -0.5};
    expect_step(s, expected);
    EXPECT_GE(p.xb[0] + s.d[0], lo[0]);
    EXPECT_GE(p.xb[1] + s.d[1], lo[1]);
}

// A box row active together with the ball while the constraint is slack:
// d_1 is held at its bound and the rest of the radius goes to d_2. The
// constrained path must respect the box as strictly as the unconstrained one
// does: the point is inside the box, not within rounding of it.
TEST(V0102Step, AnActiveBoxRowWithTheBallStaysInsideTheBox) {
    Subproblem p = two_d({1.0, 1.0}, 5.0, {0.0, 1.0}, 1.0);
    const std::vector<double> lo = {-0.2, -2.0};
    const std::vector<double> hi = {2.0, 2.0};
    p.bounds = flop::Bounds::box(lo, hi);
    const Step s = solve(p);
    const std::vector<double> expected = {-0.2, -std::sqrt(1.0 - 0.04)};
    expect_step(s, expected);
    EXPECT_GE(p.xb[0] + s.d[0], lo[0]);
    EXPECT_LE(norm(s.d), p.rho * (1.0 + kExact));
}

// A base point on its bound with the objective pushing through the bound:
// the box row is active from the start, and the step can only move along the
// bound. With the constraint slack the answer is (0, -1).
TEST(V0102Step, ABaseOnItsBoundMovesAlongIt) {
    Subproblem p = two_d({1.0, 1.0}, 5.0, {0.0, 1.0}, 1.0);
    p.xb = {-0.2, 0.0};
    const std::vector<double> lo = {-0.2, -2.0};
    const std::vector<double> hi = {2.0, 2.0};
    p.bounds = flop::Bounds::box(lo, hi);
    const Step s = solve(p);
    const std::vector<double> expected = {0.0, -1.0};
    expect_step(s, expected);
    EXPECT_GE(p.xb[0] + s.d[0], lo[0]);
}

// An infeasible origin whose repair would leave the box: the box is never
// relaxed, so the violation is minimised over the box's slice of the ball.
// -1 + d_1 >= 0 wants d_1 = 1/2 inside the ball of radius 1/2, but the box
// allows d_1 <= 0.3: d_1 lands on the bound and the violation left is 0.7
// (d_2 then goes to the objective along the chord, which is not asserted).
TEST(V0102Step, TheBoxIsNeverRelaxedByTheViolationLevel) {
    Subproblem p = two_d({0.0, 1.0}, -1.0, {1.0, 0.0}, 0.5);
    const std::vector<double> lo = {-1.0, -1.0};
    const std::vector<double> hi = {0.3, 1.0};
    p.bounds = flop::Bounds::box(lo, hi);
    const Step s = solve(p);
    EXPECT_LE(p.xb[0] + s.d[0], hi[0]);
    EXPECT_NEAR(s.d[0], 0.3, kExact);
    EXPECT_NEAR(s.predicted_violation, 0.7, kExact);
}

// ---- more variables ----------------------------------------------------------

// Five variables, one hyperplane a . d >= -c that the steepest-descent step
// crosses. The step lies on the hyperplane and the ball: its component along
// a is fixed by the hyperplane, and the rest of the radius goes along the
// part of -g orthogonal to a.
TEST(V0102Step, OneHyperplaneInFiveVariables) {
    Subproblem p;
    p.n = 5;
    p.m = 1;
    p.xb.assign(5, 0.0);
    p.G = {1.0, 2.0, 0.0, -1.0, 0.5};
    p.cval = {0.3};
    p.g = {1.0, 1.0, 1.0, 1.0, 1.0};
    p.rho = 1.0;

    // Closed form: d = -c a / |a|^2 + s w, w the unit vector along the part
    // of -g orthogonal to a, s chosen so that |d| = rho.
    const std::span<const double> a = p.G;
    const double aa = cb::dot(a, a);
    const double ag = cb::dot(a, p.g);
    std::vector<double> w(5);
    for (std::size_t i = 0; i < 5; ++i) w[i] = -(p.g[i] - ag / aa * a[i]);
    const double wn = norm(w);
    const double along = -p.cval[0] / std::sqrt(aa);  // signed distance moved along a
    const double s = std::sqrt(p.rho * p.rho - along * along);
    std::vector<double> expected(5);
    for (std::size_t i = 0; i < 5; ++i) expected[i] = -p.cval[0] * a[i] / aa + s * w[i] / wn;

    // The unconstrained step must indeed cross the hyperplane, else the test
    // would be checking the wrong branch.
    const double gn = norm(p.g);
    double crossing = p.cval[0];
    for (std::size_t i = 0; i < 5; ++i) crossing += a[i] * (-p.rho * p.g[i] / gn);
    ASSERT_LT(crossing, 0.0);

    const Step st = solve(p);
    expect_step(st, expected);
    EXPECT_NEAR(norm(st.d), p.rho, kExact);
    EXPECT_NEAR(st.predicted_violation, 0.0, kExact);
}

// ---- the workspace -----------------------------------------------------------

// A workspace that has solved a different problem must not change the answer
// to this one: warm-start state is a speed-up, never an input.
TEST(V0102Step, AWarmWorkspaceGivesTheSameStep) {
    cb::StepWorkspace ws;
    Subproblem other;
    other.n = 2;
    other.m = 2;
    other.xb = {0.0, 0.0};
    other.cval = {-0.6, -0.6};
    other.G = {1.0, 0.0, -1.0, 0.0};
    other.g = {0.0, 1.0};
    other.rho = 1.0;
    (void)solve(other, ws);

    const double a = 0.25;
    const Subproblem p = two_d({1.0, 1.0}, a, {0.0, 1.0}, 1.0);
    const Step fresh = solve(p);
    const Step warm = solve(p, ws);
    expect_step(warm, fresh.d);
}

// The same inputs twice through the same workspace give the same bits: the
// solver holds no state that survives a call.
TEST(V0102Step, TheSameInputsGiveTheSameBits) {
    cb::StepWorkspace ws;
    Subproblem p;
    p.n = 5;
    p.m = 2;
    p.xb.assign(5, 0.1);
    p.G = {1.0, 2.0, 0.0, -1.0, 0.5, 0.0, 1.0, 1.0, 0.0, -2.0};
    p.cval = {0.3, -0.2};
    p.g = {1.0, -1.0, 1.0, -1.0, 1.0};
    p.rho = 0.7;
    const Step first = solve(p, ws);
    const Step second = solve(p, ws);
    EXPECT_TRUE(v0101::same_bits(first.d, second.d));
    EXPECT_TRUE(v0101::same_bits(first.predicted_violation, second.predicted_violation));
}

}  // namespace
