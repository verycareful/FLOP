# BOBYQA

Bound optimisation by quadratic approximation, from M. J. D. Powell, "The
BOBYQA algorithm for bound constrained optimization without derivatives",
report DAMTP 2009/NA06, University of Cambridge, 2009. The updates of the
model and of the inverse matrix are the ones the report restates from
M. J. D. Powell, "Least Frobenius norm updating of quadratic models that
satisfy interpolation conditions", Mathematical Programming 100, 2004,
pp. 183-215, and the weight that chooses which point a step replaces is
from M. J. D. Powell, "The NEWUOA software for unconstrained optimization
without derivatives", in Large-Scale Nonlinear Optimization, Springer,
2006, pp. 255-297. Written from the papers; no code from any other
implementation. Section and equation numbers below are the report's unless
another paper is named.

## The method

BOBYQA minimises `f(x)` subject to `a <= x <= b` without derivatives. It
keeps `m` interpolation points `y_1, ..., y_m`, with `n + 2 <= m <=
(n + 1)(n + 2) / 2` and `2n + 1` the usual choice, and a quadratic model
`Q` with `Q(y_j) = f(y_j)` for every point (1.2). The best point is `x_k`
(1.3). A trust-region radius `Delta` and its lower bound `rho` control the
steps; `rho` starts at `initial_step` and falls to the final radius.

- **Start (section 2).** `x0`, moved onto a bound or to `Delta` inside it
  when it is closer to the bound than that, and the points `x0 +- Delta e_i`
  (or `x0 + Delta e_i` and `x0 + 2 Delta e_i` from a lower bound), then for
  `m > 2n + 1` points displaced along two coordinates (2.3). The first model
  has the gradient and diagonal second derivatives these points determine
  and the off-diagonal pairs of the extra points; the inverse matrix of
  (2.7) is written down exactly by (2.11) to (2.15).
- **Trust-region iterations (section 3, first half).** The step `d`
  approximately minimises `Q(x_k + d)` in the box and the trust region by
  truncated conjugate gradients with an active set (TRSBOX), continued
  round the trust-region boundary once it is reached. If `F(x_k + d)` is
  evaluated, one point is replaced by `x_k + d` (1.6), and `Q` and the
  inverse matrix are updated so that `Q` interpolates the new set while the
  change in its second derivatives has the least Frobenius norm (section
  4). `Delta` follows the ratio of actual to predicted reduction (6.4).
- **Alternative iterations (section 3, second half).** When the model
  needs better points rather than a better `x_k`, the point furthest from
  `x_k` is replaced (3.7) by a step that makes its Lagrange function large
  (ALTMOV): a point on one of the lines from `x_k` through the other
  interpolation points (3.11), or a constrained Cauchy step (3.13) when that
  gives the larger denominator (3.12).
- **Control (section 6).** The choice between the two kinds of iteration,
  the reduction of `rho` by (6.6), the test (6.8) to (6.11) that the model
  is good enough to give up the current `rho`, the shift of the origin
  (6.13) to (6.17), and the replacement of `Q` by the least-norm model of
  the current values when that has the much smaller projected gradient on
  three trust-region iterations in a row (6.12).
- **RESCUE (section 5).** When rounding has spoiled the update's
  denominator (5.2), the inverse matrix is rebuilt for a fresh set of
  coordinate points around `x_k`, and the old points are put back one at a
  time while that is safe (5.7); the new points that remain are evaluated
  and the model is made to interpolate them (5.12).

The second derivative matrix of `Q` is held as an explicit matrix plus a
sum of `m` rank-one terms (4.6), and the inverse matrix as `Z` with
`Omega = Z Z^T` and the `(m + n) x n` matrix of section 4, so an iteration
costs `O(m^2)` beyond its evaluation, and a shift of origin `O(m^2 n)`.

## Deviations from the paper

- **The choice of the point a trust-region step replaces.** The report
  chooses the `t` that maximises `max[1, |y_t - x_k|^2 / Delta^2] sigma_t`
  (6.1), `sigma_t` the denominator of the update. FLOP uses NEWUOA's
  weight by default (Powell 2006, equation (7.5)),
  `max[1, (|y_t - x*| / max[0.1 Delta, rho])^6]`, with `x*` the best point
  after the step, which removes far points sooner. On the report's own test
  problem (7.1), five random instances each at `n = 10` and `n = 20` with
  `m = 2n + 1`, `rhobeg = 0.1` and `rhoend = 1e-6`, the report's weight
  needed about a third more evaluations than NEWUOA's at the same accuracy
  (5926 against 4413 in total). `newuoa_weight = false` selects the report's weight.
- **Coordinates without a bound.** The report sets unwanted bounds to
  `+-1e60`. FLOP has no such values: a coordinate without a bound on a side
  has no constraint there, and every test against that side is skipped.
- **A box narrower than the first step.** The report makes an error return
  unless `b_i - a_i >= 2 initial_step` (2.1). FLOP throws
  `std::invalid_argument` at entry; with `shrink_initial_step` the first
  radius becomes `min(initial_step, min_i (b_i - a_i) / 2)` instead. A
  coordinate with `a_i = b_i` is always refused: it is not a variable.
- **Points on a bound.** The report requires a step onto a bound to put the
  point exactly on it (after (3.2)). The step routines report which
  coordinates they placed on which bound, and the point evaluated takes the
  bound's own value there, so no point is a rounding error away from a
  bound it should be on, and none is outside the box.
- **TRSBOX's search round the boundary.** The report finds `theta_B` and
  `theta_Q` "approximately". FLOP computes `theta_B` in closed form: each
  coordinate of `d(theta)` is a sinusoid, which reaches a bound at
  `phi +- acos(q / R)`. `theta_Q`, the first minimum of `Q` along the arc,
  is the first zero of the derivative, found by sampling the derivative at
  16 equal steps of `[0, pi/4]` and refining by Illinois regula falsi. A
  conjugate gradient phase takes at most as many steps as there are free
  coordinates, after which a quadratic has nothing left to give in exact
  arithmetic, and a move round the boundary that does not lower `Q` ends
  the search.
- **ALTMOV's Cauchy step.** The report takes the multiple of the solution
  of (3.13) that minimises the Lagrange function; FLOP takes it in
  `[0, 1]`, the range in which the step stays in the box and the trust
  region by construction.
- **A short trust-region step.** When `|d| < rho / 2`, the report sets
  `Delta` to `min[Delta / 10, delta_k / 2]` (the paragraph of (6.5)), with
  `delta_k` the greatest distance from `x_k` to a point, and `rho` when that
  is at most `1.5 rho`. That is implemented as read.
- **The error estimate (6.8).** A step counts as no longer than `rho` by the
  radius of its own trust region, `min(radius, |d|)`: a step to the boundary
  can come out one unit in the last place longer than the radius, and taken
  literally that made the estimate unusable at every `rho`.
- **The temporary radius of an alternative iteration.** RESCUE, when an
  alternative iteration calls it, builds its points with the iteration's
  `Delta`, not the reduced radius ALTMOV is given (after (6.5)).
- **Error returns.** The report's two error returns, a predicted reduction
  that is not positive in (6.3) and a second RESCUE in one iteration that
  would change nothing, end the run with `RoundoffLimited`.
- **Stopping rules beyond `rho`.** The report stops at `rhoend` or the
  evaluation cap. FLOP also stops on `stop_value` after every evaluation,
  and with `FtolReached` when a step that lowers the best value lowers it by
  no more than the f tolerance. As in the report, a last short step is
  evaluated before the run ends at `rhoend`, so the result can be
  `x_k + d`.
- **Batch evaluation.** With a `BatchObjective`, the first `min(m, 2n + 1)`
  points go out as one call with `x0` first, the remaining `m - 2n - 1`
  (which depend on the first values) as a second, and the new points of a
  RESCUE as one call per run of consecutive slots. Every other point is
  evaluated alone, and the two channels run the same trajectory.

## Options

`flop::bobyqa::Options` extends `flop::Options` with:

| Field | Default | Meaning |
|---|---|---|
| `final_trust_radius` | `0` | the report's `rhoend`; `0` derives it from the x tolerances |
| `interpolation_points` | `0` | `m`; `0` is `2n + 1`, otherwise in `[n + 2, (n + 1)(n + 2) / 2]` |
| `shrink_initial_step` | `false` | lower the first radius to fit a narrow box instead of throwing |
| `newuoa_weight` | `true` | NEWUOA's weight in the choice of `t`; `false` is the report's (6.1) |

`initial_step` is the report's `rhobeg`, the first radius and the first
lower bound `rho`. `rhoend` is `final_trust_radius` when that is positive,
else the larger of `xtol_abs` and `xtol_rel * initial_step`, never above
`rhobeg`. It has a floor of `epsilon * rhobeg`, the scale below which the
points cannot be told apart; reaching the floor is `RoundoffLimited`,
reaching the caller's own `rhoend` is `XtolReached`.

## Status values

| Status | When |
|---|---|
| `XtolReached` | the work at the final radius was complete (section 6) |
| `FtolReached` | a step that lowered the best value lowered it by at most the f tolerance |
| `StopValueReached` | an evaluation returned `f <= stop_value` |
| `MaxEvaluationsReached` | the cap |
| `RoundoffLimited` | `rho` reached the precision floor, or one of the report's error returns |

Whatever the status, the result is the best point evaluated, and
`Result::final_radius` is `rho` at the end.

## Verification against the paper

These rules were read against the report's text and checked numerically
while they were written, and await the tests that pin them:

- the starting points (2.2) to (2.4), including the swap of the pair at
  `+-Delta` when the second value is the lower one, and the first model
  (2.5);
- the inverse matrix (2.11) to (2.15), its update (4.9) to (4.14) with the
  identities (4.13), and the shift (6.14) to (6.17), each checked against
  an explicitly inverted `W` of (2.7) for every `m` with `n <= 5`;
- the model update (4.3) to (4.8), checked by the model interpolating every
  point to rounding after every update;
- TRSBOX's initial active set (3.3), its tests (3.4) and (3.5), and the arc
  (3.6);
- ALTMOV's line steps (3.11) and Cauchy steps (3.13) to (3.15), and the
  comparison (3.12);
- the radius rules (6.4), the short-step rule, (6.6) and (6.7), the usable
  error estimate (6.8) and the tests (6.10) and (6.11), and the alternative
  model (6.12);
- RESCUE's points (5.4) and (5.5), its safety test (5.7), its scores, and
  the model update (5.12).

On the report's test problem (7.1), five random instances, the evaluation
counts are 204 to 334 at `n = 10` and 542 to 705 at `n = 20`, against the
report's 302 to 427 and 691 to 927 for its own five instances.

## Limits

- The method has no convergence theory; the report says as much (section
  1).
- RESCUE costs `O(m^3)` and new evaluations. The report calls it unusual
  unless the requested accuracy is unattainable, which is when it is seen:
  a final radius far below the scale at which `f` changes by more than its
  rounding.
- Near the final radius on a bound, a run whose every step lowers `f` by an
  amount at the level of its rounding keeps `rho`, since the report keeps
  `rho` while `f` decreases, and can reach the cap with
  `MaxEvaluationsReached`. Whether it does depends on the rounding of the
  objective and the compiler.
- Storage is `O(m^2 + m n + n^2)`: at `m = (n + 1)(n + 2) / 2` that is
  `O(n^4)`.
- With `-ffinite-math-only` in the caller's translation unit, a NaN the
  objective returns may never reach the check. Clang declares every
  `double` a function returns or takes by value free of NaN and infinity,
  so such a value is undefined before FLOP sees it; GCC keeps the bits and
  the check catches them. `x0`, the bounds and the tolerances are read from
  memory and are caught under every compiler and model.
