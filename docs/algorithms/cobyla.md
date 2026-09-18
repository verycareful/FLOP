# COBYLA

Constrained Optimization BY Linear Approximations, from M. J. D. Powell, "A
direct search optimization method that models the objective and constraint
functions by linear interpolation", in Advances in Optimization and Numerical
Analysis (S. Gomez and J.-P. Hennart, eds.), Kluwer, 1994, pp. 51-67. Written
from the paper; no code from any other implementation. Section numbers below
are the paper's.

## The method

COBYLA minimises `f(x)` subject to `c_i(x) >= 0` without derivatives. It
keeps a simplex of `n + 1` points, interpolates `f` and each `c_i` linearly
on it, and takes a trust-region step on those linear models: first the step
that makes the largest linearised constraint violation as small as the
region allows, then, holding that violation, the step that makes the
linearised objective as small as it can be (section 3). Progress is judged
on a merit function `f + mu * violation`, with `mu` raised when a predicted
improvement would not count and lowered when the simplex shows it is larger
than the constraints need (section 2). When a step fails and the simplex
geometry is poor, the next iteration improves the simplex instead of
stepping: the vertex at the end of the longest edge, or the one closest to
its opposite face, is moved to `gamma * rho` from that face along the face
normal, on the side where the linearised merit is smaller. One such repair is
always followed by a trust-region step; the geometry is never repaired twice
in a row, and the first iteration after the initial simplex is a
trust-region step. The trust radius `rho` halves when a step is short or a
step fails on an acceptable simplex, and the run ends when `rho` reaches its
final value.

The four constants are the paper's: `alpha = 0.25`, `beta = 2.1`, `gamma =
0.5`, `delta = 1.1`.

The simplex is held as a base point and `n` displacement vectors with the
inverse of the displacement matrix alongside; the two simplex updates (a
vertex replaced, the base moved) are rank-one updates of that inverse and
cost `O(n^2)`. No inverse is ever recomputed.

## Deviations from the paper

- **Box bounds.** The paper has no bounds. FLOP respects them at every trial
  point. On the unconstrained path the trust-region step is the
  steepest-descent step to the ball boundary, clamped into the box one
  coordinate at a time with the radius redistributed over the free
  coordinates. On the constrained path the bounds join the linearised
  constraints as rows of the polyhedron. A geometry step that would leave the
  box is scaled back to it.
- **The trust-region subproblem.** The paper's step is computed by an
  active-set method. FLOP's unconstrained path has the closed form above. Its
  constrained path solves the paper's two stages exactly to tolerance by
  bisection over an inner projection: stage one bisects on the violation
  level and asks whether the polyhedron at that level reaches into the ball;
  stage two bisects on the multiplier of the ball, using the fact that the
  norm of the projected point is non-increasing in that multiplier. The
  projections are Hildreth's cyclic dual coordinate ascent, warm-started
  across the bisection. Simple and correct rather than fast; it is the first
  candidate for an active-set solver.
- **Batch initial simplex.** With a `BatchObjective`, `x0` and the `n`
  displaced points go out as one batch and the best of them becomes the base
  afterwards. With a scalar objective the paper's sequential construction
  runs, in which the base moves to any vertex that improves on it as the
  simplex is built. The evaluation cap is respected on both paths: a cap
  smaller than `n + 1` forces the sequential path.
- **No random state.** The method is deterministic given the objective's
  values.
- **No variable rescaling.** One scalar `initial_step`; a caller whose
  coordinates differ in scale rescales the problem.
- **Stopping rules beyond `rho`.** The paper stops at `rhoend` or the
  evaluation cap. FLOP also honours `ftol_rel` and `ftol_abs` on the fall in
  `f` across a step that moves the base, and `stop_value` after any
  evaluation.

## Options

`flop::cobyla::Options` extends `flop::Options` with `final_trust_radius`,
the paper's `rhoend`. When it is zero, `rhoend` is the larger of
`stopping.xtol_abs` and `stopping.xtol_rel * initial_step`. The radius has a
floor at machine precision of the initial step regardless; a run that reaches
that floor without a caller-set `rhoend` reports `RoundoffLimited`.

`initial_step` is the paper's `rhobeg` and defaults to `1.0`. There is no
rule deriving it from `x0`.

## Status values

| Status | When |
|---|---|
| `XtolReached` | `rho` reached the caller's final radius |
| `FtolReached` | a step that moved the base lowered `f` by less than the f tolerance |
| `StopValueReached` | an evaluation returned `f <= stop_value`; that point is the result |
| `MaxEvaluationsReached` | the cap; the base, the best point so far, is the result |
| `RoundoffLimited` | the simplex became singular, a geometry step had zero length, or `rho` reached the precision floor |

A non-finite objective or constraint value ends the run with
`std::runtime_error`, where the compiler lets the value reach memory.

## Verification against the paper

Every rule below is implemented as described and should be read against the
paper's text before the first test release pins it:

- The penalty increase when a predicted merit reduction is not positive:
  `mu` is set to twice the value that makes it positive when `mu` is below
  one and a half times that value, and the base is then re-chosen as the
  vertex of least merit under the new `mu`.
- The penalty decrease on a radius reduction: `mu` is capped at the range of
  `f` over the simplex divided by the smallest, over constraints whose
  minimum is below half their maximum, of `max(c_max, 0) - c_min`; with no
  such constraint `mu` becomes zero.
- The vertex a trust-region point replaces: the vertex most aligned with the
  step (a failed step may only replace a vertex it beats by that measure),
  then, among vertices whose replacement keeps the simplex acceptable, the
  one at the end of the longest edge from the new point.
- The flat-objective case: with `mu = 0` and `f` unchanged, the step is
  judged on the violation alone.
- The sign of a geometry step: the side with the smaller linearised merit.
- The branch between repair and step: a trust-region step follows the initial
  simplex, every geometry step, and every radius reduction; a geometry step
  is taken only after a failed trust-region step on a simplex the
  acceptability test rejected.

## Limits

- The constrained subproblem solver converges linearly and is capped; on a
  degenerate constraint set the step is inexact, which the trust-region
  argument tolerates and the radius update absorbs.
- With `-ffinite-math-only` in the caller's translation unit, a NaN from the
  objective may never reach memory and cannot be detected there.
