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
linearised objective as small as it can be, the shortest such step when
several qualify (section 2, the definition of `x*`). Progress is judged
on a merit function `f + mu * violation`, with `mu` raised when a predicted
improvement would not count and lowered when the simplex shows it is larger
than the constraints need (section 2). When a step fails and the simplex
geometry is poor, the next iteration improves the simplex instead of
stepping: the vertex at the end of the longest edge, or the one closest to
its opposite face, is moved to `gamma * rho` from that face along the face
normal, on the side where the linearised merit is smaller. One such repair is
always followed by a trust-region step; the geometry is never repaired twice
in a row, and the first iteration after the initial simplex is a
trust-region step. The radius `rho` halves when a step is short or a step
fails on an acceptable simplex, and the run ends when `rho` reaches its
final value.

The four constants are the paper's: `alpha = 0.25`, `beta = 2.1`, `gamma =
0.5`, `delta = 1.1`.

The penalty `mu` follows section 2 exactly: at every trust-region step,
`mu_bar` is the least penalty at which the step is predicted to lower the
merit; `mu` stays if it is at least three halves of `mu_bar` and becomes
twice `mu_bar` otherwise, whether or not the prediction was already
positive, and a raised penalty re-chooses the base as the vertex of least
merit before anything is evaluated.

The simplex is held as a base point and `n` displacement vectors with the
inverse of the displacement matrix alongside; the two simplex updates (a
vertex replaced, the base moved) are rank-one updates of that inverse and
cost `O(n^2)`. No inverse is ever recomputed.

## Deviations from the paper

- **A trust radius that grows.** The paper's `rho` is never increased
  (its abstract and page 63 say so), and along a valley that costs a crawl:
  once the simplex has flattened onto the valley floor its linear model is
  the exact gradient along the floor, every step is a perfect step of length
  `rho`, nothing fails, and `rho` stays. FLOP steps with a radius `Delta`
  that starts at `rho` and doubles after a very successful step (the merit
  fell by at least half of the prediction and by at least half of the
  previous step's fall), never past `initial_step`; a failed step halves
  `Delta` down to `rho`, and `rho` itself is cut, by the paper's rule, only
  when a step fails with `Delta` already at `rho`. Every use of the radius
  in the iteration (the step, the short-step test, the acceptability tests,
  the repair length, the replacement threshold) is `Delta`; `rho` keeps its
  schedule, the penalty reduction at each cut, and termination, so the
  stopping rules mean what they meant. The two radii follow Powell's later
  methods (NEWUOA, 2006, and BOBYQA, 2009), which keep a lower bound `rho`
  under an adaptive trust radius. `Options::trust_region_growth = false`
  pins `Delta` to `rho` and gives the paper's method exactly.
- **Box bounds.** The paper has no bounds. FLOP respects them at every trial
  point. On the unconstrained path the trust-region step is the
  steepest-descent step to the ball boundary, clamped into the box one
  coordinate at a time with the radius redistributed over the free
  coordinates. On the constrained path the bounds join the linearised
  constraints as rows of the polyhedron, never relaxed by the violation
  level, and the step is clamped into the box afterwards so that the box
  holds to the bit. A geometry step that would leave the box is scaled back
  to it.
- **The trust-region subproblem.** The paper defines `x*` and gives one
  sentence on its computation (a trajectory of straight pieces as `rho`
  grows from zero, followed by updating active sets); the details are in
  Powell's Fortran, which FLOP does not read. FLOP's unconstrained path has
  the closed form above. Its constrained path is FLOP's own exact solver for
  the paper's definition: both stages are a linear objective over the
  polyhedron of linearised constraints and box rows inside the ball, and
  each is solved by following an exact projection onto the polyhedron along
  one scalar parameter. The projection is the dual active-set method of
  Goldfarb and Idnani (Mathematical Programming 27, 1983) with the identity
  Hessian, which makes their factorisation a QR of the active rows updated
  by Givens rotations, and which certifies an empty polyhedron by the
  combination of rows that contradicts the entering one. Stage one finds the
  least violation level at which the relaxed polyhedron reaches the ball: on
  a fixed active set the projection of the origin is affine in the level, so
  the crossing is a quadratic solved in closed form, and an empty polyhedron
  hands back the level at which its certificate disappears. Stage two
  minimises the linearised objective at that level through the ball's
  multiplier: the minimiser of `g . d + nu / 2 |d|^2` over the polyhedron is
  the projection of `-g / nu`, whose norm is non-increasing in `nu`, and on
  a fixed active set it is `u - v / nu` with `u` and `v` orthogonal, so the
  crossing is again closed-form; the `nu -> 0` limit is the least-norm
  minimiser, the paper's tie-break, detected as an active set on which no
  multiplier decreases. Each search keeps a bracket and falls back to its
  midpoint when a candidate leaves it, so it terminates on any input. The
  step costs one factorisation, updated a row at a time.
- **Batch initial simplex.** With a `BatchObjective`, `x0` and the `n`
  displaced points, in that order, go out as one batch and the best of them
  becomes the base afterwards, so evaluation index 0 is `x0` on both paths.
  With a scalar objective the paper's sequential construction runs, in which
  the base moves to any vertex that improves on it as the simplex is built.
  The evaluation cap is respected on both paths: a cap smaller than `n + 1`
  forces the sequential path.
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

`trust_region_growth`, on by default, lets the step radius grow above `rho`
as described under deviations; off, the method is the paper's.

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

The test suite runs the paper's test problems (A) to (J) from the paper's
own settings (`x0 = (1, ..., 1)`, `rhobeg = 1/2`, `rhoend` of `1e-3` and
`1e-4`) and holds each to Powell's own Tables 1 and 2: an evaluation count,
a final violation and a final distance to the solution within a factor of
two of his, and a final objective error within the square of that, since
the objective is quadratic in the distance near a minimum. The factor is
where the comparison stops being path noise: his numbers are one
single-precision run, and a run of the same rules in double precision (his
own code, through NLopt) lands elsewhere by up to a factor of two as well. It also runs
problems 1 to 9 from the shared starting points to their known optima, and
the subproblem solver on hand-solved cases to rounding.

The iteration's rules have each been read against the paper's text: the
radius schedule (11), the penalty increase (page 56) and its reduction at a
radius cut (12) and (13), acceptability (14), the choice of the vertex to
regenerate (15) and (16) and its placement (17), the short-step test (18),
the replacement rules (19) to (22), the branch between repair and step
(Figure 1 and conditions (C1) to (C5)), the initial simplex, and the final
point. Two things the text leaves open are decided as follows:

- The flat-objective case: with `mu = 0` and `f` unchanged, the step is
  judged on the violation alone.
- A tie in the linearised merit between the two sides of a geometry step
  goes to the side the box lets travel further, then to the positive side.

Against NLopt 2.7.1's COBYLA (Powell's code, with its own additions),
measured over twenty starting points per problem, FLOP reaches `rhoend`
in fewer evaluations on every one of (A) to (J) and reaches a smaller gap
at a fixed budget on Rosenbrock in two and ten dimensions and on Powell's
singular function; the two trajectories agree to `1e-9` up to NLopt's first
regenerated vertex, which NLopt perturbs and FLOP does not.

## Limits

- Ten dimensions and above are where a linear model earns least: the
  simplex is rarely acceptable, a quarter of the evaluations go to repairs,
  and progress along a curved valley is a crawl for every implementation of
  the method. FLOP matches NLopt there and does not beat it.
- With `-ffinite-math-only` in the caller's translation unit, a NaN the
  objective returns may never reach the check. Clang declares every
  `double` a function returns or takes by value free of NaN and infinity,
  so such a value is undefined before FLOP sees it; GCC keeps the bits and
  the check catches them. Values FLOP reads from memory (x0, bounds,
  tolerances, and constraint values written into the span it provides) are
  caught under every compiler and model.
