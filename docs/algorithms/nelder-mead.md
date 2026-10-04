# Nelder-Mead

The downhill simplex method of J. A. Nelder and R. Mead, "A simplex method
for function minimization", The Computer Journal 7(4), 1965, pp. 308-313,
in the precise statement of J. C. Lagarias, J. A. Reeds, M. H. Wright and
P. E. Wright, "Convergence properties of the Nelder-Mead simplex method in
low dimensions", SIAM Journal on Optimization 9(1), 1998, pp. 112-147,
section 2, with the dimension-dependent coefficients of F. Gao and L. Han,
"Implementing the Nelder-Mead simplex algorithm with adaptive parameters",
Computational Optimization and Applications 51(1), 2012, pp. 259-277.
Written from the papers; no code from any other implementation.

## The method

Nelder-Mead minimises `f(x)` without derivatives and without a model. It
keeps a simplex of `n + 1` vertices ordered by their values,
`f(x_1) <= ... <= f(x_{n+1})`, and each iteration tries to replace the worst
vertex by a point on the line through it and `xbar`, the centroid of the
other `n`. One iteration, as Lagarias et al. state it:

1. **Reflect.** `x_r = xbar + rho (xbar - x_{n+1})`. If
   `f(x_1) <= f(x_r) < f(x_n)`, accept `x_r`.
2. **Expand.** If `f(x_r) < f(x_1)`, evaluate
   `x_e = xbar + rho chi (xbar - x_{n+1})` and accept whichever of `x_e` and
   `x_r` has the smaller value, `x_e` only if `f(x_e) < f(x_r)`.
3. **Contract outside.** If `f(x_n) <= f(x_r) < f(x_{n+1})`, evaluate
   `x_c = xbar + rho gamma (xbar - x_{n+1})` and accept it if
   `f(x_c) <= f(x_r)`; otherwise shrink.
4. **Contract inside.** If `f(x_r) >= f(x_{n+1})`, evaluate
   `x_cc = xbar - gamma (xbar - x_{n+1})` and accept it if
   `f(x_cc) < f(x_{n+1})`; otherwise shrink.
5. **Shrink.** Every vertex but the best moves to
   `x_1 + sigma (x_i - x_1)`, and the `n` new points are evaluated.

An accepted point replaces the worst vertex. The paper's two tie-breaking
rules fix the order completely. After a step that does not shrink, the new
vertex ranks after every kept vertex whose value equals its own. After a
shrink, if a new vertex ties with `x_1`, `x_1` stays first. Where neither
rule applies (the initial simplex, the other vertices after a shrink), FLOP
keeps the order the vertices had before, with the initial vertices in
construction order.

The coefficients:

| | reflection `rho` | expansion `chi` | contraction `gamma` | shrink `sigma` |
|---|---|---|---|---|
| standard (Nelder and Mead) | 1 | 2 | 1/2 | 1/2 |
| adaptive (Gao and Han), `n >= 2` | 1 | 1 + 2/n | 3/4 - 1/(2n) | 1 - 1/n |

The adaptive values equal the standard ones at `n = 2`. Above roughly ten
dimensions the standard values spend most iterations on reflections, which
lower the objective least, and Gao and Han's Theorem 2.1 shows the descent
an expansion or contraction guarantees shrinking as `(n - 1)/(2 n^2)`. On
their problem (4.3) the standard method then stops far from the minimiser
(their Table 1). The adaptive values keep it moving, which is why they are
the default.

The vertices stay where they are in memory. Only an index of ranks moves,
and a step that does not shrink costs `O(n)` beyond its evaluations: the
centroid comes from a running sum of the vertices, rebuilt from scratch
every `n + 1` replacements so its rounding stays bounded; the new vertex is
ranked by binary search; and the least and greatest value of each
coordinate over the vertices are kept, with a rescan of one coordinate only
when the vertex that held its extreme is the one discarded. A shrink costs
`O(n^2)`.

## Deviations from the paper

- **Box bounds.** None of the three papers defines the method on a box.
  FLOP projects every trial point onto the box, coordinate by coordinate,
  so every evaluation is inside it and a minimum on a bound can be reached
  exactly. Projection also distorts the method: a reflection and the
  contraction after it can land on the same point of a face, and the
  simplex can flatten against a face, or collapse onto a corner, away from
  the minimiser, where it meets a stopping test without being near one.
  So once the box has acted on a run (a trial point was projected, or an
  initial vertex was moved by a bound), every stop verdict is preceded by
  a poll ladder. Each rung polls the `2n` points `x_1 + d e_i` and
  `x_1 - d e_i`, each clipped to the box, in coordinate order and plus
  before minus, leaving out a point clipped back onto `x_1`. The rungs run
  `d = initial_step, initial_step / 2, ...` down to the scale of the stop:
  the larger x tolerance for `XtolReached`, otherwise the simplex radius or
  the precision floor, whichever is larger. A rung is evaluated whole. If
  its first point of least value is strictly below `f(x_1)`, the method
  starts again from that point with a fresh simplex of edge `d`. Otherwise
  the next rung runs, and when the last one fails the stop stands. The
  restarts cannot cycle, since each one starts from a strictly lower
  value. The polls are those of coordinate search on a box in R. M. Lewis
  and V. Torczon, "Pattern search algorithms for bound constrained
  minimization", SIAM Journal on Optimization 9(4), 1999, pp. 1082-1099,
  with the step halved after every failed poll; here they certify a stop
  rather than drive the search. A box that never binds leaves the run
  exactly the unbounded one.
- **The initial simplex.** Lagarias et al. leave the initial simplex to the
  user. FLOP uses `x0` and, for each coordinate `i`, the point
  `x0 + initial_step * e_i` when it is inside the box, else
  `x0 - initial_step * e_i` when that one is, else `x0` with coordinate `i`
  set to the bound on the side with more room. The vertex is the bound
  itself, never `x0` plus the room, which can round past it.
- **The shrink in rounded arithmetic.** Step 5 evaluates `n` new points.
  Rounded, a vertex a few units in the last place from `x_1` can land back
  on itself: one `d` units away stays put whenever `sigma d` rounds to `d`,
  which for Gao and Han's `sigma = 1 - 1/n` is every `d` below about
  `n / 2`. FLOP evaluates only the vertices that moved; an unmoved one keeps
  its value. A shrink that moves no vertex would leave the simplex as it
  was, and the method would repeat it forever, so it is a stop verdict,
  `RoundoffLimited`, decided before anything is evaluated.
- **Stopping rules.** None of the papers fixes one. FLOP stops on the
  simplex radius `r`, the largest coordinate distance from a vertex to the
  best one (`XtolReached` when `r <= xtol_abs` or
  `r <= xtol_rel * initial_step`), on the spread `f(x_{n+1}) - f(x_1)`
  (`FtolReached` against `ftol_abs` or `ftol_rel * |f(x_1)|`), on
  `stop_value` after every evaluation, and on the cap. The radius has a
  floor at the precision of the coordinates, `epsilon` times the larger of
  `initial_step` and the largest coordinate of the best vertex; reaching
  it is `RoundoffLimited`, as is a shrink that moves no vertex. The tests
  run at the top of each iteration. The spread is compared without being
  formed where it would exceed the largest double (values of opposite sign
  near the ends of the range), and a relative tolerance above 1 by a
  division rather than a product that could overflow.
- **One dimension.** Gao and Han's coefficients are stated for `n >= 2`;
  at `n = 1` their shrink coefficient would be zero. FLOP uses the standard
  coefficients at `n = 1` whatever the option says.
- **Batch evaluation.** With a `BatchObjective`, the initial simplex goes
  out as one call of `n + 1` points with `x0` first, every shrink as one
  call of the vertices it moved, every rung of the ladder as one call of up
  to `2n`, and the fresh simplex of a restart as one call of `n`. Every
  other evaluation depends on the one before it and goes out alone. A cap too small for the whole batch sends the points one at a
  time, up to the cap. Both channels evaluate the same points in the same
  order, with one exception: when a point inside a batch call meets
  `stop_value`, the call has already evaluated every point in it, so all of
  them count as evaluations, reach `on_evaluation`, and can be the result,
  where the scalar channel stops at the point that met it.
- **The coordinate range.** No point is evaluated with a coordinate larger
  in magnitude than `DBL_MAX / (n + 5)`. Inside that range the running sum
  of the vertices, the centroid, every trial point and every shrunk vertex
  are finite, so nothing the method computes overflows. `x0` and the
  initial vertices are input, and one beyond the range is
  `std::invalid_argument` at entry, as is an `initial_step` for which
  `x0 +- initial_step` overflows. A later point beyond it ends the run with
  `std::runtime_error`; in practice that is an objective unbounded below
  along a coordinate without a bound, where the simplex expands without
  end.
- **No random state.** The method is deterministic given the objective's
  values.

## Options

`flop::nelder_mead::Options` extends `flop::Options` with
`adaptive_coefficients`, on by default: Gao and Han's coefficients when on,
the standard ones when off.

`initial_step` is the length of the initial simplex's edges along each
coordinate and the scale `xtol_rel` is measured against. There is no rule
deriving it from `x0`.

## Status values

| Status | When |
|---|---|
| `XtolReached` | the simplex radius reached the x tolerance |
| `FtolReached` | the spread of `f` over the simplex reached the f tolerance |
| `StopValueReached` | an evaluation returned `f <= stop_value` |
| `MaxEvaluationsReached` | the cap |
| `RoundoffLimited` | the simplex radius reached the precision floor, or a shrink moved no vertex |

Whatever the status, the result is the best point evaluated. A non-finite
objective value ends the run with `std::runtime_error`, where the compiler
lets the value reach memory, and so does a point beyond the coordinate
range.

## Verification against the paper

The test suite holds the implementation to section 2 of Lagarias et al.
rule by rule. In exact arithmetic (a power-of-two dimension, a dyadic start
and step, and every coefficient of both sets dyadic) a test chooses the
objective's value at every point the method evaluates, so each acceptance
test runs on both sides of its inequality and at its boundary, and the
points the method evaluates next are compared with (2.4) to (2.7) and step
5 to the bit:

- a reflection is accepted for `f(x_1) <= f(x_r) < f(x_n)`, including
  `f(x_r) = f(x_1)`, and `f(x_r) = f(x_n)` goes to the outside contraction
  (page 116);
- the expansion point is kept only when `f(x_e) < f(x_r)`, compared with
  `f(x_r)` and not with `f(x_1)` (page 116, and section 3.1, property 4);
- the outside contraction is accepted for `f(x_c) <= f(x_r)` (2.6) and the
  inside one for `f(x_cc) < f(x_{n+1})` (2.7), with `f(x_r) = f(x_{n+1})`
  already an inside contraction;
- the nonshrink ordering rule reproduces page 118's example, `(1, 2, 2, 3,
  3)` with `f(v) = 2`;
- after a shrink, a new vertex tying `x_1` does not displace it, and the
  other ties keep the order their originals had; the initial simplex keeps
  construction order on ties.

The nonshrink ordering rule is printed on page 116 as
`j = max{ l | f(v) < f(x_{l+1}) }`, which taken literally is always `n`,
since every accepted point is better than `x_{n+1}`. The sentence above it
and the example on page 118 both define the smallest such `l`: the new
vertex goes after every kept vertex of equal value. That is the rule
implemented and tested.

In one dimension `x_n` is `x_1`, so the reflection interval is empty and no
iteration ends in step 2 (page 125). Gao and Han state their coefficients
(4.1) for `n >= 2`; at `n = 1` their shrink coefficient is zero, outside the
condition `0 < sigma < 1` of (2.1), which is why the standard values are used
there.

On whole runs, a slow transcription of section 2, which recomputes the
centroid, re-sorts the simplex and computes the radius exactly every
iteration, replays the implementation's evaluations. It must ask for the
same points, to the bit in exact arithmetic and to rounding elsewhere, and
stop at the same iteration. On those runs the suite also checks what the
papers prove: the structure (2.9) of a nonshrink step and the change index
(2.8); that the worst value falls within `n + 1` nonshrink iterations (page
118); Lemma 3.3 (2); the evaluation counts of section 3.1; the volume of
Lemma 3.1; the diameter (4.1) in one dimension; that a strictly convex
function never causes a shrink (Lemma 3.5); the proximity bound of Lemma 4.3
in one dimension; and Gao and Han's sufficient descent (2.2) for the
standard coefficients.

Gao and Han's coefficients are checked against (4.1) for every `n` up to
4096, together with the conditions (2.1) and the `rho gamma < 1` of Lemma
3.6. From their Table 1's start and settings on their problem (4.3), the
standard coefficients stop far from the minimiser for `sigma = 1e-4` at 20
and 30 dimensions while the adaptive ones do not, as the table reports, and
the adaptive share of reflection steps on `x'x` stays at or below the 0.45
of their Figure 2.

Box bounds, the initial simplex, the stopping rules, the shrink in rounded
arithmetic and the coordinate range are FLOP's own and are listed under
deviations.

The ladder follows Lewis and Torczon's coordinate search on a box: the
directions `+-e_i` (their section 6.1, with the identity as basis and core
pattern, the diagonal core pattern their section 3.5 requires near a
bound), the step multiplied by `theta = 1/2` after a failed poll (Fig. 3.3),
and the best point of the whole poll taken, their strong hypothesis (Fig.
4.1). It differs in two ways. Their poll leaves out a pattern point outside
the box (Fig. 3.1); the ladder clips it onto the bound instead, which only
adds candidates, since every pattern point inside the box is still polled.
And the polls certify a stop rather than drive the search: an improvement
restarts Nelder-Mead, and each ladder starts again from `initial_step`.
Their convergence results (Theorems 4.2 and 4.3) are about the limit of
the step going to zero; a ladder stops at a finite scale, so it certifies
a stop at that scale and no more. The ladder is tested against this rule
and against a slow transcription of it.

## Limits

- The method has no convergence guarantee. Lagarias et al. prove convergence
  to the minimiser for strictly convex functions in one dimension and
  weaker results in two. K. I. M. McKinnon, "Convergence of the Nelder-Mead
  simplex method to a nonstationary point", SIAM Journal on Optimization
  9(1), 1998, pp. 148-158, constructs a strictly convex function in two
  dimensions on which the method converges to a point that is not a
  minimiser. FLOP does not restart on convergence except through the ladder
  on a box the run has touched, as described above.
- The ladder runs only before a stop. A simplex flattened against a face
  whose values differ only by rounding can keep reflecting without ever
  meeting a stopping test; such a run ends at the cap with
  `MaxEvaluationsReached`, never with a converged status.
- The ladder costs evaluations: up to `2n` per rung, and about
  `log2(initial_step / scale)` rungs before a stop stands, where `scale` is
  the stop's scale above.
- The ladder certifies stops on a box only. A simplex that degenerates in
  the interior, which exact arithmetic rules out and rounding does not,
  is not detected.
- With `-ffinite-math-only` in the caller's translation unit, a NaN the
  objective returns may never reach the check. Clang declares every
  `double` a function returns or takes by value free of NaN and infinity,
  so such a value is undefined before FLOP sees it; GCC keeps the bits and
  the check catches them. `x0`, the bounds and the tolerances are read from
  memory and are caught under every compiler and model.
