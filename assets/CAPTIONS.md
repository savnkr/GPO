# Figures and animations

**`media/matheron_toy.*`, `media/matheron_toy_panels.png`.** Matheron's rule on a one-dimensional Gaussian process
(Matérn-5/2 kernel, length scale 0.11, noise 0.06). A prior draw $f$ is compared with the observations at their inputs,
giving residuals $r = y - f(X) - \varepsilon$ with $\varepsilon$ a draw of the observation noise. The kernel spreads the
residuals into the correction $k(\cdot, X)K^{-1}r$, which matches them at the data and decays away from it. Adding it
gives a posterior draw: it follows the data where there is data and keeps the prior's variability elsewhere. Repeating
with fresh $(f, \varepsilon)$ gives independent posterior draws, which fill the exact posterior band.

**`media/matheron_<benchmark>.*`.** The same rule on a held-out test input: four joint prior draws of the output field
morph into posterior draws, $f + t\,k(\cdot, X)K^{-1}(y - f(X) - \varepsilon)$ for $t$ from 0 to 1. Each posterior draw is
an exact sample of the GPO posterior.

**`media/gallery_<benchmark>.*`.** Six test cases spread over the error distribution (10th to 90th percentile),
crossfaded in turn. Panels: input; truth; posterior mean; the actual error (truth − mean) next to a sampled error (one
posterior draw − mean) on the same colour scale; the points where the truth leaves the 95% interval; and, along the
dashed line, both errors with the 95% interval. If the uncertainty is calibrated, the actual error looks like another
sample: same size, same smoothness, outside the interval at about 5% of points. The simulated truth is noise-free, so
the comparison uses the posterior of the field, scaled once on the validation fields. For one-dimensional problems:
input; truth and posterior mean; error of the mean with the 95% interval.

**`media/sensors_navier_stokes.*`.** Test-time sensor fusion on one Navier–Stokes test field. Sensors (noise variance
1e-6 in model units) are placed one at a time where the posterior variance is largest; the mean, a posterior draw and
the standard deviation are updated exactly after each reading.

**`figures/solvers_navier_stokes.png`.** Solving the GPO system on Navier–Stokes (3000 fields, 4096 grid points,
12.3 million unknowns). Left: dual suboptimality. Right: error of the resulting posterior mean against the exact
solve, on 20 test inputs. Conjugate gradients lowers the dual objective but its predictions oscillate; stochastic dual
descent improves the predictions steadily; solving the rank-c feature block exactly and running SDD on the Kronecker
block reaches the exact posterior mean to about 0.05% from the first iterate.

**`figures/matheron_convergence.png`.** Relative Frobenius error of the sample covariance of $S$ Matheron draws against
the exact posterior covariance (512 grid points of one Navier–Stokes test field). The error decreases as $S^{-1/2}$.

**`figures/sensor_fusion.png`.** Error (top) and 95% coverage (bottom) against the number of sensors, averaged over 50
test fields. Baselines: the WNO alone, and the WNO corrected by kriging its residual at the sensors with a
stationary Matérn kernel (length scale from the GPO's spatial kernel, variance from validation residuals).

**`figures/functionals.png`.** Coverage of 90% intervals for quantities computed from the predicted field (spatial mean,
energy, maximum, minimum, gradient energy) over the test set. Blue: joint Matheron draws. Orange: draws with the same
pointwise mean and variance but independent across grid points.

**`figures/data_scaling.png`.** Navier–Stokes, operator trained on 3000 fields and kept fixed. The GPO conditions on up
to 36 000 fields with stochastic dual descent; the exact solve runs out of memory beyond a few thousand.
