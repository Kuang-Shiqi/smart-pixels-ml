# The model as a formula

Every number below is the trained value from `exports/digi_2bit_paper_code_nexp1/constants.json`.
Nothing is left implicit: `tan(theta_L)` is already folded into two literals, because at inference
`theta_L` is a constant and the hardware should never see a tangent.

## Input

`D[r][c][t]` — a **2-bit code, 0..3**, at row `r`, column `c`, time slice `t` (t = 0, 1).

```
row r -> y, pitch p_y = 12.5 um        (16 rows)
col c -> x, pitch p_x = 50   um        (16 cols)
sensor thickness T = 100 um
centers:  X[c] = (c - 7.5) * 50.0      Y[r] = (r - 7.5) * 12.5
```

The row/col assignment is **measured**, not assumed (50k events: col centroid -> label x, corr
0.927; row centroid -> label y, corr 0.914; cross terms 0.002). An earlier version of this model
had it transposed; see README.

If the ADC is inside your synthesis scope, the codes come from three comparators:

```
code = (q > 248.0) + (q > 668.0) + (q > 1663.0)          q = charge in electrons
```

strictly greater, thresholds in electrons. These are the published values from
arXiv:2602.15946 Fig. 3 (+- 6 / 8 / 39), used verbatim.

## N = 1 — the whole model, 376 parameters

### Profiles (pure accumulation)

```
q[r][c]   = D[r][c][0] + D[r][c][1]
Px[c]     = sum_r q[r][c]                 # 16 values, column-indexed
Py[r]     = sum_c q[r][c]                 # 16 values, row-indexed
```

### Positions — charge-weighted barycenter (2 divisions)

```
x_raw = ( sum_c Px[c]*X[c] ) / ( sum_c Px[c] + 1e-6 )
y_raw = ( sum_r Py[r]*Y[r] ) / ( sum_r Py[r] + 1e-6 ) + 4.232469
                                                        ^^^^^^^^ = T*tan(theta_L)/2, um
```

### Angle magnitudes — cluster width (no division)

Cluster width is an **even** function of incidence angle, so this can only produce `|cot|`:

```
Wx = #{ c : Px[c] > 0 }
Wy = #{ r : Py[r] > 0 }
cotA_abs = max(Wx - 1, 0) * 0.5        # = p_x/T = 50/100
cotB_abs = max(Wy - 1, 0) * 0.125 + 0.023738
                             ^^^^^      ^^^^^^^^ = |lorentz_scale * tan(theta_L)|
                             = p_y/T = 12.5/100
```

### Sign — between-slice centroid drift (4 divisions, 2 tanh)

The sign lives in the spatially-even x time-odd sector. Every time-summed spatial moment is
provably blind to it, so it must come from the drift between the two time slices:

```
Cx(t) = ( sum_c (sum_r D[r][c][t]) * X[c] ) / ( sum_{r,c} D[r][c][t] + 1e-6 )
Cy(t) = ( sum_r (sum_c D[r][c][t]) * Y[r] ) / ( sum_{r,c} D[r][c][t] + 1e-6 )

Tx = clip( Cx(1) - Cx(0), -800, +800 )      # um
Ty = clip( Cy(1) - Cy(0), -200, +200 )      # um

sign_A = tanh( +0.221059 * ( -0.144608 - Tx ) )
sign_B = tanh( +0.172958 * ( -3.747275 - Ty ) )
```

Each angle is signed by the drift along **its own** axis. Do not cross them — an earlier version
did, as a consequence of the transposed axes.

### Output scaling

```
raw = [ x_raw, y_raw, cotA_abs*sign_A, cotB_abs*sign_B ]  /  labels_scale
labels_scale = [ 123.410162, 30.929850, 6.577499, 1.929565 ]

out[i] = aff_scale[i] * raw[i] + aff_bias[i]
aff_scale = [ 0.945381, 0.946112, 1.011223, 0.988519 ]
aff_bias  = [ 0.203415, 0.548133, 0.000131, -0.161913 ]
```

`out` is in **normalized** units. Multiply by `labels_scale` for um (x, y) or for the raw cot
value (cotA, cotB); the angle in degrees is `degrees(atan2(1, cot))`.

### Uncertainty head — 354 parameters, a plain MLP

```
xt[r][c] = ( D[r][c][0] + D[r][c][1] ) / 2
feat     = concat( mean_c xt[r][:] (16 values), mean_r xt[:][c] (16 values) )   # 32
h        = tanh( feat @ W1 + b1 )                                              # 8
chol     = h @ W2 + b2                                                         # 10
```

`W1 (32x8), b1 (8), W2 (8x10), b2 (10)` are in `constants.json`. The 14-vector is
`[x, chol0, y, chol1, cotA, chol2, cotB, chol3, chol4..chol9]`; the first four chol entries are
the Cholesky **diagonal before softplus**, the last six are off-diagonals.

If the covariance is not needed on-chip, this whole head can be dropped — it does not feed the
means. That takes the model to 22 physics parameters.

## Per-event op count, N = 1

| op | count | note |
|---|---|---|
| add | ~1000 | profile accumulation, dominated by 16x16x2 |
| multiply | ~100 | centroid weighting; `X[c]`, `Y[r]` are constants |
| divide | 6 | 2 barycenters + 4 for the drifts |
| compare | 32 | the two width counts |
| tanh | 2 | sign gates. Small LUT is fine — the output only needs to be right near 0 |
| MAC | 336 | the uncertainty MLP, if kept |

No square root, no logarithm, no exponential, no trigonometry anywhere in the mean path.

## N = 4 — MoE, 3272 parameters

Four copies of the above (each with **its own** 18 scalars — they specialize; expert 0 sits at
`theta_L = +49 deg`, expert 3 at `+14 deg`), blended by a softmax router:

```
means = sum_k w[k] * ansatz_k(D)
chol  = sum_k w[k] * mlp_k(D)
```

The router is the only new machinery, and it is where the HLS-awkward ops are:

```
features (36) = [ Px/total (16), Py/total (16), std, skew, drift, log(total+1) ]
      std   = sqrt( m2 + 1 )            m2, m3 = 2nd, 3rd moments of the normalized y-profile
      skew  = m3 / std^3
      drift = Cy(1) - Cy(0)  in PIXEL units (not um)
      clip all 36 features to [-8, +8]
w = softmax( ( relu( relu( features @ R1 ) @ R2 ) @ R3 ) / 1.0 )      # 36->32->16->4
```

So the router needs a `sqrt`, a `log`, a cube and a `softmax`; the experts need none of those.
All four experts are genuinely used — router occupancy 0.196 / 0.253 / 0.320 / 0.231, entropy
1.317 against a maximum of 1.386.

**Suggested order:** synthesize N=1 first — it is the entire pipeline with none of the transcendental
ops, and it validates the ADC, the datapath and the fixed-point choices. Then add the router.
