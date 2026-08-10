"""
Input digitization for the symbolic student.

Deployment reality is an on-sensor **two-bit** ADC, so the quantizer has to sit at
the very front of the model -- ahead of RouterFeatures AND the experts -- and be
differentiable enough to train through.

This module is a thin front-end over SoftQuantizeLayer
(two_bit_optimization_helpers/SoftQuantizeLayer.py), which implements the parts
that are easy to get wrong: thresholds in physical charge units, monotone by
construction, hard quantization on the forward pass at all times, and an STE
whose soft surrogate is a difference of sigmoid CDFs with a per-threshold local
scale `tau`.


PUBLISHED CONSTANTS  (arXiv:2602.15946, Section IV -- use these, do not re-derive)
---------------------------------------------------------------------------------
    sigma_noise = 80 e-        detector + electronics noise, from a Cadence
                               Virtuoso design simulated with Spectre (Sec. IV B).
    5 sigma     = 400 e-       the upward fluctuation at which pixel detectors
                               conventionally place their lowest threshold.
    T = (248, 668, 1663) e-    two-bit thresholds learned by the Max transformer
        +- ( 6,   8,   39)     with SoftQuantize, averaged over trainings from
                               random initialisations (Fig. 3). The paper then
                               used these as the *initial* values when optimising
                               thresholds for the resource-constrained models,
                               whose preferred minimum threshold came out higher
                               than the transformer's but still below 5 sigma.

The paper digitizes to the **bin index 0..3** (Eq. 1). That is `level_mode='code'`
here, and it is the only apples-to-apples configuration against the published
baselines. `level_mode='midpoint'` assigns the bin-centre charge instead, which
assumes a 4-entry code->charge LUT on-chip that the baselines do not have; it is
a legitimate design option but any performance credit it earns was bought with
chip area and must be reported that way.

WHY THIS MODULE EXISTS IN ITS CURRENT FORM
------------------------------------------
An earlier version of this study placed thresholds by occupancy on our own
training charge and got T = (100, 240, 1295) e-. T0 = 100 e- is 1.25 sigma_noise
above pedestal -- i.e. it digitizes noise as signal -- and the whole set sits
roughly one slot below the published one. Those numbers are not comparable to the
paper. Hence: `scheme='paper'` is the default, and every run records where its
thresholds came from (`describe()`), so this cannot go unnoticed again.

Order of operations is physical and fixed: noise is added to the ANALOG charge
and the result is digitized (Eq. 6). The data generator adds the noise
(`noise=[mu, sigma]`, `-1` means OFF -- verified in OptimizedDataGenerator_v3
`_read_tfrecord`, which applies it before its own quantize/digitize steps) and
must be left with `digitize=False` so this layer is the only quantizer.
"""
import numpy as np
import tensorflow as tf

from SoftQuantizeLayer import SoftQuantizeLayer

# ---- published constants (arXiv:2602.15946 Sec. IV, Fig. 3) ----
SIGMA_NOISE_E = 80.0
FIVE_SIGMA_E = 5.0 * SIGMA_NOISE_E                       # 400 e-
PAPER_THRESHOLDS_2BIT = (248.0, 668.0, 1663.0)
PAPER_THRESHOLD_STD_2BIT = (6.0, 8.0, 39.0)
PAPER_REF = "arXiv:2602.15946 Sec. IV / Fig. 3 (Max transformer, SoftQuantize)"


def build_thresholds(x, n_bits, scheme='paper', offset=SIGMA_NOISE_E,
                     t0_min=None, hi_q=0.995):
    """Place 2^n_bits - 1 thresholds in charge units (electrons).

    Returns (thresholds, bin_edges, provenance) where `provenance` is a dict that
    goes verbatim into the run's summary.json -- it is the record of whether the
    numbers are published or derived, which is the thing that was missing before.

    scheme
      'paper'    published two-bit values, used verbatim; `x` is ignored.
                 Only defined for n_bits=2.
      'lowfirst' T0 pinned at `t0_min`, the rest spread by occupancy over what
                 survives it. Preserves cluster width, which plain equal
                 occupancy erodes -- the cluster edge is what sets |cot b| on the
                 fine 12.5 um axis.
      'quantile' equal occupancy among above-offset pixels.
      'linear'   uniform between `offset` and the hi_q quantile.

    t0_min
      floor for the lowest threshold. Defaults to the published T0 (248 e-) for
      the derived schemes, so a data-driven placement can never put T0 down in
      the noise again. Pass an explicit value to override.
    """
    n_levels = 2 ** int(n_bits)
    notes = []

    if scheme == 'paper':
        if int(n_bits) != 2:
            raise ValueError(
                f"scheme='paper' is only defined for n_bits=2; the paper publishes "
                f"no {n_bits}-bit thresholds. Use scheme='lowfirst', which floors "
                f"T0 at the published 248 e- and derives the rest from occupancy.")
        thr = np.asarray(PAPER_THRESHOLDS_2BIT, dtype=np.float32)
        source = f"published verbatim, {PAPER_REF}"
        published = list(PAPER_THRESHOLDS_2BIT)
        published_std = list(PAPER_THRESHOLD_STD_2BIT)
    else:
        v = np.asarray(x, dtype=np.float32).ravel()
        v = v[v > offset]
        if v.size == 0:
            raise ValueError("no charge above offset -- wrong units or offset too high")
        t0 = float(PAPER_THRESHOLDS_2BIT[0] if t0_min is None else t0_min)
        source = (f"occupancy-derived from training charge (scheme={scheme!r}); "
                  f"NOT published. T0 floored at {t0:.1f} e-")
        published = None
        published_std = None
        notes.append(
            f"No published {n_bits}-bit thresholds exist; only the two-bit set is "
            f"published. These are derived and are not comparable to the paper's "
            f"numbers on the same footing as the two-bit point.")

        if scheme == 'quantile':
            qs = np.arange(1, n_levels) / n_levels
            thr = np.quantile(v, qs).astype(np.float32)
        elif scheme == 'linear':
            hi = float(np.quantile(v, hi_q))
            thr = np.linspace(offset, hi, n_levels + 1)[1:-1].astype(np.float32)
        elif scheme == 'lowfirst':
            rest = v[v > t0]
            if rest.size == 0 or n_levels == 2:
                thr = np.array([t0], dtype=np.float32)
            else:
                qs = np.arange(1, n_levels - 1) / (n_levels - 1)
                thr = np.concatenate([[t0], np.quantile(rest, qs)]).astype(np.float32)
        else:
            raise ValueError(f"unknown scheme {scheme!r}")

        thr = np.maximum(thr, t0)

    # SoftQuantizeLayer requires offset < T0 < T1 < ... strictly increasing.
    thr = np.maximum.accumulate(np.maximum(thr, offset + 1.0)).astype(np.float32)
    eps = np.arange(thr.size, dtype=np.float32) * 1e-3
    thr = (thr + eps).astype(np.float32)
    assert thr.size == n_levels - 1, f"n_bits={n_bits} needs {n_levels - 1} thresholds"

    hi_edge = (float(np.quantile(np.asarray(x, np.float32).ravel(), hi_q))
               if scheme != 'paper' else float(thr[-1]) * 3.0)
    edges = np.concatenate(
        [[offset], thr, [max(hi_edge, float(thr[-1]) + 1.0)]]).astype(np.float32)

    t0_sigma = float(thr[0]) / SIGMA_NOISE_E
    if float(thr[0]) < 3.0 * SIGMA_NOISE_E:
        notes.append(f"WARNING: T0 = {thr[0]:.1f} e- is only {t0_sigma:.2f} sigma_noise "
                     f"above pedestal -- this digitizes noise as signal.")

    provenance = dict(
        n_bits=int(n_bits),
        scheme=scheme,
        source=source,
        thresholds_e=[float(t) for t in thr],
        published_thresholds_e=published,
        published_threshold_std_e=published_std,
        paper_reference=PAPER_REF,
        offset_e=float(offset),
        t0_min_e=None if scheme == 'paper' else float(
            PAPER_THRESHOLDS_2BIT[0] if t0_min is None else t0_min),
        sigma_noise_e=SIGMA_NOISE_E,
        five_sigma_e=FIVE_SIGMA_E,
        t0_over_sigma_noise=t0_sigma,
        t0_below_five_sigma=bool(float(thr[0]) < FIVE_SIGMA_E),
        notes=notes,
    )
    return thr, edges, provenance


def thresholds_from_data(x, n_bits, offset=SIGMA_NOISE_E, scheme='lowfirst', hi_q=0.995):
    """Back-compat wrapper. Prefer build_thresholds() -- it returns provenance."""
    thr, edges, _ = build_thresholds(x, n_bits, scheme=scheme, offset=offset, hi_q=hi_q)
    return thr, edges


def levels_for(n_bits, mode='code', bin_edges=None):
    """Output levels for a 2^n_bits quantizer.

    'code'     -> 0..2^n-1, the bin index. This is the paper (Eq. 1) and the only
                  configuration comparable to the published baselines.
    'midpoint' -> the charge at the middle of each bin. Assumes an on-chip
                  code->charge LUT the baselines do not have.
    """
    n_levels = 2 ** int(n_bits)
    if mode == 'code':
        return np.arange(n_levels, dtype=np.float32)
    if mode == 'midpoint':
        if bin_edges is None:
            raise ValueError("mode='midpoint' needs bin_edges from build_thresholds")
        e = np.asarray(bin_edges, dtype=np.float32)
        assert e.size == n_levels + 1, f"bin_edges must have 2^n_bits+1 entries, got {e.size}"
        lv = 0.5 * (e[:-1] + e[1:])
        # Level 0 is the below-threshold bin: it MUST read exactly zero. A bin
        # midpoint here puts a pedestal on every empty pixel, which makes every
        # cluster look 16 pixels wide and buries the centroid drift.
        lv[0] = 0.0
        return lv.astype(np.float32)
    raise ValueError(f"unknown level mode {mode!r}")


class DigitizeLayer(tf.keras.layers.Layer):
    """Front-of-model ADC. `n_bits=None` is an identity passthrough.

    n_bits               : bit depth, or None for analog
    thresholds           : (2^n-1,) charge thresholds; required when n_bits is set
    levels               : (2^n,) output levels; defaults to codes 0..2^n-1
    threshold_offset     : T_min in Eq. 3 -- the lower bound the cumulative sum
                           starts from. When thresholds are trainable this is the
                           floor the learned T0 cannot fall below, so it is the
                           knob that keeps T0 out of the noise.
    ste                  : True  -> straight-through estimator while training
                           False -> hard quantize with no gradient path
    trainable_thresholds : let the thresholds co-train (paper Sec. IV)
    trainable_levels     : let the output levels move. Leave False: the ADC emits
                           a bin index, it does not get to choose its own scale.
    initial_k            : softness of the STE surrogate. The paper starts near
                           k=1 and anneals to k~67; a large fixed k gives the
                           thresholds almost no gradient, so use ~1 with an
                           annealing callback whenever trainable_thresholds=True.
    parametrization      : 'paper'  -> Delta = softplus(theta)  (Eq. 2), correct
                                       gradient scale
                           'legacy' -> the historical softplus(expm1(theta)) form
                           Only affects trainable thresholds. See SoftQuantizeLayer.
    """

    def __init__(self, n_bits=2, thresholds=None, levels=None,
                 threshold_offset=SIGMA_NOISE_E, ste=True,
                 trainable_thresholds=False, trainable_levels=False,
                 initial_k=50.0, trainable_k=False, parametrization='paper',
                 name='digitize', **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_bits = None if n_bits is None else int(n_bits)
        self.ste = bool(ste)
        self.threshold_offset = float(threshold_offset)
        self.initial_k = float(initial_k)
        self.trainable_k = bool(trainable_k)
        self.trainable_thresholds = bool(trainable_thresholds)
        self.trainable_levels = bool(trainable_levels)
        self.parametrization = parametrization

        if self.n_bits is None:
            self.quantizer = None
            self.thresholds_value = None
            self.levels_value = None
            return

        if thresholds is None:
            raise ValueError("n_bits set but thresholds is None -- build them with "
                             "build_thresholds(x, n_bits, scheme='paper')")
        thr = np.asarray(thresholds, dtype=np.float32).ravel()
        n_levels = 2 ** self.n_bits
        assert thr.size == n_levels - 1, \
            f"n_bits={self.n_bits} needs {n_levels - 1} thresholds, got {thr.size}"
        lv = (np.arange(n_levels, dtype=np.float32) if levels is None
              else np.asarray(levels, dtype=np.float32).ravel())
        assert lv.size == n_levels, f"n_bits={self.n_bits} needs {n_levels} levels, got {lv.size}"

        if self.trainable_thresholds and self.initial_k > 5.0:
            raise ValueError(
                f"trainable_thresholds=True with initial_k={self.initial_k}: the soft "
                f"sigmoids are already step-like, so the thresholds get almost no "
                f"gradient. The paper starts at k~1 and anneals to k~67 -- pass "
                f"initial_k=1.0 and attach anneal_k().")

        self.thresholds_value = thr
        self.levels_value = lv
        self.quantizer = SoftQuantizeLayer(
            n_bits=self.n_bits,
            initial_levels=lv,
            initial_thresholds=thr,
            threshold_offset=self.threshold_offset,
            trainable_levels=self.trainable_levels,
            trainable_thresholds=self.trainable_thresholds,
            initial_k=self.initial_k,
            trainable_k=self.trainable_k,
            parametrization=self.parametrization,
            name=f'{name}_soft_quantizer',
        )

    def call(self, x, training=None):
        if self.quantizer is None:
            return x
        # SoftQuantizeLayer: hard forward always; STE only when training=True.
        return self.quantizer(x, training=bool(training) and self.ste)

    @property
    def thresholds(self):
        return None if self.quantizer is None else self.quantizer.thresholds

    @property
    def levels(self):
        return None if self.quantizer is None else self.quantizer.levels

    def describe(self, provenance=None):
        """Everything needed to reproduce this quantizer, for summary.json."""
        if self.quantizer is None:
            return dict(n_bits=None, mode="analog passthrough",
                        sigma_noise_e=SIGMA_NOISE_E)
        thr_now = [float(t) for t in np.asarray(self.thresholds)]
        d = dict(
            n_bits=self.n_bits,
            level_mode=("code" if np.allclose(self.levels_value,
                                              np.arange(len(self.levels_value)))
                        else "midpoint/custom"),
            levels_initial=[float(v) for v in self.levels_value],
            levels_final=[float(v) for v in np.asarray(self.levels)],
            thresholds_initial_e=[float(v) for v in self.thresholds_value],
            thresholds_final_e=thr_now,
            thresholds_moved_e=[float(a - b) for a, b in
                                zip(thr_now, self.thresholds_value)],
            threshold_offset_e=self.threshold_offset,
            trainable_thresholds=self.trainable_thresholds,
            trainable_levels=self.trainable_levels,
            parametrization=self.parametrization,
            ste=self.ste,
            initial_k=self.initial_k,
            final_k=float(np.asarray(self.quantizer.k).reshape(-1)[0]),
            trainable_k=self.trainable_k,
            sigma_noise_e=SIGMA_NOISE_E,
            five_sigma_e=FIVE_SIGMA_E,
            published_thresholds_e=list(PAPER_THRESHOLDS_2BIT) if self.n_bits == 2 else None,
            paper_reference=PAPER_REF,
        )
        if self.n_bits == 2:
            d["offset_from_published_e"] = [float(a - b) for a, b in
                                            zip(thr_now, PAPER_THRESHOLDS_2BIT)]
        if provenance is not None:
            d["threshold_provenance"] = provenance
        return d

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'n_bits': self.n_bits,
            'thresholds': (None if self.thresholds_value is None
                           else self.thresholds_value.tolist()),
            'levels': (None if self.levels_value is None
                       else self.levels_value.tolist()),
            'threshold_offset': self.threshold_offset,
            'ste': self.ste,
            'trainable_thresholds': self.trainable_thresholds,
            'trainable_levels': self.trainable_levels,
            'initial_k': self.initial_k,
            'trainable_k': self.trainable_k,
            'parametrization': self.parametrization,
        })
        return cfg


class AnnealK(tf.keras.callbacks.Callback):
    """Cosine-anneal the SoftQuantize temperature k, per the paper (Sec. IV A).

    The paper starts at k_init ~ 1 -- sigmoids soft enough that the thresholds
    actually receive gradient -- and raises it to k_max ~ 67 so the surrogate
    converges to the hard step by the end of training.

    Takes the layer by REFERENCE. The stock AnnealingScheduler looks the layer up
    with model.get_layer(name), which cannot reach a sublayer of a subclassed
    model (same failure mode as get_layer("physics_ansatz") through a wrapper).
    """

    def __init__(self, digitize_layer, k_init=1.0, k_max=67.0,
                 total_epochs=None, verbose=0):
        super().__init__()
        self.q = getattr(digitize_layer, "quantizer", digitize_layer)
        self.k_init, self.k_max = float(k_init), float(k_max)
        self.total_epochs, self.verbose = total_epochs, verbose
        self.history = []

    def on_train_begin(self, logs=None):
        if self.total_epochs is None:
            self.total_epochs = int(self.params.get("epochs", 1))

    def on_epoch_begin(self, epoch, logs=None):
        if self.q is None:
            return
        r = 0.5 * (1.0 - np.cos(np.pi * min(epoch / max(self.total_epochs - 1, 1), 1.0)))
        k = self.k_init + (self.k_max - self.k_init) * r
        self.q.log_k.assign([np.log(np.float32(k))])
        thr = [float(t) for t in np.asarray(self.q.thresholds)]
        self.history.append(dict(epoch=int(epoch), k=float(k), thresholds_e=thr))
        if self.verbose and epoch % max(self.total_epochs // 10, 1) == 0:
            print(f"  [anneal] epoch {epoch:3d}  k={k:6.2f}  T={np.round(thr, 1)}")
