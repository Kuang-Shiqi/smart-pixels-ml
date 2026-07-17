"""
Produce paper-style residual and pull plots comparing the ViT_Max teacher
and the three distilled symbolic+tiny-NN students. Output to
runs/distill_sweep_vit_teacher/plots/.

Residual = truth - prediction (per output)
Pull     = residual / sigma  (per output; sigma from the parquet)

Both are histogrammed per output and overlaid with a Gaussian fit.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

REPO = "/work/users/das214/SmartPixels/smart-pixels-ml"
SWEEP = f"{REPO}/runs/distill_sweep_vit_teacher"
PLOTS = f"{SWEEP}/plots"
os.makedirs(PLOTS, exist_ok=True)

PARQUETS = {
    "ViT_Max teacher":      f"{REPO}/runs/processed_parquets/test_3src/2bit_optimized/2t-ViT_Max-2bit_optimized-6cb36456-vars.parquet",
    "Conv2D_Max baseline":  f"{REPO}/runs/processed_parquets/test_3src/2bit_optimized/2t-Conv2D_Max-2bit_optimized-961a8dfa-vars.parquet",
    "student (barycenter)": f"{REPO}/runs/processed_parquets/test_3src/2bit_optimized_students/2t-Student_barycenter-distilled-vit_teacher-vars.parquet",
    "student (localreco)":  f"{REPO}/runs/processed_parquets/test_3src/2bit_optimized_students/2t-Student_localreco-distilled-vit_teacher-vars.parquet",
    "student (blend)":      f"{REPO}/runs/processed_parquets/test_3src/2bit_optimized_students/2t-Student_blend-distilled-vit_teacher-vars.parquet",
}

COLORS = {
    "ViT_Max teacher":      "tab:red",
    "Conv2D_Max baseline":  "tab:gray",
    "student (barycenter)": "tab:blue",
    "student (localreco)":  "tab:orange",
    "student (blend)":      "tab:green",
}

OUTPUTS = [
    dict(name="x",     resid_col="residuals_x",    sigma_col="sigmax",    unit="(scaled units)"),
    dict(name="y",     resid_col="residuals_y",    sigma_col="sigmay",    unit="(scaled units)"),
    dict(name="cotA",  resid_col="residuals_cotA", sigma_col="sigmacotA", unit="(scaled units)"),
    dict(name="cotB",  resid_col="residuals_cotB", sigma_col="sigmacotB", unit="(scaled units)"),
]


def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def fit_gauss(values, n_bins=80, range_clip=None):
    if range_clip is None:
        lo, hi = np.percentile(values, [0.5, 99.5])
    else:
        lo, hi = range_clip
    counts, edges = np.histogram(values, bins=n_bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    A0 = float(counts.max())
    mu0 = float(np.median(values))
    sigma0 = float(np.std(values))
    try:
        popt, _ = curve_fit(gauss, centers, counts, p0=(A0, mu0, sigma0), maxfev=4000)
    except Exception:
        popt = (A0, mu0, sigma0)
    return counts, edges, centers, popt


def panel(ax, dfs, col, title, n_bins=80, x_range=None):
    summary = {}
    for label, df in dfs.items():
        values = df[col].to_numpy()
        values = values[np.isfinite(values)]
        if x_range is None:
            lo, hi = np.percentile(values, [0.5, 99.5])
        else:
            lo, hi = x_range
        counts, edges, centers, popt = fit_gauss(values, n_bins=n_bins, range_clip=(lo, hi))
        ax.step(centers, counts, where='mid', color=COLORS[label], label=label, lw=1.5, alpha=0.85)
        xs = np.linspace(lo, hi, 200)
        ax.plot(xs, gauss(xs, *popt), color=COLORS[label], lw=1.0, ls=':')
        summary[label] = dict(mu=float(popt[1]), sigma=float(abs(popt[2])),
                              n=int(values.size))
    ax.set_title(title); ax.set_xlabel(col); ax.set_ylabel('events')
    ax.axvline(0, color='k', lw=0.4, alpha=0.4)
    ax.legend(fontsize=7)
    return summary


def main():
    dfs = {k: pd.read_parquet(v) for k, v in PARQUETS.items()}
    for k, df in dfs.items():
        print(f"{k}: {len(df):,} rows")

    # ---- residual figure (truth - pred) ----
    summary_res = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, out in zip(axes.flat, OUTPUTS):
        s = panel(ax, dfs, out['resid_col'],
                  f"residual {out['name']} = truth - pred {out['unit']}")
        summary_res[out['name']] = s
    fig.suptitle("Residuals: teacher vs baseline vs students", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/residuals_overlay.png", dpi=130)
    plt.close(fig)

    # ---- pull figure (residual / sigma) ----
    for df in dfs.values():
        for out in OUTPUTS:
            df['pull_' + out['name']] = df[out['resid_col']] / df[out['sigma_col']]
    summary_pull = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, out in zip(axes.flat, OUTPUTS):
        s = panel(ax, dfs, 'pull_' + out['name'],
                  f"pull {out['name']} = residual / sigma", x_range=(-5, 5))
        summary_pull[out['name']] = s
        ax.axvline(-1, color='k', lw=0.3, ls='--', alpha=0.4)
        ax.axvline(1, color='k', lw=0.3, ls='--', alpha=0.4)
    fig.suptitle("Pulls: teacher vs baseline vs students (ideal: mu=0, sigma=1)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/pulls_overlay.png", dpi=130)
    plt.close(fig)

    # ---- summary aggregate (mean residual + pull width per model + output) ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    labels = list(PARQUETS.keys())
    xs = np.arange(len(labels))
    for ax, out in zip(axes.flat, OUTPUTS):
        means = [summary_res[out['name']][l]['mu'] for l in labels]
        sigmas = [summary_res[out['name']][l]['sigma'] for l in labels]
        cs = [COLORS[l] for l in labels]
        ax.errorbar(xs, means, yerr=sigmas, fmt='o', color='k', ecolor='gray')
        for i, (m, s, c) in enumerate(zip(means, sigmas, cs)):
            ax.errorbar(i, m, yerr=s, fmt='o', color=c, ecolor=c, capsize=4)
        ax.axhline(0, color='k', lw=0.4)
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_title(f"residual {out['name']}: Gaussian fit mu +/- sigma")
        ax.grid(alpha=0.3)
    fig.suptitle("Summary: residual bias and width per model and output", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/residual_summary_grid.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, out in zip(axes.flat, OUTPUTS):
        mus = [summary_pull[out['name']][l]['mu'] for l in labels]
        sgs = [summary_pull[out['name']][l]['sigma'] for l in labels]
        cs = [COLORS[l] for l in labels]
        for i, (m, s, c) in enumerate(zip(mus, sgs, cs)):
            ax.errorbar(i, m, yerr=s, fmt='o', color=c, ecolor=c, capsize=4)
        ax.axhline(0, color='k', lw=0.4)
        ax.axhline(1, color='k', lw=0.3, ls='--')
        ax.axhline(-1, color='k', lw=0.3, ls='--')
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_title(f"pull {out['name']}: Gaussian mu +/- sigma  (ideal: 0, 1)")
        ax.grid(alpha=0.3)
    fig.suptitle("Summary: pull mean and width per model and output", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/pull_summary_grid.png", dpi=130)
    plt.close(fig)

    json.dump({"residuals": summary_res, "pulls": summary_pull},
              open(f"{SWEEP}/fit_summary.json", "w"), indent=1, default=float)

    # print a compact table
    print("\n=== pull fit summary (mu, sigma; ideal 0, 1) ===")
    for out in OUTPUTS:
        print(f"  {out['name']}")
        for l in labels:
            s = summary_pull[out['name']][l]
            print(f"    {l:30s} mu={s['mu']:+.3f}  sigma={s['sigma']:.3f}")
    print(f"\nplots in {PLOTS}/")
    print(f"raw fits in {SWEEP}/fit_summary.json")


if __name__ == '__main__':
    main()
