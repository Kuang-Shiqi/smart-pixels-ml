"""
Final-report generator for the 1000-epoch run:
  - sweep dir: runs/distill_sweep_vit_teacher_1000ep
  - teacher : runs/vit_max_run_1000ep
  - student parquets: runs/processed_parquets/test_3src/2bit_optimized_students_1000ep

Produces:
  runs/distill_sweep_vit_teacher_1000ep/REPORT.md      (paper-style writeup)
  runs/distill_sweep_vit_teacher_1000ep/plots/*.png    (training + residual + pull)
  runs/distill_sweep_vit_teacher_1000ep/fit_summary.json
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

REPO = "/work/users/das214/SmartPixels/smart-pixels-ml"
SWEEP = f"{REPO}/runs/distill_sweep_vit_teacher_1000ep"
TEACHER_DIR = f"{REPO}/runs/vit_max_run_1000ep"
STUDENT_PARQUETS_DIR = f"{REPO}/runs/processed_parquets/test_3src/2bit_optimized_students_1000ep"
TEACHER_PARQUET_DIR = f"{REPO}/runs/processed_parquets/test_3src/2bit_optimized"
BASELINE_SUMMARY = f"{REPO}/runs/conv2d_max_run/summary.json"
PLOTS = f"{SWEEP}/plots"
os.makedirs(PLOTS, exist_ok=True)

VARIANTS = ['barycenter', 'localreco', 'blend']
OUTPUTS = [
    dict(name="x",    resid_col="residuals_x",    sigma_col="sigmax"),
    dict(name="y",    resid_col="residuals_y",    sigma_col="sigmay"),
    dict(name="cotA", resid_col="residuals_cotA", sigma_col="sigmacotA"),
    dict(name="cotB", resid_col="residuals_cotB", sigma_col="sigmacotB"),
]
COLORS = {
    "ViT_Max teacher (1000ep)": "tab:red",
    "Conv2D_Max baseline":      "tab:gray",
    "student (barycenter)":     "tab:blue",
    "student (localreco)":      "tab:orange",
    "student (blend)":          "tab:green",
}


def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def fit_gauss(values, n_bins=80, x_range=None):
    v = values[np.isfinite(values)]
    if x_range is None:
        lo, hi = np.percentile(v, [0.5, 99.5])
    else:
        lo, hi = x_range
    counts, edges = np.histogram(v, bins=n_bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    p0 = (float(counts.max()), float(np.median(v)), float(np.std(v)) or 1.0)
    try:
        popt, _ = curve_fit(gauss, centers, counts, p0=p0, maxfev=4000)
    except Exception:
        popt = p0
    return counts, edges, centers, popt


def find_teacher_parquet():
    for f in os.listdir(TEACHER_PARQUET_DIR):
        if f.startswith("2t-ViT_Max-") and 'vars.parquet' in f and '1000ep' in f:
            return os.path.join(TEACHER_PARQUET_DIR, f)
    for f in os.listdir(TEACHER_PARQUET_DIR):
        if f.startswith("2t-ViT_Max-") and 'vars.parquet' in f:
            return os.path.join(TEACHER_PARQUET_DIR, f)
    raise FileNotFoundError("no teacher parquet")


def find_baseline_parquet():
    for f in os.listdir(TEACHER_PARQUET_DIR):
        if f.startswith("2t-Conv2D_Max-") and 'vars.parquet' in f:
            return os.path.join(TEACHER_PARQUET_DIR, f)
    return None


def main():
    teacher_sum = json.load(open(f"{TEACHER_DIR}/summary.json"))
    baseline_sum = json.load(open(BASELINE_SUMMARY)) if os.path.exists(BASELINE_SUMMARY) else None
    sweep_sum = json.load(open(f"{SWEEP}/sweep_summary.json"))

    # ---- training-curve plots (per variant) ----
    fig, ax = plt.subplots(figsize=(7, 4))
    for v, c in zip(VARIANTS, ['tab:blue','tab:orange','tab:green']):
        h = np.genfromtxt(f"{SWEEP}/{v}/history.csv", delimiter=',', names=True, dtype=float)
        ax.plot(h['epoch']+1, h['val_loss_data'], label=f"{v} (val)", color=c)
    ax.set_xlabel('epoch'); ax.set_ylabel('per-event NLL'); ax.set_title('val NLL per epoch')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/loss_vs_epoch.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for v, c in zip(VARIANTS, ['tab:blue','tab:orange','tab:green']):
        h = np.genfromtxt(f"{SWEEP}/{v}/history.csv", delimiter=',', names=True, dtype=float)
        ax.semilogy(h['epoch']+1, h['val_kl'], label=f"{v}", color=c)
    ax.set_xlabel('epoch'); ax.set_ylabel('val KL[T||S]'); ax.set_title('KL constraint per epoch')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(f"{PLOTS}/kl_vs_epoch.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for v, c in zip(VARIANTS, ['tab:blue','tab:orange','tab:green']):
        h = np.genfromtxt(f"{SWEEP}/{v}/history.csv", delimiter=',', names=True, dtype=float)
        ax.plot(h['epoch']+1, h['lam'], label=v, color=c)
    ax.set_xlabel('epoch'); ax.set_ylabel('lambda'); ax.set_title('MDMM lambda per epoch')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/lambda_vs_epoch.png", dpi=130); plt.close(fig)

    # ---- residual + pull plots ----
    parquets = {"ViT_Max teacher (1000ep)": find_teacher_parquet()}
    baseline_pq = find_baseline_parquet()
    if baseline_pq:
        parquets["Conv2D_Max baseline"] = baseline_pq
    for v in VARIANTS:
        p = f"{STUDENT_PARQUETS_DIR}/2t-Student_{v}-distilled-vit_teacher-vars.parquet"
        if os.path.exists(p):
            parquets[f"student ({v})"] = p
    dfs = {k: pd.read_parquet(v) for k, v in parquets.items()}
    for k, df in dfs.items():
        print(f"{k}: {len(df):,} rows")
        for out in OUTPUTS:
            df['pull_' + out['name']] = df[out['resid_col']] / df[out['sigma_col']]

    # residual overlay
    summary_res, summary_pull = {}, {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, out in zip(axes.flat, OUTPUTS):
        s = {}
        for label, df in dfs.items():
            v = df[out['resid_col']].to_numpy()
            counts, edges, centers, popt = fit_gauss(v)
            ax.step(centers, counts, where='mid', color=COLORS.get(label,'k'), label=label, lw=1.5, alpha=0.85)
            xs = np.linspace(centers[0], centers[-1], 200)
            ax.plot(xs, gauss(xs, *popt), color=COLORS.get(label,'k'), lw=1.0, ls=':')
            s[label] = dict(mu=float(popt[1]), sigma=float(abs(popt[2])))
        ax.set_title(f"residual {out['name']}"); ax.axvline(0, color='k', lw=0.4)
        ax.legend(fontsize=7); summary_res[out['name']] = s
    fig.suptitle("Residuals (1000-ep teacher)"); fig.tight_layout()
    fig.savefig(f"{PLOTS}/residuals_overlay.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, out in zip(axes.flat, OUTPUTS):
        s = {}
        for label, df in dfs.items():
            v = df['pull_' + out['name']].to_numpy()
            counts, edges, centers, popt = fit_gauss(v, x_range=(-5, 5))
            ax.step(centers, counts, where='mid', color=COLORS.get(label,'k'), label=label, lw=1.5, alpha=0.85)
            xs = np.linspace(-5, 5, 200)
            ax.plot(xs, gauss(xs, *popt), color=COLORS.get(label,'k'), lw=1.0, ls=':')
            s[label] = dict(mu=float(popt[1]), sigma=float(abs(popt[2])))
        ax.set_title(f"pull {out['name']} (ideal: mu=0, sigma=1)"); ax.axvline(0, color='k', lw=0.4)
        ax.axvline(-1, color='k', lw=0.3, ls='--'); ax.axvline(1, color='k', lw=0.3, ls='--')
        ax.legend(fontsize=7); summary_pull[out['name']] = s
    fig.suptitle("Pulls (1000-ep teacher)"); fig.tight_layout()
    fig.savefig(f"{PLOTS}/pulls_overlay.png", dpi=130); plt.close(fig)

    json.dump({"residuals": summary_res, "pulls": summary_pull},
              open(f"{SWEEP}/fit_summary.json", "w"), indent=1, default=float)

    # ---- REPORT.md ----
    teacher_p2_per_event = teacher_sum['part2_best_val_loss'] / 5000.0
    baseline_per_event = baseline_sum['part2_best_val_loss'] / 5000.0 if baseline_sum else None
    md = []
    md.append("# Distilling a 1000-epoch ViT_Max teacher into a 363-parameter physics-symbolic student")
    md.append("")
    md.append(f"Teacher trained 1000+1000 epochs ({teacher_sum.get('total_seconds','n/a')} s wall-clock). "
              f"Three symbolic+tiny-MLP students (barycenter, localreco, blend) distilled for 1000 epochs each "
              f"with the v2 ansatz patch (labels_scale pre-division). MDMM dual ascent on `lambda` enforces "
              f"KL[ teacher || student ] -> 0.")
    md.append("")
    md.append("## Setup")
    md.append(f"- Teacher: ViT_Max, 418,702 params, 1000+1000 epochs, Part 1 best val_loss = "
              f"{teacher_sum['part1_best_val_loss']:.2f}, Part 2 best val_loss = "
              f"{teacher_sum['part2_best_val_loss']:.2f} ({teacher_p2_per_event:.4f} per event)")
    if baseline_sum:
        md.append(f"- Conv2D_Max baseline (30+30 ep): 1898 params, val NLL = {baseline_per_event:.4f}/event")
    md.append(f"- Student: PhysicsAnsatz (variant-dependent) + tiny 354-param projected MLP, total ~367 params.")
    md.append(f"- v2 patch: PhysicsAnsatz pre-divides raw output by `labels_scale` (read from TFRecord metadata).")
    md.append(f"- Distillation: 1000 epochs per variant; Adam lr=1e-3 + clipnorm=1; MDMM eta=1e-3; warm-up 200 steps.")
    md.append("")
    md.append("## Results")
    md.append("")
    md.append("| model | params | per-event val NLL | best val KL |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| ViT_Max teacher (1000+1000 ep) | 418,702 | **{teacher_p2_per_event:.4f}** | - |")
    if baseline_sum:
        md.append(f"| Conv2D_Max baseline (30+30 ep) | 1,898 | {baseline_per_event:.4f} | - |")
    for v in VARIANTS:
        s = sweep_sum['results'].get(v, {})
        md.append(f"| student {v} | {s.get('student_params','?')} | "
                  f"{s.get('best_val_loss_data','?'):.4f} | "
                  f"{s.get('best_val_kl','?'):.4f} |")
    md.append("")
    md.append(f"**Ranking by best_val_kl**: {sweep_sum.get('ranking_by_best_val_kl','n/a')}")
    md.append("")
    md.append("## Pull Gaussian fits (ideal: mu=0, sigma=1)")
    md.append("")
    md.append("| output | model | mu | sigma |")
    md.append("|---|---|---:|---:|")
    for out in OUTPUTS:
        for label, s in summary_pull[out['name']].items():
            md.append(f"| {out['name']} | {label} | {s['mu']:+.3f} | {s['sigma']:.3f} |")
    md.append("")
    md.append("## Plots")
    md.append("![loss](plots/loss_vs_epoch.png)")
    md.append("![KL](plots/kl_vs_epoch.png)")
    md.append("![lambda](plots/lambda_vs_epoch.png)")
    md.append("![residuals](plots/residuals_overlay.png)")
    md.append("![pulls](plots/pulls_overlay.png)")
    md.append("")
    md.append("## Learned ansatz scalars per variant")
    md.append("")
    for v in VARIANTS:
        sp = f"{SWEEP}/{v}/ansatz_scalars.json"
        if os.path.exists(sp):
            md.append(f"### {v}")
            md.append("```json")
            md.append(open(sp).read().strip())
            md.append("```")
            md.append("")
    md.append("## Reproducibility")
    md.append("")
    md.append("| artifact | path |")
    md.append("|---|---|")
    md.append(f"| teacher run | runs/vit_max_run_1000ep/ |")
    md.append(f"| teacher Part 2 checkpoints | {teacher_sum.get('part2_checkpoints','n/a')} |")
    for v in VARIANTS:
        md.append(f"| {v} student weights | runs/distill_sweep_vit_teacher_1000ep/{v}/student_final.weights.h5 |")
    md.append(f"| teacher parquet | {find_teacher_parquet()} |")
    md.append(f"| student parquets | runs/processed_parquets/test_3src/2bit_optimized_students_1000ep/ |")
    md.append(f"| sweep summary | runs/distill_sweep_vit_teacher_1000ep/sweep_summary.json |")
    md.append(f"| fit summary | runs/distill_sweep_vit_teacher_1000ep/fit_summary.json |")

    out_path = f"{SWEEP}/REPORT.md"
    open(out_path, 'w').write('\n'.join(md))
    print(f"wrote {out_path}")


if __name__ == '__main__':
    main()
