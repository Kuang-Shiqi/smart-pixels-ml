## Getting Started (for new collaborators)

To reproduce the current symbolic MoE pipeline, run these two notebooks in order:

1. **Train the model:** [`train_symbolic_nexp_ablation.ipynb`](https://github.com/Kuang-Shiqi/smart-pixels-ml/blob/symbolic/train_symbolic_nexp_ablation.ipynb)
   - Trains the symbolic MoE student model (distilled from the ViT_Max teacher).
   - Update dataset paths at the top of the notebook to point to your local/cluster copy before running.

2. **Evaluate and plot:** [`compare_baselines_vs_symbolic.ipynb`](https://github.com/Kuang-Shiqi/smart-pixels-ml/blob/symbolic/compare_baselines_vs_symbolic.ipynb)
   - Loads the trained checkpoint from step 1 and compares against paper baselines (Conv2D, Conv1D, MLP).
   - Produces residual/pull plots and summary stats.

Run both **top to bottom** (Restart Kernel and Run All) rather than executing cells out of order.

Questions? Ping Shiqi.
