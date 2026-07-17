# Plain-Python scripts (converted from the notebooks)

Generated with `jupyter nbconvert --to script`, then cleaned (cell markers removed)
and wired to the local paths. Run them with the `smartpix-2bit` env:

    /work/users/das214/envs/smartpix-2bit/bin/python scripts/two_bit_optimization.py

- **two_bit_optimization.py** — main training pipeline (Part 1 soft-quantize threshold
  optimization -> Part 2 2-bit training -> Part 3 evaluation). Edit the config block near
  the top (model_type, epochs, dataset dirs) before running. Best on a GPU node.
- **training_tracker.py** — single-model loss / threshold visualization (analysis; uses matplotlib).
- **comparison_plots.py** — multi-model residual / pull plots (analysis; edit the parquet_files list).

`codesign_catapult.ipynb` is intentionally NOT converted: it relies on IPython cell magics,
interactive shell commands, and the licensed Catapult/Siemens HLS toolchain, so it only runs
as a notebook.
