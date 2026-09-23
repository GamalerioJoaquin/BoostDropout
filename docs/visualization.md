# Reconstructing analysis figures

Figures can be generated from trained models and stored history without a
running notebook kernel. Install the visualization dependencies with
`python -m pip install -e ".[torch,visualization]"`.

The `boostdropout.visualization` package provides:

- `plot_classification_history`: loss, accuracy, and error curves from the
  historical four-series history dictionary.
- `plot_coadaptation`: encoder filters with global symmetric normalization,
  including the same grid arrangement and gray floor as the thesis notebook.
- `plot_representational_sparsity`: hidden activations before regularization,
  with the fraction of exactly zero activations and other summary statistics.
- `plot_reconstruction_comparison`: the same input images reconstructed by
  multiple autoencoders.

Each function returns a Matplotlib figure and accepts an optional `save_path`.
Passing `save_path` writes a PNG while leaving display and figure closing to
the caller. This supports both headless scripts and interactive notebooks.

The existing notebooks retain their experiment selection and publication
specific figure assembly. Their shared curves, coadaptation, activation
distribution, reconstruction grid, and autoencoder training call the package.
The legacy artifacts (`model.pt`, `history.json`, `metrics.json`) remain
loadable by the autoencoder notebook.
