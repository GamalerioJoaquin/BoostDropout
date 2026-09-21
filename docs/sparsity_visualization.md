# Short procedure for reproducing the sparsity figure

1. Train a single-hidden-layer autoencoder on MNIST with architecture `784 -> 256 -> 784` and **ReLU** activations in the latent layer. For the Dropout variant, use the probability defined for the corresponding comparison.
2. Take **one random minibatch from the test set**.
3. For that minibatch, compute the hidden activations `h` with shape `[batch_size, 256]`.
4. Plot two histograms:
   - **Left**: the distribution of the **mean activation per hidden unit** across the minibatch, that is, `mean(h, dim=0)`.
   - **Right**: the distribution of **all activations** in the same minibatch, that is, `h.flatten()`.
5. For the Dropout variant, do not apply additional weight scaling when building the figure. The comparison should reflect the activations obtained under the evaluation procedure.

This is the procedure implemented by `plot_representational_sparsity_from_model_path(...)` in `autoencoder_protocol.ipynb`.
