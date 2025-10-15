# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import jensenshannon

from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
import torch
from torch.distributions import MultivariateNormal, kl_divergence


def compute_js_divergence(samples1, samples2, param_dim, nbins=50):
    """
    Compute Jensen–Shannon divergence dimension-wise and plot histograms.

    Args:
        samples1 (np.ndarray): Shape (N, D). Samples from distribution 1.
        samples2 (np.ndarray): Shape (N, D). Samples from distribution 2.
        param_dim (int): Number of dimensions (D) to evaluate.
        nbins (int): Number of histogram bins per dimension.

    Returns:
        float: The mean JS divergence across `param_dim` dimensions.
    """
    jsd_sum = 0.0

    plt.figure(figsize=(6 * param_dim, 6))  # one subplot per dimension

    # Iterate over each dimension
    for dim in range(param_dim):
        ax = plt.subplot(1, param_dim, dim + 1)
        plt.xlabel(f"Parameter {dim+1}")

        # Histograms for both distributions
        h1, bins1 = np.histogram(samples1[:, dim], density=True, bins=nbins)
        h2, bins2 = np.histogram(samples2[:, dim], density=True, bins=nbins)

        # Bin centers
        bin_centres1 = 0.5 * (bins1[1:] + bins1[:-1])
        bin_centres2 = 0.5 * (bins2[1:] + bins2[:-1])

        # Plot histograms
        plt.bar(
            bin_centres1,
            h1,
            width=bin_centres1[1] - bin_centres1[0],
            alpha=0.3,
            label="Distribution 1",
        )
        plt.bar(
            bin_centres2,
            h2,
            width=bin_centres2[1] - bin_centres2[0],
            alpha=0.3,
            label="Distribution 2",
        )

        # Unify x-range using overlap
        min_val = max(bin_centres1[0], bin_centres2[0])
        max_val = min(bin_centres1[-1], bin_centres2[-1])

        # Smooth densities via cubic interpolation
        fit1 = interp1d(bin_centres1, h1, kind="cubic", fill_value="extrapolate")
        fit2 = interp1d(bin_centres2, h2, kind="cubic", fill_value="extrapolate")

        # Common grid
        x_grid = np.linspace(min_val, max_val, 1000)

        # Evaluate densities
        p_density = fit1(x_grid)
        q_density = fit2(x_grid)

        # Clamp to avoid numerical issues
        p_density = np.maximum(p_density, 1e-6)
        q_density = np.maximum(q_density, 1e-6)

        # Normalize to probability vectors
        p_density /= np.sum(p_density)
        q_density /= np.sum(q_density)

        # JS divergence (square of Jensen-Shannon distance from SciPy)
        jsd = jensenshannon(p_density, q_density) ** 2
        jsd_sum += jsd

        # Density lines
        plt.plot(x_grid, p_density, label="Fit 1")
        plt.plot(x_grid, q_density, label="Fit 2")

        plt.title(f"JS Divergence = {jsd:.5f}")
        plt.legend()

    avg_jsd = jsd_sum / float(param_dim)
    plt.suptitle(f"Average JS Divergence = {avg_jsd:.5f}")
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs('figures', exist_ok=True)

    # Save figure
    plt.savefig(f'figures/JS_Divergence_dim{param_dim}.png', bbox_inches='tight')
    plt.show()

    return avg_jsd


# def kl_divergence_knn(samples_p, samples_q, k=5):
#     """
#     Estimate KL divergence D_KL(P || Q) using k-nearest neighbors.
#
#     Args:
#         samples_p (np.ndarray): Samples from P, shape (N, D).
#         samples_q (np.ndarray): Samples from Q, shape (N, D).
#         k (int): Number of nearest neighbors.
#
#     Returns:
#         float: Estimated KL divergence.
#     """
#     N = samples_p.shape[0]
#     d = samples_p.shape[1]
#
#     tree_p = cKDTree(samples_p)
#     tree_q = cKDTree(samples_q)
#
#     # k+1 for P because the nearest neighbor includes the point itself
#     r_p, _ = tree_p.query(samples_p, k + 1)
#     r_p = r_p[:, -1]
#
#     r_q, _ = tree_q.query(samples_p, k)
#     r_q = r_q[:, -1]
#
#     kl = np.log(r_q / r_p).sum() * d / N + np.log(N / (N - 1))
#     return kl
#
#
# def kl_divergence_kde(samples_p, samples_q):
#     """
#     Estimate KL divergence using kernel density estimation (KDE).
#
#     Args:
#         samples_p (np.ndarray): Samples from P, shape (N, D).
#         samples_q (np.ndarray): Samples from Q, shape (N, D).
#
#     Returns:
#         float: Estimated KL divergence D_KL(P || Q).
#     """
#     kde_p = gaussian_kde(samples_p.T)
#     kde_q = gaussian_kde(samples_q.T)
#
#     p_density = kde_p(samples_p.T)
#     q_density = kde_q(samples_p.T)
#
#     kl = np.mean(np.log(p_density / q_density))
#     return kl
#
#
# # Alternative (parametric Gaussian approximation):
# # network_mean = network_samples.mean(dim=0)
# # network_cov = torch.cov(network_samples.T)
# # network_posterior = MultivariateNormal(network_mean, network_cov)
# # kl_div = kl_divergence(network_posterior, true_posterior).sum().item()
# # print(f"KL Divergence (Network || Theory): {kl_div:.4f}")
