## Bimodal 10-D Example

This example demonstrates four training regimes for simulation-based inference (SBI) on a controlled 10-dimensional bimodal signal model. It is self-contained and runnable with the falcon CLI and the provided Hydra configuration files.

## Overview

We compare four data-curation / training strategies while keeping the simulator fixed:

- **DS-A (dynamic)**: Iteratively alternates between training and dataset curation (proposal update + data update with discard/resample).

- **Round-based (renew)**: No discard within a round; at each new round, add a fresh batch from the updated proposal.

- **Round-based (fill)**: Never discard; continuously accumulate new simulations across rounds.

- **Amortized**: Fixed dataset; no discard and no proposal refresh; train until convergence or stop.

These regimes allow controlled studies of sample efficiency, stability, and compute allocation when focusing on high-posterior-mass regions.

## Files Structure
```
bimodal/
├── README.md                          # This file
├── config_regular.yaml                # DS-A (dynamic) training
├── config_rounds_renew.yaml           # Round-based: renew per round (no within-round discard)
├── config_rounds_fill.yaml            # Round-based: fill/accumulate (never discard)
├── config_amortized.yaml              # Amortized baseline (fixed dataset)
├── JSD.py                             # Jensen–Shannon divergence utilities (compute/plot)
├── plot_os_a.py                       # Corner plot + side panels (loss/range/extreme-error)
├── plot_compare_four_loss_lifespan.py # Compare losses & sample lifespans across 4 methods
├── plot_sample_lifespan.py            # Scatter of training/validation sample IDs over time
├── data/
│   ├── x_obs_10dim.npy                # Observed datum x0 (shape: 10,)
│   ├── true_z_10dim.npy               # Reference z* (shape: 10,) for diagnostics
│   ├── shift_10dim.npy                # Per-dimension mode shifts (±3σ) used in the signal
│   └── eps_10dim.npy                  # Optional Gaussian noise draw ε (for reproducibility)
└── src/
    ├── model.py                       # Simulator components and graph wiring
    └── gen_obs.py                     # Script to regenerate x0 / z* / shift / ε

```
Note: File names under data/ reflect the standard setup for this example. If your copies differ, update the paths in the YAMLs accordingly.
## Dependencies & Installation

This example assumes you have the FALCON package available. 

```bash
# (Optional but recommended) new env
conda create -n falcon-dsbi python=3.9 -y && conda activate falcon-dsbi
# or: python -m venv .venv && source .venv/bin/activate

git clone https://github.com/cweniger/falcon.git
cd falcon
pip install .

git clone https://github.com/lvhf123/falcon-dsbi.git
pip install corner
```

## Usage
```bash
cd falcon-dsbi/examples/bimodal
python src/gen_obs.py  
falcon launch --config-name=<your_config_name> 
falcon sample posterior --config-name=<your_config_name> paths.graph=<your/graph/path>
```

- <your_config_name> refers to a YAML under examples/bimodal/ (e.g., config_regular, config_amortized, config_rounds_renew, …).

- <your/graph/path> is the output graph directory produced by step 2 (typically something like outputs/<run>/graph_dir).

## Configuration Tips

Key parameters are exposed in the YAMLs and can be overridden via Hydra:

You can append Hydra overrides after the launch command to tweak settings on the fly, e.g.:

```bash
falcon launch --config-name=config_regular.yaml \
  graph.z.estimator.num_epochs=100 \
  graph.z.estimator.lr=0.005 \
  graph.z.estimator.batch_size=512 \
  graph.z.estimator.log_ratio_threshold=-10 \
  buffer.max_training_samples=16384 \
  paths.graph=${hydra:run.dir}/graph_dir_alt \
  graph.z.ray.num_gpus=0

  # Posterior sampling count and output path override
falcon sample posterior --config-name=config_regular \
  sample.posterior.n=5000 \
  sample.posterior.path=samples_posterior_5k.joblib \
  paths.graph=<your/graph/path>

# Proposal / prior sampling similarly:
falcon sample proposal --config-name=config_regular \
  sample.proposal.n=2000 sample.proposal.path=samples_proposal_2k.joblib \
  paths.graph=<your/graph/path>

falcon sample prior --config-name=config_regular \
  sample.prior.n=2000 sample.prior.path=samples_prior_2k.joblib

  ```


