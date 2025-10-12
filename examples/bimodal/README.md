# FALCON - Federated Adaptive Learning of CONditional distributions

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
├── README.md                    # This file
├── config_regular.yaml          # DS-A (dynamic) training
├── config_rounds_renew.yaml     # Round-based: renew per round (no within-round discard)
├── config_rounds_fill.yaml      # Round-based: fill/accumulate (never discard)
├── config_amortized.yaml        # Amortized baseline (fixed dataset)
├── data/
│   ├── x_obs_10dim.npy          # Observed datum x0 (shape: 10,)
│   ├── true_z_10dim.npy         # Reference z* (shape: 10,) for diagnostics
│   ├── shift_10dim.npy          # Per-dimension mode shifts (±3σ) used in the signal
│   └── eps_10dim.npy            # Example Gaussian noise draw ε (optional, for reproducibility)
└── src/
    ├── model.py                 # Simulator components and graph wiring
    └── gen_obs.py               # (Optional) script to regenerate x0 / z* / shift / ε
```
Note: File names under data/ reflect the standard setup for this example. If your copies differ, update the paths in the YAMLs accordingly.
## Dependencies & Installation

This example assumes you have the FALCON package available. 


```bash
cd falcon-dsbi
pip install -e .
```

## Usage
```bash
cd falcon-dsbi/examples/bimodal
python src/gen_obs.py  
falcon launch --config-name=<your_config_name> 
falcon sample posterior --config-name=<your_config_name> paths.graph=<your/graph/path>

```

## Configuration Tips

Key parameters are exposed in the YAMLs and can be overridden via Hydra:

Nomber of epochs: num_epochs

Learning rate: lr

Discard threshold: log_ratio_threshold

Network type, Batch size, early stoppatience, etc.

```bash
falcon launch --config-name=config_regular.yaml   graph.z.estimator.num_epochs=100   graph.z.estimator.lr=0.005   graph.z.estimator.log_ratio_threshold=-10   graph.z.estimator.batch_size=512
```

