# 🦅 Falcon-DSBI — Dynamic Simulation-Based Inference Examples

This repository contains runnable examples built on top of **[Falcon](https://github.com/cweniger/falcon)** for dynamic simulation-based inference (DS-A / SNPE-A variants) and accompanying plotting scripts.

> **Core dependency:** all training, sampling, and CLI functionality comes from **Falcon**.  
> Please install Falcon from source first:
> ```bash
> git clone https://github.com/cweniger/falcon.git
> cd falcon
> pip install .
> ```

---

## Requirements

Falcon installs its own runtime dependencies (PyTorch, Ray, sbi, Hydra, OmegaConf, WandB, etc.).  
To run the **bimodal** example and the provided plotting scripts in this repo, you additionally need:

- `corner` – corner plots for posterior diagnostics  
- `matplotlib` – plotting
- `scipy` – smoothing / filters used by the figures (`scipy.ndimage`)
- `joblib` – saving/loading sample arrays in `.joblib` format

Install the extra packages (on top of Falcon) with:
```bash
pip install corner matplotlib scipy joblib
```

## Quick Start

```bash
# 1) Generate a synthetic observation for the bimodal example
cd falcon-dsbi/examples/bimodal
python src/gen_obs.py

# 2) Launch training (choose your Hydra config)
falcon launch --config-name=<your_config_name>

# 3) Draw posterior samples using the trained graph
falcon sample posterior --config-name=<your_config_name> paths.graph=<your/graph/path>
```
- <your_config_name> refers to a YAML under examples/bimodal/ (e.g., config_regular, config_amortized, config_rounds_renew, …).

- <your/graph/path> is the output graph directory produced by step 2 (typically something like outputs/<run>/graph_dir).

## Project Structure

```
falcon-dsbi/
├── README.md                          # This guide
└── examples/
    ├── bimodal/
    │   ├── README.md                    # Bimodal-specific notes
    │   ├── config_regular.yaml          # DS-A (dynamic) training
    │   ├── config_rounds_renew.yaml     # Round-based: renew per round (no within-round discard)
    │   ├── config_rounds_fill.yaml      # Round-based: fill/accumulate (never discard)
    │   ├── config_amortized.yaml        # Amortized baseline (fixed dataset)
    │   ├── JSD.py                       # Jensen–Shannon divergence utilities
    │   ├── plot_os_a.py                 # Corner plot + side panels (loss/range/error)
    │   ├── plot_compare_four_loss_lifespan.py  # Compare losses & sample lifespans (4 methods)
    │   ├── plot_sample_lifespan.py      # Scatter of training/validation sample IDs over time
    │   ├── data/
    │   │   ├── x_obs_10dim.npy          # Observed datum x0 (shape: 10,)
    │   │   ├── true_z_10dim.npy         # Reference z* (shape: 10,) for diagnostics
    │   │   ├── shift_10dim.npy          # Per-dimension mode shifts (±3σ) used in the signal
    │   │   └── eps_10dim.npy            # Optional Gaussian noise draw ε (for reproducibility)
    │   └── src/
    │       ├── model.py                 # Simulator components and graph wiring
    │       └── gen_obs.py               # Script to regenerate x0 / z* / shift / ε
    └── .../                             # More examples coming soon (placeholder)
```

## What each file/folder does

- `examples/bimodal/*.yaml`:  
  Hydra configuration files consumed by the Falcon CLI. They define the graph (prior/simulator/estimator), buffer settings, logging, and Ray resources.  

- `examples/bimodal/src/gen_obs.py`:  
  Generates the observation vector `x_obs` for the bimodal example and saves it to `data/`. Run this before launching training.

- `examples/bimodal/src/model.py`:  
  Defines the forward model used by the example graph node(s). Falcon calls this during simulation to generate training pairs `(z, x)`.  
  A small PyTorch module producing summary features `s` from raw observations. It is referenced by the estimator config.

- `examples/bimodal/data/`:  
  Location for observation arrays and any auxiliary `.npy` inputs the example needs. `gen_obs.py` writes here.

- `examples/bimodal/JSD.py`  
  Computes per-dimension Jensen–Shannon divergence between two sample sets and can plot density-aligned histograms.  
  **Requires:** `matplotlib`, `scipy`, `numpy`.

- `examples/bimodal/plot_os_a.py`  
  Produces a corner plot overlaying learned posterior versus reference and adds three side panels (train/validation loss, parameter min/max evolution, extreme-error log plot). Optionally reports average JS divergence via `JSD.py`.  
  **Inputs:** posterior samples (e.g., `samples_posterior.joblib`), model histories under `outputs/.../graph_dir/z/*.pth`, and data under `examples/bimodal/data/`.  
  **Requires:** `corner`, `matplotlib`, `scipy`, `numpy`, `joblib`, `torch`.

- `examples/bimodal/plot_compare_four_loss_lifespan.py`  
  2×3 dashboard comparing four training strategies: left column shows train/validation loss and total samples; right 2×2 shows simulation-ID vs time scatter for each method.  
  **Requires:** `matplotlib`, `numpy`, `torch`.

- `examples/bimodal/plot_sample_lifespan.py`  
  Single-figure scatter of training/validation simulation IDs over time using `train_id_history.pth` and `validation_id_history.pth`.  
  **Requires:** `matplotlib`, `numpy`, `torch`.


## Notes on Configuration

- **Choosing a network:**  
  In your YAML, `graph.z.estimator.net_type` may be one of `nsf`, `maf`, `made`, or Zuko variants like `zuko_nsf`, `zuko_maf`, etc.  
  - If you select a Zuko architecture, ensure `zuko` is installed.

- **GPU/CPU:**  
  GPU use is controlled by Falcon’s Ray config (e.g., `graph.z.ray.num_gpus: 1`). CPU-only works as well.

- **Outputs:**  
  Falcon writes runs into `outputs/<timestamp>/` with `graph_dir/` (models), and posterior samples (e.g., `samples_posterior.joblib`).


## Citation

If you use **Falcon-DSBI** or **Falcon** in your research, please cite both repositories:

```bibtex
@software{falcon_dsbi_2025,
  title        = {Falcon-DSBI: Dynamic Simulation-Based Inference Examples},
  author       = {Lyu and Contributors},
  year         = {2025},
  url          = {https://github.com/lvhf123/falcon-dsbi},
  note         = {Examples and plotting utilities built on top of Falcon}
}

@software{falcon2024,
  title        = {Falcon: Federated Adaptive Learning of CONditional distributions},
  author       = {Weniger, and Contributors},
  year         = {2024},
  url          = {https://github.com/cweniger/falcon}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.  
If you plan a larger change, consider opening an issue first to discuss the design.

## Support
For questions and support, please open an issue on GitHub:  
- Falcon-DSBI: https://github.com/lvhf123/falcon-dsbi/issues  
- Falcon (core framework): https://github.com/cweniger/falcon/issues

You can also mention maintainers in your issue for quicker feedback.
