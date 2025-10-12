# FALCON — Federated Adaptive Learning of CONditional distributions

> **Code base:** This example and CLI usage are built on top of the open-source **FALCON** framework.  
> Upstream repository: <https://github.com/cweniger/falcon.git>

---


## Install development version
```bash
git clone https://github.com/lvhf123/falcon-dsbi.git
cd falcon-dsbi
pip install -e .
```

## Usage

* falcon launch [hydra_options...]
* falcon sample prior [hydra_options...]
* falcon sample proposal [hydra_options...]
* falcon sample posterior [hydra_options...]

- For step-by-step instructions per example, see the README files inside the corresponding subfolders under examples/ (e.g., examples/bimodal/README.md).
- For a conceptual overview and design principles, see CLAUDE.md.
