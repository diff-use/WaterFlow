# WaterFlow
Predicting water moelcule placements on protein surfaces using flow matching conditioned on learned protein structure embeddings.

This repo contains all the code required to process a dataset, load in features from a pre-trained model for protein structure embeddings, and run inference (and training) of the flow matching classes. 

# Environment Setup

We use `uv` for our environment and package management, with python 3.12.

You can install the environment by running `uv sync` and running the scripts with `uv run python <script>`. Or if you want to install a fresh virtual environment from scratch, follow the steps below.

Installing the environment:

```
uv venv water --python 3.12
source water/bin/activate

uv pip install torch==2.8.0
uv pip install torch_geometric
uv pip install torch_cluster torch_scatter pyg_lib -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
uv pip install biotite wandb Bio networkx e3nn pytest pytest-cov
```

If you have trouble installing torch_cluster or scatter, I would suggest changing the cuda version in the wheel.

# TODO

- [ ] Encoder.py integrating original SLAE encoder
- [ ] Integrating ESM3 and ESM-C embeddigs for the scalar features in the encoder, equivariant features can be init to zeros and learnt
- [ ] Integration tests for the whole pipeline from data processing and caching to forward/backward passes, loss computation, and eval
- [ ] inference.py
- [ ] torch lightning or DDP for multi gpu training
- [ ] write pdb method in the dataset class 
