# CPDD
Contrastive Prototype Dataset Distillation

## Introduction

This project presents an implementation of the CPDD work of Elouan Colybes, Dipl.-Ing., M.Sc., and Shirin Salehi, Dr.-Ing., started in November 2025. Based on SimCLR [[1]](#references), and inspired from the work on MKDT [[2]](#references).

## Walkthrough
You can find here a description of the structure of the project.

```python
├── examples            # Examples of project components
│   ├── dinov3_task.py
│   └── image.jpg
├── src                 # Project core
│   ├── datasets        # Datasets for training
│   ├── distillation    # Distillation framework
│   ├── losses          # Loss functions
│   ├── models          # Model architectures for teacher and client
│   ├── simclr          # SimCLR framework
│   ├── utils           # Useful functions
│   └── main.py
├── README.md
└── requirements.txt
```

## First steps
_Recommanded:_ use a Python virtual environment. To set up the environment, use the next commands:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 # See next section
```
The first command will create a folder `venv` which contains the environment files. The second line activates the virtual environment (execute this one each time you reopen the project). The third line installs the package dependencies. To exit the virtual environment, use the `deactivate` command.

## CUDA (Windows)

The project is designed to use CUDA for GPU-accelerated processing. See following the requirements.

### Requirements
- NVIDIA GPU with recent driver (`nvidia-smi` works)
- **Python >= 3.11.x (64-bit)**
- PyTorch: install via the tutorial: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/). Use a command adapted to your OS and your CUDA driver version.


## References

[1]: Chen _et al._, 2020. A simple framework for contrastive learning of visual representations. In Proceedings of the 37th International Conference on Machine Learning (ICML'20), Vol. 119. JMLR.org, Article 149, 1597–1607. [https://proceedings.mlr.press/v119/chen20j/chen20j.pdf](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf)

[2]: Joshi _et al._, 2025. Dataset Distillation via Knowledge Distillation: Towards Efficient Self-Supervised Pre-Training of Deep Networks. In The Thirteenth International Conference on Learning Representations. [http://arxiv.org/abs/2410.02116](http://arxiv.org/abs/2410.02116)

## Acknowledgements

Our code in the folders `src/distillation` and `src/simclr` is widely inspired from other repositories, listed in the corresponding `README` files. We thank them for their excellent work.
