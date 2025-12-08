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
│   ├── augmentations   # Set of data augmentations
│   ├── datasets        # Datasets for training
│   ├── losses          # Loss functions
│   ├── models          # Model architectures for teacher and client
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
```
The first command will create a folder `venv` which contains the environment files. The second line activates the virtual environment (execute this one each time you reopen the project). The third line installs the package dependencies. To exit the virtual environment, use the `deactivate` command.

## References

[1]: Chen _et al._, 2020. A simple framework for contrastive learning of visual representations. In Proceedings of the 37th International Conference on Machine Learning (ICML'20), Vol. 119. JMLR.org, Article 149, 1597–1607. [https://proceedings.mlr.press/v119/chen20j/chen20j.pdf](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf)

[2]: Joshi _et al._, 2025. Dataset Distillation via Knowledge Distillation: Towards Efficient Self-Supervised Pre-Training of Deep Networks. In The Thirteenth International Conference on Learning Representations. [http://arxiv.org/abs/2410.02116](http://arxiv.org/abs/2410.02116)