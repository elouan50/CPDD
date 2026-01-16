# Distillation

Please note that the entirety of this folder is adapted from [this repository](https://github.com/Li-Hongcheng/DSDM) by Hongcheng Li.
Following in this file is their README:

## Diversified Semantic Distribution Matching for Dataset Distillation (MM 2024)

### Run

1. prepare pre-trained models

```
python pre_train_model.py --reproduce  -d [dataset]
```
2. distilling process 

```
python DSDM.py  --reproduce -d [dataset] -f 2 --ipc [instance/class]
```
### Acknowledgements

Our code in this project is built upon the work of [IDC](https://github.com/snu-mllab/efficient-dataset-condensation). We thank them for their excellent work.

