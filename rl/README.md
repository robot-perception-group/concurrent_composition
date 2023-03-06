=================================================================

# Copyright and License

All Code in this repository - unless otherwise stated in local license or code headers is

Copyright 2020 Max Planck Institute for Intelligent Systems

Licensed under the terms of the GNU General Public Licence (GPL) v3 or higher.
See: https://www.gnu.org/licenses/gpl-3.0.en.html


# installation
1. clone this repository 
```console
git clone https://github.com/robot-perception-group/TransferLearning.git
```
2. create a conda environment
```console
cd rl
conda create --name <env_name> --file requirement.txt python=3.10
```


# Experiment1: hyperparameter tuning and ablation study 
```console
wandb sweep rl/rl_torch/sweep/sweep_dacgpi_pm3_augment.yml count==100 
```
