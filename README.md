=================================================================

# Copyright and License

All Code in this repository - unless otherwise stated in local license or code headers is

Copyright 2023 Max Planck Institute for Intelligent Systems

Licensed under the terms of the GNU General Public Licence (GPL) v3 or higher.
See: https://www.gnu.org/licenses/gpl-3.0.en.html



# installation
- clone this repository 
- specify the workspace in the console
```console
export RAISIM_WORKSPACE="$PWD/raisimLib"
export WORKSPACE="$PWD/concurrent_composition"
```

- create conda environment and install the dependencies 
```console
cd $WORKSPACE/rl
conda create --name concurrent --file requirement.txt python=3.10
```

- clone and install raisim follow the instruction: https://raisim.com/sections/Installation.html


- copy the environment to the Raisim workspace
```console
cp -r $WORKSPACE/raisim_multitask_env/* $RAISIM_WORKSPACE/raisimGymTorch/raisimGymTorch/env/envs/
```

- build the environment 
```console
cd $RAISIM_WORKSPACE/raisimGymTorch
python setup.py develop
```
Note that a Raisim License is required to continue. Follow the instruction in the RaisimLib to acquire a license.


# Run experiments
- hyperparameter tuning and ablation study 
```console
wandb sweep rl/rl_torch/sweep/sweep_dacgpi_pm3_augment.yml 
```
add an argument, --count 100, when activating the wandb agent 



