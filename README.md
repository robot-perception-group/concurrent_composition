=================================================================

# Copyright and License

All Code in this repository - unless otherwise stated in local license or code headers is

Copyright 2023 Max Planck Institute for Intelligent Systems

Licensed under the terms of the GNU General Public Licence (GPL) v3 or higher.
See: https://www.gnu.org/licenses/gpl-3.0.en.html



# install the environment
1. install raisim follow the instruction: https://raisim.com/sections/Installation.html
2. clone this repository 
3. specify the workspace in the console
```console
export RAISIM_WORKSPACE="$PWD/raisimLib"
export WORKSPACE="$PWD/concurrent_composition"
```

4. copy the environment to the Raisim workspace
```console
cp -r $WORKSPACE/raisim_multitask_env/ $RAISIM_WORKSPACE/raisimLib/raisimGymTorch/raisimGymTorch/env/envs/
```

5. build the environment 
```console
python3 $RAISIM_WORKSPACE/raisimLib/raisimGymTorch setup.py develop
```
Note that a Raisim License is required.


# install the agent
6. create a conda environment and install dependencies. 
```console
cd rl
conda create --name <env_name> --file requirement.txt python=3.10
```


# Run experiments
- hyperparameter tuning and ablation study 
```console
wandb sweep rl/rl_torch/sweep/sweep_dacgpi_pm3_augment.yml 
```
add an argument, --count 100, when activating the wandb agent 


