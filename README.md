=================================================================

# Copyright and License

All Code in this repository - unless otherwise stated in local license or code headers is

Copyright 2023 Max Planck Institute for Intelligent Systems

Licensed under the terms of the GNU General Public Licence (GPL) v3 or higher.
See: https://www.gnu.org/licenses/gpl-3.0.en.html



# installation
- create workspace
```console
mkdir MultitaskRL
cd MultitaskRL
```

- download isaac-gym https://developer.nvidia.com/isaac-gym
- extract isaac-gym to the workspace 
- create conda environment and install the dependencies 
```console
bash IsaacGym_Preview_4_Package/isaacgym/create_conda_env_rlgpu.sh 
conda activate rlgpu
```

- clone this repository to the workspace and install dependencies
```console
git clone https://github.com/robot-perception-group/concurrent_composition.git
pip install -r concurrent_composition/requirements.txt
```

# Run experiments
- enter the RL workspace
```
cd concurrent_composition/
```

The experiments are stored in the sweep folder
- For example, hyperparameter tuning for dac in Pointmass environment
```console
wandb sweep sweep/dac_ptr_transfer.yml --count 100
```





