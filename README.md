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

- setup isaac-gym 
1. download isaac-gym from https://developer.nvidia.com/isaac-gym
2. extract isaac-gym to the workspace 
3. create conda environment and install the dependencies 
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
```console
cd concurrent_composition/
```

- The experiments are stored in the sweep folder. For example, hyperparameter tuning for dac in Pointmass environment
```console
wandb sweep sweep/dac_ptr_transfer.yml --count 25
```
The experimental results will gather in Wandb. 

- To train a agent
```console
python3 run.py agent=DACGPI env=Pointer2D env.save_model=True
```

- To play a trained agent, first specify the path to the model in *play_ptr2d.py*, then run the command
```console
python3 play.py agent=DACGPI env=PointMass2D env.sim.headless=False env.num_envs=1
```




