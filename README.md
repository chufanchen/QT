## Contents

- [Contents](#contents)
- [Overview](#overview)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Acknowledgments](#acknowledgments)


## Overview

## Requirements
```console
conda env create -f conda_env.yml
source activate erqt
```

Then add your conda environment include to CPATH (put this in your .bashrc to make it permanent):

```console
export CPATH=$CONDA_PREFIX/include
```

If you encounter the `libstdc++.so.6: version 'GLIBCXX_3.4.xx' not found` error, the following command might help:

```console
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path-to-your-conda-env>/lib
```

I have also found that `tensorboard` wants `protobuf` version to be `3.20.x`, and this helped

```console
# you might need to uninstall dm-control
pip3 install --upgrade protobuf==3.20.0
```

## Quick Start

When your environment is ready, you could run scripts in the "run.sh". For example:

``` Bash
## AUG
python main.py run_params=aug env_params=halfcheetah_medium env_params.pct_traj=0.1

## BC
python main.py agent_params=bc 

## filtered BC
python main.py agent_params=bc env_params.pct_traj=0.1

## ERQT
python main.py agent_params.behavior_ckpt_file=./save/10%_bc_stochastic-halfcheetah-medium-replay-123-250324-112957/epoch_15.pth env_params.use_aug=true env_params.pct_traj=0.1 

## QT
python main.py agent_params=qt
```

## Acknowledgments

This repo benefits from [DT](https://github.com/kzl/decision-transformer) and [Diffusion-QL](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL). Thanks for their wonderful works!
