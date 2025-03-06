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
source activate qdt
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
python experiment.py --seed 123 \
    --env hopper --dataset medium   \
    --eta 1.0 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  \
```

## Acknowledgments

This repo benefits from [DT](https://github.com/kzl/decision-transformer) and [Diffusion-QL](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL). Thanks for their wonderful works!
