#!/bin/bash

pip3 install hickle \
    hydra-core \
    wandb \
    joblib \
    pandas
pip3 install efficientnet_pytorch \
    "depth-anything-v2 @ git+https://github.com/debOliveira/depth-anything-V2.git@7885bbc0647bc64d55ff5803561ea2c7dea1af72"