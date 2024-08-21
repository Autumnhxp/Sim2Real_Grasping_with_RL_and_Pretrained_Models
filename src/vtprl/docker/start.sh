#!/bin/bash
nohup tensorboard --logdir /home/vtprl/agent/tensorboard_logging --bind_all > /dev/null 2>&1 &
