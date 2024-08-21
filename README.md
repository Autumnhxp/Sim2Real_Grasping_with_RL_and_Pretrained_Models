# Training on GPU

## Instructions
1. SSH to remote server

2. Set DISPLAY variable and endpoint

Run `xdpyinfo` to check if there is a display running

If not, initialize display with `Xvfb :1 -screen 0 1024x768x24 > /dev/null 2>&1 &`

Set DISPLAY variable `export DISPLAY=:1` to match display server

3. Build and run docker

`docker build . -t vtprl_image -f docker/update.Dockerfile`


This uses the  `--gpus all` flag:

`docker run --rm -it --name vtprl_container --gpus all -e DISPLAY -v $(pwd):/home/vtprl:rw -v $(pwd)/external/stable-baselines3:/home/repos/stable-baselines3:ro --privileged --net="host" vtprl_image:latest bash`

This does **not use gpus** flag:

`docker run --rm -it --name vtprl_container -e DISPLAY -v $(pwd):/home/vtprl:rw -v $(pwd)/external/stable-baselines3:/home/repos/stable-baselines3:ro --privileged --net="host" vtprl_image:latest bash`

## Setup tensorboard

Attach shell to container:
`sudo docker exec -it vtprl_container /bin/bash`

Start tensorboard:
`tensorboard --logdir /home/vtprl/agent/tensorboard_logging/ --bind_all`

Link localhost:
ssh -L 6006:localhost:6006 [gpu_address]

## Further commands

Check if Unity's gRPC server is running: 

`netstat -tuln | grep 9092`

Check display variable

`echo $DISPLAY`

Run simulator

`./ManipulatorEnvironment.x86_64`


# Setup Docker
To run the `update.Dockerfile`, from the `/vtprl` directory:

`docker build . -t vtprl_image -f docker/update.Dockerfile`

`docker run  --rm -it             --name vtprl_container             -e DISPLAY             -v $(pwd):/home/vtprl:rw             -v $(pwd)/external/stable-baselines3:/home/repos/stable-baselines3:ro             --privileged             --net="host"             -p 5678:5678             vtprl_image:latest             bash`
