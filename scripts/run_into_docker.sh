#!/bin/bash

# 1. start a docker container
# GPU version
docker run -it -d \
--name=ironman \
--network=host \
--shm-size=8G \
--runtime=nvidia \
--gpus all \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v $HOME/.Xauthority:/root/.Xauthority:rw \
-u babiking:babiking \
-v /mnt/data:/mnt/data \
babiking/ubuntu:20.04_cuda12.0_opencv5.0.0 \
/bin/bash

# 2. attach into a running docker container
docker exec -it ironman /bin/bash

# 3. commit docker container (i.e. snapshot) into image
docker commit --author babiking --message "add sudoer babiking" ironman babiking/ubuntu:20.04

# 4. push docker image to remote hub
docker push babiking/ubuntu:20.04