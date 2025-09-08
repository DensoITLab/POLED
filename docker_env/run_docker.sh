#!/bin/bash

# Add your Docker commands here
docker rm poled
docker run \
    --name poled \
    -it \
    -v $POLED_PATH:/home/user/app \
    -v $POLED_DATA:/home/user/datasets \
    -e POLED_PATH=/home/user/app \
    -e POLED_DATA=/home/user/datasets \
    poled
