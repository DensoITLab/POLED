#!/bin/bash

# Build docker image
docker build --no-cache -t poled:latest -f docker_env/Dockerfile .