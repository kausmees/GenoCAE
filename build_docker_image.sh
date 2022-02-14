#!/bin/bash
#
# Build the Docker container.
#
# Usage:
#
# ./build_docker_container.sh
#
#

sudo docker build -t gcae/genocae:build -f docker/build.dockerfile .

if [[ $HOSTNAME == "N141CU" ]]; then
  notify-send "Done creating Dockerfile" "Done creating Dockerfile"
fi



