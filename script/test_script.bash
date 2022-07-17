#!/bin/bash
xhost +si:localuser:root
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]
    then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

nvidia-docker run -d -it \
    --privileged --network=host \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
    -e DISPLAY ros-yolo3-restricted-area