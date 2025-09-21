#!/bin/bash

NAME_CONTAINER="eonsim_container" # need to be changed
NAME_IMAGE="embed_image:test" # need to be changed

if [ "$1" == "build" ]; then
    read -p "Do you want to rebuild the image '$NAME_IMAGE'? (y/n): " answer
    if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
        echo "Removing existing image '$NAME_IMAGE'..."
        docker rmi -f $NAME_IMAGE
        echo "Building new image '$NAME_IMAGE'..."
        docker build --build-arg GROUP_ID=$(id -g) --build-arg USER_ID=$(id -u) -t $NAME_IMAGE .
        exit 0
    else
        echo "Build canceled."
        exit 0
    fi
fi

# Check if the container already exists
if [ "$(docker ps -a -q -f name=$NAME_CONTAINER)" ]; then
    echo "Container '$NAME_CONTAINER' already exists."

    if [ "$1" == "rm" ]; then
        echo "Stopping and removing the container with all associated volumes..."
        docker stop $NAME_CONTAINER
        docker rm -v $NAME_CONTAINER
        echo "Container '$NAME_CONTAINER' and its volumes have been stopped and removed."
    else
        # Check if the container is in stopped state
        if [ "$(docker inspect -f '{{.State.Status}}' $NAME_CONTAINER)" == "exited" ]; then
            echo "Container '$NAME_CONTAINER' is stopped. Starting it..."
            docker start $NAME_CONTAINER
            echo "Attaching to the container..."
            docker exec -it $NAME_CONTAINER /bin/bash
        else
            echo "Container '$NAME_CONTAINER' is already running."
            docker exec -it $NAME_CONTAINER /bin/bash
        fi
    fi
else
    echo "Container '$NAME_CONTAINER' does not exist. Running a new container..."
    docker run \
        -e TERM=xterm-256color -e COLORTERM=truecolor -e FORCE_COLOR=true \
        --name $NAME_CONTAINER \
        --user $(id -u):$(id -g) \
        --volume /home/choi/Energy-Efficient-Embedding-Vector-Operation:/workspace \
        --volume /home/choi/chango:/chango \
        --volume /home/choi/dlrm_criteo_tera:/dlrm \
        -it $NAME_IMAGE
fi
