#!/bin/bash
set -euo pipefail

NAME_CONTAINER="eonsim_container" # need to be changed
NAME_IMAGE="eonsim_image:test" # need to be changed

print_help() {
    echo "Usage: ./create_container.sh [build|rm|imagerm|help]"
    echo
    echo "Commands:"
    echo "  (no args)   Create/start container and attach shell"
    echo "  build       Rebuild image (removes existing image first)"
    echo "  rm          Remove container (with volumes)"
    echo "  imagerm     Remove image only (not container)"
    echo "  help        Show this help message"
}

if [ "${1:-}" == "help" ] || [ "${1:-}" == "-h" ] || [ "${1:-}" == "--help" ]; then
    print_help
    exit 0
fi

if [ "${1:-}" == "imagerm" ]; then
    echo "[CHECK] Removing IMAGE (not container): docker image inspect '$NAME_IMAGE' --format '{{.Id}} {{.RepoTags}}'"
    read -p "Delete image '$NAME_IMAGE'? (y/n): " answer
    if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
        docker rmi -f "$NAME_IMAGE"
        echo "Image '$NAME_IMAGE' removed."
    else
        echo "Image removal canceled."
    fi
    exit 0
fi

if [ "${1:-}" == "build" ]; then
    read -p "Do you want to rebuild the image '$NAME_IMAGE'? (y/n): " answer
    if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
        echo "Removing existing image '$NAME_IMAGE'..."
        docker rmi -f "$NAME_IMAGE"
        echo "Building new image '$NAME_IMAGE'..."
        docker build --build-arg GROUP_ID="$(id -g)" --build-arg USER_ID="$(id -u)" -t "$NAME_IMAGE" .
        exit 0
    else
        echo "Build canceled."
        exit 0
    fi
fi

# Check if the container already exists
if [ "$(docker ps -a -q -f name="$NAME_CONTAINER")" ]; then
    echo "Container '$NAME_CONTAINER' already exists."

    if [ "${1:-}" == "rm" ]; then
        echo "Stopping and removing the container with all associated volumes..."
        docker rm -fv "$NAME_CONTAINER"
        echo "Container '$NAME_CONTAINER' and its volumes have been stopped and removed."
    else
        # Check if the container is in stopped state
        if [ "$(docker inspect -f '{{.State.Status}}' "$NAME_CONTAINER")" == "exited" ]; then
            echo "Container '$NAME_CONTAINER' is stopped. Starting it..."
            docker start "$NAME_CONTAINER"
            echo "Attaching to the container..."
            docker exec -it "$NAME_CONTAINER" /bin/bash
        else
            echo "Container '$NAME_CONTAINER' is already running."
            docker exec -it "$NAME_CONTAINER" /bin/bash
        fi
    fi
else
    echo "Container '$NAME_CONTAINER' does not exist. Running a new container..."
    mount_dir="$(pwd)/..:/workspace"
    docker run \
        -e TERM=xterm-256color -e COLORTERM=truecolor -e FORCE_COLOR=true \
        --name "$NAME_CONTAINER" \
        --user "$(id -u):$(id -g)" \
        --volume "$mount_dir" \
        -it "$NAME_IMAGE"
fi
