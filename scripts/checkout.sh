#!/bin/bash

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(realpath "$THIS_DIR/..")"
SRC_DIR="$PROJECT_ROOT/src"

cd $SRC_DIR

function clone_or_pull {
    REPO_BRANCH=$1
    REPO_URL=$2
    REPO_DIR=$3

    if [ -d "$REPO_DIR/.git" ]; then
        echo "Pulling latest changes for $REPO_DIR"
        cd "$REPO_DIR"
        git pull origin "$REPO_BRANCH"
        git submodule update --init --recursive
        cd "$SRC_DIR"
    else
        echo "Cloning $REPO_URL into $REPO_DIR"
        git clone -b "$REPO_BRANCH" "$REPO_URL" "$SRC_DIR/$REPO_DIR"
    fi
}

clone_or_pull master git@github.com:ut-amrl/spot_nav.git spot_nav
clone_or_pull main https://github.com/bdaiinstitute/spot_ros2.git spot_ros2
clone_or_pull master git@github.com:ut-amrl/spot_velodyne.git spot_velodyne
clone_or_pull ros2_spot git@github.com:ut-amrl/SuperOdom.git SuperOdom
clone_or_pull main git@github.com:artzha/cotnav.git cotnav
clone_or_pull master https://github.com/Livox-SDK/livox_ros_driver2.git livox_ros_driver2
clone_or_pull master git@github.com:ut-amrl/config-reader.git config_reader
clone_or_pull master git@github.com:ut-amrl/amrl_shared_lib.git amrl_shared_lib
clone_or_pull master git@github.com:ut-amrl/amrl_maps.git amrl_maps
clone_or_pull artzha/foresight git@github.com:ut-amrl/amrl_msgs.git amrl_msgs
clone_or_pull utamrl_alphatruck git@github.com:ut-amrl/SuperOdom.git
clone_or_pull ros2_avrecorder git@github.com:ut-amrl/joystick.git joystick
clone_or_pull ros2_migration git@github.com:ut-amrl/graph_navigation.git graph_navigation
clone_or_pull arthz/foresight git@github.com:ut-amrl/webviz.git webviz
clone_or_pull master git@github.com:artzha/av_recorder.git
# clone_or_pull amrl_ros2 git@github.com:ut-amrl/elevation_mapping_cupy.git elevation_mapping_cupy

# livox_ros_driver2 uses package_ROS2.xml for ROS2, but colcon/ament requires
# the manifest to be named package.xml.
LIVOX_DIR="$SRC_DIR/livox_ros_driver2"
if [ -d "$LIVOX_DIR" ] && [ -f "$LIVOX_DIR/package_ROS2.xml" ]; then
    cp "$LIVOX_DIR/package_ROS2.xml" "$LIVOX_DIR/package.xml"
    echo "Prepared livox_ros_driver2/package.xml from package_ROS2.xml"
fi