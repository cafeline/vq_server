#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("vq_server")
    params_file = os.path.join(pkg_share, "config", "vq_server.params.yaml")
    default_map = os.path.join(pkg_share, "maps", "tsudanuma_voxelsize_05_compressed_map.h5")

    map_file_arg = DeclareLaunchArgument(
        "map_file",
        default_value=default_map,
    )

    node = Node(
        package="vq_server",
        executable="vq_server",
        name="vq_server",
        parameters=[params_file, {"map_file": LaunchConfiguration("map_file")}],
        output="screen",
    )

    return LaunchDescription([
        map_file_arg,
        node,
    ])
