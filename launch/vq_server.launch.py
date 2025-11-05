#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("vq_server")
    params_file = os.path.join(pkg_share, "config", "vq_server.params.yaml")

    return LaunchDescription([
        Node(
            package="vq_server",
            executable="vq_server",
            name="vq_server",
            parameters=[params_file],
            output="screen",
        ),
    ])
