from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lyapunov_adaptive_transformer',
            namespace='astro2',
            executable='lyapunov_adaptive_transformer',
            parameters=[os.path.join(get_package_share_directory('px4_telemetry'), 'param', 'park_coordinates.yaml')],
            name='lyapunov_adaptive_transformer_node'
        )
    ])
