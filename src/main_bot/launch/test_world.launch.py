import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg_path = get_package_share_directory('main_bot')
    world_file = os.path.join(pkg_path, 'worlds', 'race_way.world')

    # Fix snap-pthread conflict when running inside VSCode snap environment
    fix_pthread = SetEnvironmentVariable(
        'LD_PRELOAD',
        '/usr/lib/x86_64-linux-gnu/libpthread.so.0'
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])
        ]),
        launch_arguments={
            'gz_args': f'-r {world_file}',
            'on_exit_shutdown': 'True',
        }.items()
    )

    return LaunchDescription([
        fix_pthread,
        gz_sim,
    ])
