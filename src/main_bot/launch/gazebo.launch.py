import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg_path = get_package_share_directory('main_bot')
    world_file = os.path.join(pkg_path, 'worlds', 'race_way.world')

    xacro_file = os.path.join(pkg_path, 'description', 'robot.urdf.xacro')
    robot_description = xacro.process_file(xacro_file).toxml()

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

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
        output='screen'
    )

    # Spawn at lane 1 start position (right side of top straight, facing +X)
    spawn_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-world', 'oval_lane_world',
            '-name', 'dvt_robot',
            '-topic', 'robot_description',
            '-x', '1.0',
            '-y', '2.267',
            '-z', '0.15',
            '-Y', '0.0',
        ],
        output='screen'
    )

    bridge_config = os.path.join(pkg_path, 'config', 'gz_bridge.yaml')
    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': bridge_config, 'use_sim_time': True}],
        output='screen'
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )

    ackermann_steering_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['ackermann_steering_controller'],
        output='screen'
    )

    # Chuyển đổi Twist (/cmd_vel) → TwistStamped (/ackermann_steering_controller/reference)
    # để teleop_twist_keyboard kết nối được với ackermann_steering_controller
    twist_stamper_node = Node(
        package='twist_stamper',
        executable='twist_stamper',
        parameters=[{'use_sim_time': True}],
        remappings=[
            ('cmd_vel_in',  '/cmd_vel'),
            ('cmd_vel_out', '/ackermann_steering_controller/reference'),
        ],
        output='screen',
    )

    # Relay ~/tf_odometry → /tf dùng TransformBroadcaster (đúng QoS cho tf2)
    tf_relay_node = Node(
        package='main_bot',
        executable='odom_tf_relay.py',
        name='odom_tf_relay',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    return LaunchDescription([
        fix_pthread,
        gz_sim,
        robot_state_publisher_node,
        bridge_node,
        twist_stamper_node,
        tf_relay_node,
        TimerAction(period=8.0, actions=[spawn_node]),
        TimerAction(period=12.0, actions=[joint_state_broadcaster_spawner]),
        TimerAction(period=14.0, actions=[ackermann_steering_controller_spawner]),
    ])
