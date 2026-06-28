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

    # EKF: fuse wheel odometry + IMU → /odometry/filtered + TF odom→base_footprint
    ekf_config = os.path.join(pkg_path, 'config', 'ekf.yaml')
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_node',
        parameters=[ekf_config],
        remappings=[('odometry/filtered', '/odometry/filtered')],
        output='screen',
    )

    lane_follower_node = Node(
        package='main_bot',
        executable='lane_follower_node.py',
        name='lane_follower_node',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    overtake_vision_node = Node(
        package='main_bot',
        executable='overtake_vision_node.py',
        name='overtake_vision_node',
        parameters=[{
            'use_sim_time':  True,
            'publish_debug': True,
        }],
        output='screen',
    )

    lane_control_node = Node(
        package='main_bot',
        executable='lane_control_node',
        name='lane_control_node',
        parameters=[{
            'use_sim_time': True,
            'speed':      1.0,
            'k':          1.0,
            'max_steer':  0.52,
            'timeout':    0.5,
            # kappa_blend=0.85: odom 85% + camera 15%
            # Camera cho kappa âm (phải) khi robot ở làn ngoài trên đường cong trái
            # → kappa_use sai dấu → feed-forward lái ngược chiều.
            # 0.85 odom đảm bảo kappa_use đúng dấu ngay cả khi camera sai hoàn toàn.
            'kappa_blend': 0.85,
        }],
        output='screen',
    )

    # Overtake pipeline (Layer 2–7)
    # Tunable params:
    #   front_detect_range: khoảng cách tối đa phát hiện NPC phía trước (m)
    #   gap_time_threshold: ngưỡng thời gian kịch bản vượt (s)
    #   overtake_offset:    độ dịch ngang khi vượt (-0.534m = sang làn ngoài)
    #   overtake_hold_time: thời gian tối thiểu ở trong OVERTAKE (s)
    overtake_node = Node(
        package='main_bot',
        executable='overtake_node',
        name='overtake_node',
        parameters=[{
            'use_sim_time':        True,
            'front_detect_range':  1.0,    # khoảng cách phát hiện NPC (m); BUG FIX: was 0.1 → front_ok impossible
            'front_safe_min':      0.35,
            'front_sector_deg':   21.5,    # arc tại 1m = 2×23°×π/180×1.0 = 0.80m ≈ 1.5× lane_width(0.534m)
            'adjacent_clear_min':  0.30,
            'npc_speed':           0.25,
            'gap_time_threshold':  8.0,
            'prepare_hold_time':   0.5,
            'overtake_hold_time':  6.0,    # cần đủ thời gian để robot ổn định heading trong làn ngoài
            'return_tol':          0.04,
            'return_hold_time':    3.0,    # tăng lên 3s: camera cần thời gian tái bắt làn gốc sau RETURN
            'overtake_offset':    -0.534,
            'offset_rate_limit':   0.45,   # tốc độ đổi làn sang ngoài (m/s)
            'return_rate_limit':   0.30,   # tốc độ hồi về (m/s); giảm từ 0.60 → robot không bị tông tường
            'abort_front_dist':    0.10,
            'imu_ay_limit':        5.0,
            'same_lane_half_width': 0.15,
            'normal_speed':        0.9,
            'follow_speed':        0.25,   # = npc_speed → equilibrium ổn định, không oscillate
            'creep_speed':         0.15,
        }],
        output='screen',
    )

    # Lane geometry: road_top width=1.068m (Y=2.0→3.068), centre divider at Y=2.534
    #   Inner (left)  lane centre: lane_y=2.267  L_total≈38.24m  spacing≈9.56m
    #   Outer (right) lane centre: lane_y=2.801  L_total≈41.60m  spacing≈10.40m
    # Robot spawn: inner lane at arc≈7.0 → all NPC arcs keep ≥1.5m clearance.
    def npc(n, lane_y, arc):
        return Node(
            package='main_bot',
            executable='npc_driver_node',
            name=f'npc_{n}_driver',
            parameters=[{
                'use_sim_time': True,
                'npc_name': f'npc_{n}',
                'lane_y': lane_y,
                'initial_arc': arc,
            }],
            output='screen',
        )

    npc_nodes = [
        # Cùng làn với robot (inner lane y=2.267)
        # Robot spawn tại arc≈7.0 (X=1.0)
        # NPC 1: arc=9.5  → cách robot ~2.5m phía trước
        # NPC 2: arc=20.0 → cách NPC 1 ~10.5m (robot vượt NPC 1 xong mới gặp NPC 2)
        npc(1, 2.267, 9.5),
        npc(2, 2.267, 20.0),
    ]

    return LaunchDescription([
        fix_pthread,
        gz_sim,
        robot_state_publisher_node,
        bridge_node,
        twist_stamper_node,
        TimerAction(period=8.0,  actions=[spawn_node]),
        TimerAction(period=12.0, actions=[joint_state_broadcaster_spawner]),
        TimerAction(period=14.0, actions=[ackermann_steering_controller_spawner]),
        TimerAction(period=15.0, actions=[ekf_node]),
        TimerAction(period=16.0, actions=[lane_follower_node, lane_control_node,
                                          overtake_node, overtake_vision_node]),
        TimerAction(period=16.0, actions=npc_nodes),
    ])
