from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    launch_publisher_arg = DeclareLaunchArgument(
        'launch_publisher',
        default_value='true',
        description='Whether to launch the publisher node'
    )
    
    launch_subscriber_arg = DeclareLaunchArgument(
        'launch_subscriber',
        default_value='true',
        description='Whether to launch the subscriber node'
    )
    
    launch_service_arg = DeclareLaunchArgument(
        'launch_service',
        default_value='true',
        description='Whether to launch the service node'
    )
    
    launch_client_arg = DeclareLaunchArgument(
        'launch_client',
        default_value='false',
        description='Whether to launch the client node'
    )
    
    client_input_arg = DeclareLaunchArgument(
        'client_input',
        default_value='hello',
        description='Input string for the client node'
    )

    # Create nodes
    publisher_node = Node(
        package='yura3208_hw2',
        executable='publisher',
        name='minimal_publisher',
        output='screen',
        condition=IfCondition(LaunchConfiguration('launch_publisher'))
    )
    
    subscriber_node = Node(
        package='yura3208_hw2',
        executable='subscriber',
        name='minimal_subscriber',
        output='screen',
        condition=IfCondition(LaunchConfiguration('launch_subscriber'))
    )
    
    service_node = Node(
        package='yura3208_hw2',
        executable='service',
        name='minimal_service',
        output='screen',
        condition=IfCondition(LaunchConfiguration('launch_service'))
    )
    
    client_node = Node(
        package='yura3208_hw2',
        executable='client',
        name='minimal_client',
        output='screen',
        arguments=[LaunchConfiguration('client_input')],
        condition=IfCondition(LaunchConfiguration('launch_client'))
    )

    return LaunchDescription([
        # Launch arguments
        launch_publisher_arg,
        launch_subscriber_arg,
        launch_service_arg,
        launch_client_arg,
        client_input_arg,
        
        # Log info
        LogInfo(msg='Launching multiple nodes from yura3208_hw2 package'),
        
        # Nodes
        publisher_node,
        subscriber_node,
        service_node,
        client_node,
    ])
