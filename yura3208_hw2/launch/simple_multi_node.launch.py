from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Simple launch file that launches all nodes from the yura3208_hw2 package.
    This is a basic version without launch arguments for easy testing.
    """
    
    # Publisher node
    publisher_node = Node(
        package='yura3208_hw2',
        executable='publisher',
        name='minimal_publisher',
        output='screen'
    )
    
    # Subscriber node
    subscriber_node = Node(
        package='yura3208_hw2',
        executable='subscriber',
        name='minimal_subscriber',
        output='screen'
    )
    
    # Service node
    service_node = Node(
        package='yura3208_hw2',
        executable='service',
        name='minimal_service',
        output='screen'
    )

    # Client node
    client_node = Node(
        package='yura3208_hw2',
        executable='client',
        name='minimal_client',
        output='screen'
    )

    return LaunchDescription([
        publisher_node,
        subscriber_node,
        service_node,
        client_node,
    ])
