# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import time

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(PointStamped, 'topic', 10)
        self.timer = self.create_timer(0.05, self.timer_callback)  # Faster publishing for latency measurement
        self.message_count = 0
        self.max_messages = 400

    def timer_callback(self):
        if self.message_count >= self.max_messages:
            self.get_logger().info('Published all 400 messages. Stopping publisher.')
            self.timer.cancel()
            return
            
        msg = PointStamped()
        
        # Set header with current timestamp
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'latency_test'
        
        # Use point coordinates to store message data
        msg.point.x = float(self.message_count)  # Message ID
        msg.point.y = 0.0
        msg.point.z = 0.0
        
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published message {self.message_count} at {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}')
        self.message_count += 1

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
