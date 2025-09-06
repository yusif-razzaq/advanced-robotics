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
import time
from geometry_msgs.msg import PointStamped
from matplotlib import pyplot as plt
import numpy as np

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.latencies = []
        self.message_count = 0
        self.max_messages = 400
        self.subscription = self.create_subscription(
            PointStamped,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        # Get current time
        current_time = self.get_clock().now()
        
        # Calculate latency (difference between publish time and receive time)
        publish_time = rclpy.time.Time.from_msg(msg.header.stamp)
        latency_ns = (current_time - publish_time).nanoseconds
        latency_ms = latency_ns / 1_000_000.0  # Convert to milliseconds
        
        # Store latency measurement
        self.latencies.append(latency_ms)
        self.message_count += 1
        
        self.get_logger().info(f'Received message {int(msg.point.x)}: latency = {latency_ms:.3f} ms')
        
        # Check if we've received all messages
        if self.message_count >= self.max_messages:
            self.get_logger().info('Received all 400 messages. Generating histogram...')
            self.generate_histogram()
            self.subscription.destroy()
            self.destroy_node()
            rclpy.shutdown()
    
    def generate_histogram(self):
        """Generate and save a histogram of latency measurements"""
        if not self.latencies:
            self.get_logger().error('No latency data to plot')
            return
            
        # Convert to numpy array for easier manipulation
        latencies_array = np.array(self.latencies)
        
        # Calculate statistics
        mean_latency = np.mean(latencies_array)
        std_latency = np.std(latencies_array)
        min_latency = np.min(latencies_array)
        max_latency = np.max(latencies_array)
        
        self.get_logger().info(f'Latency Statistics:')
        self.get_logger().info(f'  Mean: {mean_latency:.3f} ms')
        self.get_logger().info(f'  Std Dev: {std_latency:.3f} ms')
        self.get_logger().info(f'  Min: {min_latency:.3f} ms')
        self.get_logger().info(f'  Max: {max_latency:.3f} ms')
        
        # Create histogram
        plt.figure(figsize=(12, 8))
        
        # Plot histogram
        plt.hist(latencies_array, bins=50, alpha=0.7, color='blue', edgecolor='black')
        
        # Add vertical line for mean
        plt.axvline(mean_latency, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_latency:.3f} ms')
        
        # Add vertical lines for mean ± std
        plt.axvline(mean_latency + std_latency, color='orange', linestyle=':', linewidth=2, label=f'Mean + Std: {mean_latency + std_latency:.3f} ms')
        plt.axvline(mean_latency - std_latency, color='orange', linestyle=':', linewidth=2, label=f'Mean - Std: {mean_latency - std_latency:.3f} ms')
        
        # Customize plot
        plt.xlabel('Latency (milliseconds)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('ROS2 Topic Latency Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f'Statistics:\nMean: {mean_latency:.3f} ms\nStd Dev: {std_latency:.3f} ms\nMin: {min_latency:.3f} ms\nMax: {max_latency:.3f} ms\nSamples: {len(self.latencies)}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save the plot
        plt.xlim(0.1, 0.7)
        plt.tight_layout()
        
        plt.savefig('/home/yusif/ROS2/ws/topic_latency.png', dpi=300, bbox_inches='tight')
        
        self.get_logger().info('Histogram saved as topic_latency.png')


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
