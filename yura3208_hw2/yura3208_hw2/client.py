import sys
from matplotlib import pyplot as plt
from yura3208_service.srv import Yura3208Service
import rclpy
from rclpy.node import Node
import time
import numpy as np


class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(Yura3208Service, 'yura3208_service')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Yura3208Service.Request()

    def send_request(self, input):
        self.req.input = input
        return self.cli.call_async(self.req)


def main():
    rclpy.init()

    minimal_client = MinimalClientAsync()
    latency = []
    for i in range(400):
        start_time = time.time()
        future = minimal_client.send_request(sys.argv[1])
        rclpy.spin_until_future_complete(minimal_client, future)
        response = future.result()
        end_time = time.time()
        latency.append((end_time - start_time - response.runtime)*1000)
        minimal_client.get_logger().info(
            'Result of yura3208_service: for %s = %s' %
            (sys.argv[1], response.output))
        
    plt.figure(figsize=(12, 8))
    mean_latency = np.mean(latency)
    std_latency = np.std(latency)
    min_latency = np.min(latency)
    max_latency = np.max(latency)
    # Plot histogram
    plt.hist(latency, bins=50, alpha=0.7, color='blue', edgecolor='black')
    
    # Add vertical line for mean
    plt.axvline(mean_latency, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_latency:.3f} ms')
    
    # Add vertical lines for mean ± std
    plt.axvline(mean_latency + std_latency, color='orange', linestyle=':', linewidth=2, label=f'Mean + Std: {mean_latency + std_latency:.3f} ms')
    plt.axvline(mean_latency - std_latency, color='orange', linestyle=':', linewidth=2, label=f'Mean - Std: {mean_latency - std_latency:.3f} ms')
    
    # Customize plot
    plt.xlabel('Latency (milliseconds)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('ROS2 Service Latency Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Statistics:\nMean: {mean_latency:.3f} ms\nStd Dev: {std_latency:.3f} ms\nMin: {min_latency:.3f} ms\nMax: {max_latency:.3f} ms\nSamples: {len(latency)}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save the plot
    plt.xlim(0.1, 0.7)
    plt.tight_layout()
    plt.savefig('/home/yusif/ROS2/ws/service_latency.png', dpi=300, bbox_inches='tight')
    minimal_client.get_logger().info('Histogram saved as service_latency.png')

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()