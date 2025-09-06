from yura3208_service.srv import Yura3208Service
import numpy as np
import time
import rclpy
from rclpy.node import Node


class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(Yura3208Service, 'yura3208_service', self.yura3208_service_callback)

    def yura3208_service_callback(self, request, response):
        start_time = time.time()
        output = request.input[::-1]
        self.get_logger().info('Incoming request\ninput: %s output: %s' % (request.input, output))
        end_time = time.time()
        response.output = output
        response.runtime = end_time - start_time
        return response


def main():
    rclpy.init()

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()