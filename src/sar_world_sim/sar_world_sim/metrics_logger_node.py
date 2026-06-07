import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import csv
import os
from datetime import datetime


class MetricsLoggerNode(Node):
    def __init__(self):
        super().__init__('metrics_logger_node')

        self.declare_parameter('run_name', 'test_run')
        self.declare_parameter('output_dir', 'results')

        self.run_name = self.get_parameter('run_name').get_parameter_value().string_value
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value

        os.makedirs(self.output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.file_path = os.path.join(
            self.output_dir,
            f'{self.run_name}_{timestamp}_events.csv'
        )

        self.csv_file = open(self.file_path, mode='w', newline='')
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                'wall_time',
                'event_time',
                'event',
                'robot',
                'poi',
                'epoch',
                'value',
                'raw'
            ]
        )
        self.writer.writeheader()
        self.csv_file.flush()

        self.sub = self.create_subscription(
            String,
            '/mission_event',
            self.event_callback,
            10
        )

        self.get_logger().info(f"Metrics logger ready. Writing to: {self.file_path}")

    def event_callback(self, msg: String):
        raw = msg.data.strip()
        parsed = self.parse_event(raw)

        row = {
            'wall_time': self.now_sec(),
            'event_time': parsed.get('time', ''),
            'event': parsed.get('event', ''),
            'robot': parsed.get('robot', ''),
            'poi': parsed.get('poi', ''),
            'epoch': parsed.get('epoch', ''),
            'value': parsed.get('value', ''),
            'raw': raw
        }

        self.writer.writerow(row)
        self.csv_file.flush()

        self.get_logger().info(
            f"logged event={row['event']} robot={row['robot']} poi={row['poi']}"
        )

    def parse_event(self, raw: str):
        result = {}

        parts = raw.split(',')

        for part in parts:
            if '=' not in part:
                continue

            key, value = part.split('=', 1)
            result[key.strip()] = value.strip()

        return result

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def destroy_node(self):
        try:
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MetricsLoggerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
