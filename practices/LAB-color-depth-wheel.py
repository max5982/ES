import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np
import json

def make_twist(lx=0.0, ly=0.0, lz=0.0, ax=0.0, ay=0.0, az=0.0):
    """Helper to generate a Twist message."""
    twist = Twist()
    twist.linear.x = lx
    twist.linear.y = ly
    twist.linear.z = lz
    twist.angular.x = ax
    twist.angular.y = ay
    twist.angular.z = az
    return twist

def load_lab_bounds(json_name="LAB-cal.json"):
    """Load LAB color bounds from JSON file, or use defaults if loading fails."""
    # Default values
    l_min, a_min, b_min = 0, 0, 0
    l_max, a_max, b_max = 255, 255, 255
    try:
        with open(json_name) as f:
            cfg = json.load(f)
        lower = np.array([cfg["l_min"], cfg["a_min"], cfg["b_min"]])
        upper = np.array([cfg["l_max"], cfg["a_max"], cfg["b_max"]])
        print(f"{json_name} loaded successfully.")
        return lower, upper
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Failed to load {json_name}. Using default values. (Reason: {e})")
        lower = np.array([l_min, a_min, b_min])
        upper = np.array([l_max, a_max, b_max])
        return lower, upper

class DepthRgbFilter(Node):
    def __init__(self):
        super().__init__('depth_rgb_filter')
        self.bridge = CvBridge()
        # Load LAB color threshold values
        self.lower, self.upper = load_lab_bounds()
        # Publisher for Mecanum drive commands
        self.pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)

        # Subscribers for synchronized RGB and Depth images
        rgb_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/rgb0/image')
        depth_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/depth0/image_raw')
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.05)
        ts.registerCallback(self.callback)

    def send_control(self, kind, speed=0.2):
        """Send Twist command according to the kind of movement."""
        if kind == "stop":
            self.pub.publish(make_twist())
        elif kind == "forward":
            self.pub.publish(make_twist(lx=speed))
        elif kind == "left":
            self.pub.publish(make_twist(lx=0.2, az=speed))
        elif kind == "right":
            self.pub.publish(make_twist(lx=0.2, az=-speed))

    def detect_object(self, mask, width):
        """
        Detect the largest object in the mask.
        Returns center, bounding box, and position index.
        pos: 0=center, 1=left, 2=right
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, 0
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        if w > 10 or h > 10: # Ignore small objects less than 10x10
            third = width // 3
            cx, cy = x + w // 2, y + h // 2 # center position of the object
            # pos: 1=left, 2=right, 0=center
            if cx < third: pos = 1       # Left
            elif cx < 2 * third: pos = 0 # Center
            else: pos = 2                # Right
            return (cx, cy), (x, y, w, h), pos
        return None, None, 0

    def callback(self, rgb_msg, depth_msg):
        """Main callback for synchronized RGB and Depth images."""
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        height, width = rgb.shape[:2]   # Get image height, width
        # LAB color filter
        lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
        mask = cv2.inRange(lab, self.lower, self.upper)
        filtered = cv2.bitwise_and(rgb, rgb, mask=mask)

        # Object detection and depth extraction
        center, bbox, pos = self.detect_object(mask, width)

        depth_val = 0
        if center:
            cx, cy = center
            cv2.circle(filtered, (cx, cy), 6, (0, 0, 200), 3)
            depth_val = int(depth[cy, cx])
            cv2.putText(filtered, f"{depth_val} mm", (cx - 15, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 250, 0), 2)
        # Decide movement based on object position and depth
        if depth_val > 500: # 500 mm
            if pos == 0:
                self.send_control("forward", 0.3)
            elif pos == 1:
                self.send_control("left", 0.3)
            elif pos == 2:
                self.send_control("right", 0.3)
        else:
            self.send_control("stop")
        # Show result
        cv2.imshow('RGB Preview', filtered)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DepthRgbFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
