#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
import math
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import PointCloud2, Image, Imu
from nav_msgs.msg import Path
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


class SearchCircleMission(Node):
    def __init__(self):
        super().__init__('search_circle_mission_mavros')

        # Publisher
        self.vel_pub = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)
        self.path_pub = self.create_publisher(Path, '/circle_path', 10)

        # Subscribers
        self.create_subscription(Image, '/flir_camera/image_raw', self.camera_cb, 10)
        self.create_subscription(Point, '/obstacle_centroids', self.centroid_cb, 10)
        self.create_subscription(Imu, '/mavros/imu/data', self.imu_cb, 10)
        self.create_subscription(State, '/mavros/state', self.state_cb, 10)

        # center_point (LiDAR 상대좌표 입력)
        self.center_point = None
        self.create_subscription(Point, '/center_point', self.center_cb, 10)

        # Services
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # Common variables
        self.bridge = CvBridge()
        self.target_color = "red"
        self.yaw_aligned = False
        self.closest_dist = None
        self.approach_dist = 2.0
        self.hfov_deg = 90.0
        self.image_width = 1280
        self.current_yaw = 0.0

        # Circle LOS variables
        self.center_x = 0.0
        self.center_y = 0.0
        self.radius = 3.0
        self.kp_yaw = 1.5
        self.linear_speed = 0.4
        self.lookahead_angle = math.radians(20)
        self.turn_dir = 1

        # Rotation check variables
        self.prev_angle = None
        self.total_angle = 0.0
        self.completed = False
        self.current_state = None
        self.start_circle = False

        # Camera window
        cv2.startWindowThread()
        cv2.namedWindow("Theia Camera", cv2.WINDOW_NORMAL)

        # Control loop timer
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("Search + CircleLOS mission started")

    def state_cb(self, msg):
        self.current_state = msg

    def center_cb(self, msg):
        self.center_point = (msg.x, msg.y)

    def arm_and_guided(self):
        if not self.arming_client.service_is_ready() or not self.mode_client.service_is_ready():
            return

        arm_req = CommandBool.Request()
        arm_req.value = True
        self.arming_client.call_async(arm_req)

        mode_req = SetMode.Request()
        mode_req.custom_mode = "GUIDED"
        self.mode_client.call_async(mode_req)

    def camera_cb(self, msg):
        try:
            # 항상 frame 먼저 읽기
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # yaw 정렬 완료되면 화면만 유지
            if self.yaw_aligned:
                cv2.imshow("Theia Camera", frame)
                cv2.waitKey(10)
                return

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            center_x = self.image_width // 2

            color_ranges = {
                "red1": ([0, 80, 80], [10, 255, 255]),
                "red2": ([160, 80, 80], [179, 255, 255]),
                "green": ([35, 60, 60], [85, 255, 255]),
                "yellow": ([15, 80, 80], [55, 255, 255]),
            }

            target_info = None
            mask_red1 = None

            for cname in ["yellow", "red1", "red2", "green"]:
                lower = np.array(color_ranges[cname][0])
                upper = np.array(color_ranges[cname][1])
                mask = cv2.inRange(hsv, lower, upper)

                if cname == "red1":
                    mask_red1 = mask
                    continue

                if cname == "red2" and mask_red1 is not None:
                    mask = cv2.bitwise_or(mask_red1, mask)

                # Morphology
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0:
                    continue

                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) < 400:
                    continue

                x, y, w, h = cv2.boundingRect(c)
                cx = int(x + w / 2)
                yaw_error = ((cx - center_x) / center_x) * (self.hfov_deg / 2)

                if self.target_color in cname:
                    target_info = yaw_error

            # 찾지 못했을 때
            if target_info is None:
                self.yaw_aligned = False
                self.publish_vel(0.0, 0.0)
                cv2.imshow("Theia Camera", frame)
                cv2.waitKey(10)
                return

            # yaw 오차에 따라 회전
            if abs(target_info) > 5.0:
                turn_speed = 0.2 * math.copysign(1, target_info)
                self.publish_vel(0.0, turn_speed)
            else:
                self.publish_vel(0.0, 0.0)
                self.yaw_aligned = True
                self.turn_dir = 1 if self.target_color in ["red", "green"] else -1

            cv2.imshow("Theia Camera", frame)
            cv2.waitKey(10)

        except Exception as e:
            self.get_logger().warn(f"Camera error: {e}")

            
    def centroid_cb(self, msg: Point):
        # DBSCAN에서 퍼블리시된 중심점 → 로컬 좌표계 기준
        self.center_x = msg.x
        self.center_y = msg.y

        # 중심점까지의 거리
        self.closest_dist = math.sqrt(msg.x**2 + msg.y**2)

        self.get_logger().info(
            f"[DBSCAN] Obstacle centroid = ({self.center_x:.2f}, {self.center_y:.2f}), dist={self.closest_dist:.2f}"
        )

        # 반경 진입 여부 체크
        if self.closest_dist <= self.approach_dist and not self.start_circle:
            self.publish_vel(0.0, 0.0)
            self.get_logger().info("Reached 3m from centroid. Creating circle path.")
            self.create_circle_path()
            self.start_circle = True


    def create_circle_path(self):
        path = Path()                                      
        path.header.frame_id = "map"
        self.path_points = []

        for i in range(36):
            angle = i * 10 * math.pi / 180.0
            px = self.center_x + self.radius * math.cos(angle)
            py = self.center_y + self.radius * math.sin(angle)

            pose = PoseStamped()
            pose.pose.position.x = px
            pose.pose.position.y = py

            path.poses.append(pose)
            self.path_points.append((px, py))

        self.path_pub.publish(path)
        self.get_logger().info(f"Circle path created with {len(self.path_points)} points")

    def imu_cb(self, msg):
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy ** 2 + qz ** 2)
        self.current_yaw = math.atan2(siny, cosy)

    def control_loop(self):
        if self.current_state is None:
            return

        if not self.current_state.armed:
            self.arm_and_guided()
            return

        if not self.start_circle or self.completed:
            return

        dx = -self.center_x
        dy = -self.center_y
        theta_c = math.atan2(dy, dx)

        if self.prev_angle is None:
            self.prev_angle = theta_c

        delta = (theta_c - self.prev_angle + math.pi) % (2 * math.pi) - math.pi
        self.total_angle += abs(delta)
        self.prev_angle = theta_c

        if self.total_angle >= 2 * math.pi:
            self.publish_vel(0.0, 0.0)
            self.completed = True
            self.get_logger().info("Completed 360 degree rotation")
            return

        target_theta = theta_c + self.turn_dir * self.lookahead_angle
        xt = self.radius * math.cos(target_theta)
        yt = self.radius * math.sin(target_theta)
        desired_heading = math.atan2(yt, xt)

        error_yaw = (desired_heading - self.current_yaw + math.pi) % (2 * math.pi) - math.pi
        angular_speed = self.kp_yaw * error_yaw

        self.publish_vel(self.linear_speed, angular_speed)

    def publish_vel(self, linear_x, angular_z):
        vel = Twist()
        vel.linear.x = linear_x
        vel.angular.z = angular_z
        self.vel_pub.publish(vel)


def main(args=None):
    rclpy.init(args=args)
    node = SearchCircleMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_vel(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
