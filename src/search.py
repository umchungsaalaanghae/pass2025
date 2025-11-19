#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
import math
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2, Image, Imu
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


class SearchCircleMission(Node):
    def __init__(self):
        super().__init__('search_circle_mission_mavros')

        # MAVROS 속도 제어 퍼블리셔
        self.vel_pub = self.create_publisher(
            Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10
        )

        # 구독 (카메라, 라이다, IMU, 상태)
        self.create_subscription(Image, '/flir_camera/image_raw', self.camera_cb, 10)
        self.create_subscription(PointCloud2, '/ouster/points', self.lidar_cb, 10)
        self.create_subscription(Imu, '/mavros/imu/data', self.imu_cb, 10)
        self.create_subscription(State, '/mavros/state', self.state_cb, 10)

        # Arm / 모드 변경 서비스
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # 공통 변수
        self.bridge = CvBridge()
        self.target_color = "red"
        self.yaw_aligned = False
        self.closest_dist = None
        self.approach_dist = 2.0
        self.hfov_deg = 90.0
        self.image_width = 1280
        self.current_yaw = 0.0

        # Circle LOS 변수 (중요)
        self.center_x = 0.0
        self.center_y = 0.0
        self.radius = 3.0
        self.kp_yaw = 1.8
        self.linear_speed = 0.4
        self.turn_dir = 1

        # 회전 각도 관리
        self.theta = 0.0
        self.completed = False
        self.start_circle = False

        # 상태 변수
        self.current_state = None
        self.last_req = self.get_clock().now()

        # 카메라 창
        cv2.startWindowThread()
        cv2.namedWindow("Theia Camera", cv2.WINDOW_NORMAL)

        # 제어 루프 (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("Search + CircleLOS mission started")

    def state_cb(self, msg):
        self.current_state = msg

    # ---------------- GUIDED + ARM ----------------
    def try_arm_and_guided(self):
        if not self.arming_client.service_is_ready() or not self.mode_client.service_is_ready():
            return

        now = self.get_clock().now()
        if (now - self.last_req).nanoseconds < 5e9:
            return

        # 1) GUIDED 모드
        if self.current_state.mode != "GUIDED":
            self.get_logger().info("Requesting GUIDED mode...")
            req = SetMode.Request()
            req.custom_mode = "GUIDED"
            self.mode_client.call_async(req)
            self.last_req = now
            return

        # 2) ARM
        if not self.current_state.armed:
            self.get_logger().info("Requesting ARM...")
            req = CommandBool.Request()
            req.value = True
            self.arming_client.call_async(req)
            self.last_req = now
            return

        self.get_logger().info("Vehicle is ARMED and in GUIDED mode.")

    # ---------------- CAMERA ----------------
    def camera_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
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

            for color_name in ["yellow", "red1", "red2", "green"]:
                lower, upper = np.array(color_ranges[color_name][0]), np.array(color_ranges[color_name][1])
                mask = cv2.inRange(hsv, lower, upper)

                if color_name == "red1":
                    mask_red1 = mask
                    continue
                if color_name == "red2" and mask_red1 is not None:
                    mask = cv2.bitwise_or(mask_red1, mask)

                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0:
                    continue

                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) < 400:
                    continue

                x, y, w, h = cv2.boundingRect(c)
                cx = int(x + w/2)
                yaw_error = ((cx - center_x) / center_x) * (self.hfov_deg / 2)

                if self.target_color in color_name:
                    target_info = yaw_error

            # 정렬 실패
            if target_info is None:
                self.yaw_aligned = False
                self.publish_vel(0.0, 0.0)
                return

            # yaw 정렬 제어
            if abs(target_info) > 5.0:
                turn_speed = 0.25 * math.copysign(1, target_info)
                self.publish_vel(0.0, turn_speed)
                self.yaw_aligned = False
            else:
                self.publish_vel(0.0, 0.0)
                self.yaw_aligned = True

        except Exception as e:
            self.get_logger().warn(f"Camera error: {e}")

    # ---------------- LIDAR: 부표 위치 얻기 ----------------
    def lidar_cb(self, msg):
        if not self.yaw_aligned:
            return
        try:
            pts = np.array(list(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True)))
            if len(pts) == 0:
                return

            mask = np.abs(np.arctan2(pts[:,1], pts[:,0])) < math.radians(10)
            front = pts[mask]
            if len(front) == 0:
                return

            d = np.sqrt(front[:,0]**2 + front[:,1]**2)
            idx = np.argmin(d)

            self.closest_dist = float(d[idx])
            self.center_x, self.center_y = front[idx][0], front[idx][1]

            # 가까워지기 제어
            psi_des = math.atan2(self.center_y, self.center_x)
            yaw = self.current_yaw
            psi_err = math.atan2(math.sin(psi_des - yaw), math.cos(psi_des - yaw))

            if self.closest_dist > self.approach_dist:
                linear = 0.3 if abs(math.degrees(psi_err)) < 5 else 0.0
                angular = 1.5 * psi_err
                self.publish_vel(linear, angular)

            else:
                self.publish_vel(0.0, 0.0)
                self.start_circle = True
                self.theta = 0.0

        except Exception as e:
            self.get_logger().error(f"LiDAR error: {e}")

    # ---------------- IMU ----------------
    def imu_cb(self, msg):
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        siny = 2*(qw*qz + qx*qy)
        cosy = 1 - 2*(qy*qy + qz*qz)
        self.current_yaw = math.atan2(siny, cosy)

    # ---------------- Circle LOS 제어 ----------------
    def control_loop(self):
        if self.current_state is None:
            return

        if not self.current_state.armed:
            self.try_arm_and_guided()
            return

        if not self.start_circle or self.completed:
            return

        # LiDAR 기반 원 둘레 목표점 계산 (body frame)
        px = self.center_x + self.radius * math.cos(self.theta)
        py = self.center_y + self.radius * math.sin(self.theta)

        desired_yaw = math.atan2(py, px)
        yaw = self.current_yaw
        yaw_err = math.atan2(math.sin(desired_yaw - yaw), math.cos(desired_yaw - yaw))

        # 제어 명령
        self.publish_vel(self.linear_speed, self.kp_yaw * yaw_err)

        # 회전 각도 증가
        self.theta += self.turn_dir * 0.05

        # 360도 완료
        if abs(self.theta) >= 2 * math.pi:
            self.completed = True
            self.publish_vel(0.0, 0.0)
            self.get_logger().info("360° Circle LOS completed!")

    def publish_vel(self, linear, angular):
        vel = Twist()
        vel.linear.x = linear
        vel.angular.z = angular
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
