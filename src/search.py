#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
import math
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import PointCloud2, Image, Imu
from nav_msgs.msg import Path
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


class SearchCircleMission(Node):
    def __init__(self):
        super().__init__('search_circle_mission_mavros')

        # 퍼블리셔 (MAVROS 속도 제어)
        self.vel_pub = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)
        self.path_pub = self.create_publisher(Path, '/circle_path', 10)

        # 구독 (Theia + Ouster + IMU + 상태)
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

        # CircleLOS 변수
        self.center_x = 0.0
        self.center_y = 0.0
        self.radius = 3.0
        self.kp_yaw = 1.5     # 선회를 빠르게 하려면 3.0정도로 올려주기
        self.linear_speed = 0.4
        self.lookahead_angle = math.radians(20)
        self.turn_dir = 1

        # 상태 변수
        self.prev_angle = None
        self.total_angle = 0.0
        self.completed = False
        self.current_state = None

        # 카메라 창
        cv2.startWindowThread()
        cv2.namedWindow("Theia Camera", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Theia Camera", 50, 50)

        # 제어 루프 (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("Search + CircleLOS mission started")

    def state_cb(self, msg):
        self.current_state = msg

    def arm_and_guided(self):
        """Arm 및 GUIDED 모드 설정"""
        if not self.arming_client.service_is_ready() or not self.mode_client.service_is_ready():
            return

        arm_req = CommandBool.Request()
        arm_req.value = True
        self.arming_client.call_async(arm_req)

        mode_req = SetMode.Request()
        mode_req.custom_mode = "GUIDED"
        self.mode_client.call_async(mode_req)

        self.get_logger().info("Sent arming + GUIDED mode command")

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

            for color_name in ["yellow", "red1", "red2", "green"]:
                lower, upper = np.array(color_ranges[color_name][0]), np.array(color_ranges[color_name][1])
                mask = cv2.inRange(hsv, lower, upper)

                if color_name == "red1":
                    mask_red1 = mask
                    continue
                elif color_name == "red2":
                    mask = cv2.bitwise_or(mask_red1, mask)

                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))     # 노이즈 제거 
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0:
                    continue       # 윤곽선 추출

                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)
                if area < 400:
                    continue     # 가장 큰 물체 선택 

                x, y, w, h = cv2.boundingRect(c)
                cx = int(x + w / 2)     # 부표의 중심점 계산
                yaw_error = ((cx - center_x) / center_x) * (self.hfov_deg / 2)  # yaw 오차 계산

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{color_name} yaw={yaw_error:.1f}°", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  #시각화

                if self.target_color in color_name:
                    target_info = yaw_error   # 목표 색상만 yaw_error로 선택

            if target_info is None:
                self.yaw_aligned = False
                self.publish_vel(0.0, 0.0)
                cv2.imshow("Theia Camera", frame)
                cv2.waitKey(10)
                return   # 목표 부표 못 봤을 때 선박이 멈춰서 대기

            yaw_error = target_info
            
            if abs(yaw_error) > 3.0:
                turn_speed = 0.2 * math.copysign(1, yaw_error)
                self.publish_vel(0.0, turn_speed)
                self.yaw_aligned = False
                target = self.target_color
                self.get_logger().info(f"[{target}] yaw_error={yaw_error:.2f}°")
            else:
                self.publish_vel(0.0, 0.0)   # 움직임 멈춤
                self.yaw_aligned = True
                
                if self.target_color in ["red", "green"]:
                    self.turn_dir = 1
                elif self.target_color == "yellow":
                    self.turn_dir = -1
                else:
                    self.turn_dir = 1

            cv2.imshow("Theia Camera", frame)
            cv2.waitKey(10)

        except Exception as e:
            self.get_logger().warn(f"Camera error: {e}")

    # ---------------- LIDAR ----------------
    def lidar_cb(self, msg):
        if not self.yaw_aligned:   # yaw 정렬 후 실행
            return
        try:
            points_iter = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            points = np.array(list(points_iter))  # PointCloud2 → numpy 배열 로딩
            if len(points) == 0:
                return

            mask = np.abs(np.arctan2(points[:, 1], points[:, 0])) < math.radians(10)
            front = points[mask]
            if len(front) == 0:       # 정면 ±10 안에 있는 점만 선택 
                return

            dists = np.sqrt(front[:, 0] ** 2 + front[:, 1] ** 2)
            min_idx = np.argmin(dists)
            closest = front[min_idx]       # 부표로 추정되는 가장 가까운 점 선택

            self.closest_dist = float(np.min(dists))
            self.center_x, self.center_y = closest[0], closest[1]
            self.get_logger().info(f"[부표 인식] 중심=({self.center_x:.2f}, {self.center_y:.2f}), 거리={self.closest_dist:.2f}m")

            psi_desired = math.atan2(self.center_y, self.center_x)
            yaw = self.current_yaw
            psi_error = math.atan2(math.sin(psi_desired - yaw),
                                   math.cos(psi_desired - yaw))

            if self.closest_dist > self.approach_dist:

                if abs(math.degrees(psi_error)) > 5:
                    linear = 0.0
                else:
                    linear = 0.3

                angular = 1.5 * psi_error

                self.publish_vel(linear, angular)      # LOS 접근 제어

            else:
                self.publish_vel(0.0, 0.0)
                self.get_logger().info("목표 부표 중심 반경 3m 도달 — 회전 시작 준비")
                self.create_circle_path()
                self.start_circle = True
                
        except Exception as e:
            self.get_logger().error(f"LiDAR error: {e}")

    # ---------------- PATH 생성 ----------------
    def create_circle_path(self):
        path = Path()
        path.header.frame_id = "map"
        self.path_points = []

        for i in range(36):  # 10도 간격
            angle = i * 10 * math.pi / 180.0
            px = self.center_x + self.radius * math.cos(angle)
            py = self.center_y + self.radius * math.sin(angle)

            pose = PoseStamped()
            pose.pose.position.x = px
            pose.pose.position.y = py

            path.poses.append(pose)
            self.path_points.append((px, py))

        self.path_pub.publish(path)
        self.get_logger().info(f"Path 생성 완료 ({len(self.path_points)} points)")

    # ---------------- IMU ----------------
    def imu_cb(self, msg):
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy ** 2 + qz ** 2)
        self.current_yaw = math.atan2(siny, cosy)

    # ---------------- LOS CONTROL ----------------
    def control_loop(self):
        if self.current_state is None:
            return

        if not self.current_state.armed:
            self.arm_and_guided()
            return

        if (not hasattr(self, "start_circle")) or (not self.start_circle) or self.completed:
            return

        # path_points가 충분한지 확인
        if not hasattr(self, "path_points") or len(self.path_points) == 0:
            return

        # 현재 목표 웨이포인트
        target_x, target_y = self.path_points[0]

        # 위치 오차 계산
        dx = target_x
        dy = target_y
        dist_to_wp = math.sqrt(dx**2 + dy**2)

        # 웨이포인트 도달하면 다음 웨이포인트로 이동
        if dist_to_wp < 0.5:
            self.path_points.pop(0)

            # 모두 끝났다면 종료
            if len(self.path_points) == 0:
                self.publish_vel(0.0, 0.0)
                self.completed = True
                self.get_logger().info("360° 회전 완료")
                return
            return

        # LOS 헤딩 계산
        desired_heading = math.atan2(dy, dx)
        yaw = self.current_yaw
        yaw_error = math.atan2(math.sin(desired_heading - yaw),
                               math.cos(desired_heading - yaw))

        # 제어 출력
        linear_speed = self.linear_speed
        angular_speed = self.kp_yaw * yaw_error

        self.publish_vel(linear_speed, angular_speed)


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
        node.get_logger().info("Shutting down")
        node.publish_vel(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
