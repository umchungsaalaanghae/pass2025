#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from ultralytics import YOLO
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from visualization_msgs.msg import MarkerArray
import math
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import  Image, Imu
from nav_msgs.msg import Path
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


class SearchCircleMission(Node):
    def __init__(self):
        super().__init__('search_circle_mission_mavros')

        # =========================
        # YOLO 모델 경로 파라미터
        # =========================
        self.declare_parameter('yolo_model_path', 'weights/docking_rsp.pt')
        self.yolo_model_path = (
            self.get_parameter('yolo_model_path')
            .get_parameter_value()
            .string_value
        )
        self.get_logger().info(f"[YOLO] Model path = {self.yolo_model_path}")

        # YOLO 모델 로드
        self.model = YOLO(self.yolo_model_path)
        self.get_logger().info("[YOLO] Model loaded successfully")

        # Publisher
        self.vel_pub = self.create_publisher(
            Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10
        )
        self.path_pub = self.create_publisher(Path, '/circle_path', 10)

        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.create_subscription(Image, '/flir_camera/image_raw', self.camera_cb, 10)
        self.create_subscription(MarkerArray, '/obstacle_centroids', self.centroid_cb, 10)
        self.create_subscription(State, '/mavros/state', self.state_cb, 10)
        self.create_subscription(Imu, '/mavros/imu/data', self.imu_cb, imu_qos)

        # center_point (LiDAR 상대좌표 입력) – 필요 시 사용
        self.center_point = None
        self.create_subscription(Point, '/center_point', self.center_cb, 10)

        # Services
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # Common variables
        self.bridge = CvBridge()
        # YOLO에서 사용하는 클래스 이름 (예: Rock / Paper / Scissors 중 하나)
        self.target_color = "Scissors"
        self.yaw_aligned = False
        self.closest_dist = None    # 라이다에서 들어온 장애물까지의 거리
        self.approach_dist = 2.5   # ★ 2.5 m 이내로 접근하면 원회전 시작
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
        self.start_circle = False   # 원형 선회 시작 여부

        self.yaw_aligned = False     # 현재 정렬 상태                        
        self.yaw_finished = False    # ★ 한번 정렬 끝났는지 여부
        
        # ★ 부표 lock-on용 플래그
        self.locked = False         # True면 특정 부표에 lock
        self.locked_id = None       # lock된 Marker의 id

        self.last_lidar_log_time = None
        self.last_dist_log_time = None

        # Camera window
        cv2.startWindowThread()
        cv2.namedWindow("Theia Camera", cv2.WINDOW_NORMAL)

        # Control loop timer
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("Search + CircleLOS mission started")

    # =======================
    #  콜백 함수들
    # =======================
    def state_cb(self, msg: State):
        self.current_state = msg

    def center_cb(self, msg: Point):
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

    # ========== 카메라 + YOLO ==========
    def camera_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"Camera error: {e}")
            return

        # ★ 한 번 yaw 정렬이 끝났으면, 이후에는 카메라로 yaw 제어 안 함
        if self.yaw_finished:
            try:
                cv2.imshow("Theia Camera", frame)
                cv2.waitKey(1)
            except Exception:
                pass
            # yaw_aligned는 계속 True로 유지
            self.yaw_aligned = True
            return

        center_x = self.image_width // 2
        target_info = None

        # ------------------------------
        # YOLO 추론
        # ------------------------------
        try:
            results = self.model(frame, verbose=False)
        except Exception as e:
            self.get_logger().warn(f"YOLO error: {e}")
            return

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                cls_name = r.names[cls]

                if cls_name != self.target_color:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = int((x1 + x2) / 2)

                yaw_error = ((cx - center_x) / center_x) * (self.hfov_deg / 2)

                cv2.rectangle(frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_name} yaw={yaw_error:.1f}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)

                target_info = yaw_error

            # ------------------------------
            # 타겟이 안 보이면 정지
            # ------------------------------
            if target_info is None:
                self.publish_vel(0.0, 0.0) #--------------------------> 타겟이 안 보이면 정지. 근데 만약에 배가 돌아가서 계속 안보이면 어떡해? 그니까 제자리 선회나 이런걸 넣어야지. ㅇㅋ 수정하셈
                self.yaw_aligned = False

            else:
                yaw_error = target_info

                # ------------------------------
                # Yaw 정렬 여부 판단 (±5°)
                # ------------------------------
                if abs(yaw_error) > 5.0:
                    # 정렬 전 → 회전 계속
                    turn_speed = 0.2 * math.copysign(1, yaw_error)
                    self.publish_vel(0.0, turn_speed)
                    self.yaw_aligned = False
                else:
                    # 정렬 완료
                    self.publish_vel(0.0, 0.0)
                    self.yaw_aligned = True
                    self.yaw_finished = True 

                    # Rock/Paper/Scissors에 따라 회전 방향 설정 (예시)
                    if self.target_color in ["Paper", "Rock"]:
                        self.turn_dir = 1      # 시계 방향
                    elif self.target_color == "Scissors":
                        self.turn_dir = -1     # 반시계 방향
                    else:
                        self.turn_dir = 1

                    self.get_logger().info(
                        f"[{self.target_color}] yaw 정렬 완료 → "
                        f"{'시계방향' if self.turn_dir == 1 else '반시계방향'} 회전 준비"
                    )

        # ------------------------------
        # 항상 마지막에 카메라 창 출력
        # ------------------------------
        try:
            cv2.imshow("Theia Camera", frame)
            cv2.waitKey(1)
        except Exception:
            pass

    # ========== 라이다 중심점 ==========
    def centroid_cb(self, msg: MarkerArray):
        # yaw 정렬 안 됐으면 아직 라이다 안 씀
        if not self.yaw_aligned:
            return

        # 정면 ±10도 안에 있는 점들 중에서 "가장 가까운" 것 하나만 선택
        candidates = []

        for marker in msg.markers:
            if marker.ns != "cluster_centroids_sphere":
                continue

            x = marker.pose.position.x   # LiDAR 기준
            y = marker.pose.position.y

            # 전방을 -x 방향으로 가정
            angle_rad = math.atan2(y, -x)
            angle_deg = math.degrees(angle_rad)

            # 정면 ±10도만 사용
            if abs(angle_deg) > 10.0:
                continue

            dist = math.sqrt(x**2 + y**2)
            candidates.append((dist, x, y, angle_deg))

        # 후보 없으면 타겟 없음
        if not candidates:
            self.closest_dist = None
            return

        # 거리 기준으로 가장 가까운 점 하나 선택
        dist, cx, cy, angle_deg = min(candidates, key=lambda t: t[0])

        # 이 점만 계속 사용
        self.center_x = cx
        self.center_y = cy
        self.closest_dist = dist

        # 0.5초마다 로그
        now = self.get_clock().now()
        if (
            self.last_lidar_log_time is None
            or (now - self.last_lidar_log_time).nanoseconds > 5e8
        ):
            self.get_logger().info(
                f"[LIDAR-TRACK] x={self.center_x:.2f}, y={self.center_y:.2f}, "
                f"angle={angle_deg:.1f}°, dist={self.closest_dist:.2f} m"
            )
            self.last_lidar_log_time = now

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

    def imu_cb(self, msg: Imu):
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy ** 2 + qz ** 2)
        self.current_yaw = math.atan2(siny, cosy)

    # =======================
    #  메인 제어 루프
    # =======================
    def control_loop(self):
        # Pixhawk 상태 확인
        if self.current_state is None:
            return

        # ARM + GUIDED 모드 전환
        if not self.current_state.armed:
            self.arm_and_guided()
            return

        # --------------------------------
        # 1) 아직 원형 선회 시작 전 (접근 단계)
        # --------------------------------
        if not self.start_circle:
            # 유효한 라이다 타겟(정면 ±10°)이 아직 없으면 대기
            if self.closest_dist is None:
                return

            now = self.get_clock().now()
            if self.last_dist_log_time is None or \
               (now - self.last_dist_log_time).nanoseconds > 5e8:  # 0.5초(=5e8ns)
                self.get_logger().info(
                    f"[APPROACH] 현재 장애물까지 거리 = {self.closest_dist:.2f} m "
                    f"(center=({self.center_x:.2f}, {self.center_y:.2f}))"
                )
                self.last_dist_log_time = now

            # 아직 3m 밖이면 → 직진
            if self.closest_dist > self.approach_dist:
                # 여기서는 단순 직진 (yaw는 카메라/요요로 맞춘 상태라고 가정)
                self.publish_vel(self.linear_speed, 0.0)

            else:
                # 3m 이내 들어오면 → 멈추고 원형 경로 생성 + 선회 시작
                self.publish_vel(0.0, 0.0)

                if not self.start_circle:
                    self.get_logger().info(
                        f"장애물까지 거리 {self.closest_dist:.2f} m ≤ {self.approach_dist:.1f} m. "
                        f"원회전 시작."
                    )
                    self.create_circle_path()
                    self.start_circle = True
                    # 회전 각도 누적 초기화
                    self.prev_angle = None
                    self.total_angle = 0.0
            return

        # --------------------------------
        # 2) 원형 선회 완료된 경우
        # --------------------------------
        if self.completed:
            return

        # --------------------------------
        # 3) Circle LOS 기반 궤도 추종 (360° 회전)
        # --------------------------------
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
