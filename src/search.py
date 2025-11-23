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
from sensor_msgs.msg import Image, Imu
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
        self.approach_dist = 2.0   # ★ 2.0 m 이내로 접근하면 원회전 시작
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
        
        # ★ [수정] 락온 손실 방지용 변수 추가
        self.max_lost_frames = 10   # 락온 마커가 사라져도 허용할 최대 프레임 수 (약 0.5초)
        self.lost_frame_count = 0   # 마커를 놓친 프레임 카운터
        # ★ [수정] 좌표 급변 필터링 변수 추가
        self.max_update_dist = 1.0  # 락온된 마커가 한 프레임에 허용되는 최대 이동 거리 (1.0m)

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

    # ========== 카메라 + YOLO (수정됨) ==========
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
            # 타겟이 안 보이면 정지 → 제자리 선회
            # ------------------------------
            if target_info is None:
                # --- 타겟이 안 보이면 제자리 선회 ---
                turn_speed = 0.2  # 제자리 선회 속도
                self.publish_vel(0.0, turn_speed) # 전진 속도 0.0
                self.yaw_aligned = False
            
            
            else:
                yaw_error = target_info

                # ------------------------------
                # Yaw 정렬 여부 판단 (±5°)
                # ------------------------------
                if abs(yaw_error) > 5.0:
                    # 정렬 전 → 회전 + 느린 전진
                    turn_speed = 0.2 * math.copysign(1, yaw_error)
                    
                    # 💡 [수정] 정렬 중 느린 전진 속도 (0.1 m/s) 추가
                    forward_speed = 0.1 
                    self.publish_vel(forward_speed, turn_speed) 
                    
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

    # ========== 라이다 중심점 (수정됨 - 락온 손실 방지 및 좌표 안정화 필터 추가) ==========
    def centroid_cb(self, msg: MarkerArray):

        # yaw 정렬 안 됐으면 아직 라이다 안 씀
        if not self.yaw_aligned:
            return

        candidates = []
        found_locked_marker = False
        
        # ----------------------------------------
        # 모든 마커를 후보군에 저장 (근접 물체 필터링 적용)
        # ----------------------------------------
        for marker in msg.markers:
            if marker.ns != "cluster_centroids_sphere":
                continue

            x = marker.pose.position.x
            y = marker.pose.position.y

            angle_rad = math.atan2(y, -x)
            angle_deg = math.degrees(angle_rad)
            dist = math.sqrt(x**2 + y**2)

            # --- 근접 거리 필터링 (1.0m 이내 제외) ---
            if dist < 1.0:
                continue

            candidates.append((dist, marker.id, x, y, angle_deg))
            
            # 락온된 마커가 있는지 확인
            if self.locked and marker.id == self.locked_id:
                found_locked_marker = True
        
        # 후보가 없으면 종료
        if not candidates:
            # 락온 상태였다면 락온 해제
            if self.locked:
                # 락온 마커뿐만 아니라 모든 후보가 사라진 경우 (즉시 해제)
                self.get_logger().warn("[TRACK] All markers lost! Reverting to search mode.")
                self.locked = False
                self.locked_id = None
                self.closest_dist = None
                self.lost_frame_count = 0 # 카운터 초기화
                
            self.closest_dist = None
            return


        # ----------------------------------------
        # [A] 락온된 상태 (안정적인 추적)
        # ----------------------------------------
        if self.locked:
            
            if found_locked_marker:
                # 락온된 마커를 찾았으므로 해당 마커의 좌표를 사용
                for dist, mid, cx, cy, angle_deg in candidates:
                    if mid == self.locked_id:
                        
                        # ★ [수정] 1. 좌표 급변 필터링 (LiDAR 좌표 안정화)
                        if self.closest_dist is not None:
                            dist_change = math.sqrt(
                                (cx - self.center_x)**2 + (cy - self.center_y)**2
                            )
                            
                            if dist_change > self.max_update_dist:
                                self.get_logger().warn(
                                    f"[TRACK] Rejected large jump ({dist_change:.2f} m > {self.max_update_dist:.1f} m). Holding previous coordinates."
                                )
                                self.lost_frame_count = 0 # ID는 찾았으므로 손실 카운트는 초기화
                                return # 좌표 업데이트 건너뛰고 기존 값 유지
                        
                        # 2. 좌표 갱신 (필터 통과 시)
                        self.center_x = cx
                        self.center_y = cy
                        self.closest_dist = dist
                        
                        # 마커를 찾았으므로 손실 카운터를 초기화
                        self.lost_frame_count = 0 
                        
                        # 로그 주기 제한
                        now = self.get_clock().now()
                        if (
                            self.last_lidar_log_time is None
                            or (now - self.last_lidar_log_time).nanoseconds > 5e8
                        ):
                            self.get_logger().info(
                                f"[TRACK] id={mid}, x={cx:.2f}, y={cy:.2f}, angle={angle_deg:.1f}°, dist={dist:.2f} (Locked)"
                            )
                            self.last_lidar_log_time = now
                        return # 락온 추적 완료

            else:
                # 락온된 마커를 찾지 못함 -> 카운트 증가
                self.lost_frame_count += 1
                
                if self.lost_frame_count < self.max_lost_frames:
                    # 최대 허용 프레임에 도달하지 않았으면 락온 상태 유지 (이전 좌표 사용)
                    self.get_logger().warn(
                        f"[TRACK] Locked marker lost ({self.lost_frame_count}/{self.max_lost_frames}). Holding previous position."
                    )
                    return
                else:
                    # 최대 허용 프레임을 초과하면 락온 해제
                    self.get_logger().warn("[TRACK] Locked marker lost! Reverting to search mode.")
                    self.locked = False
                    self.locked_id = None
                    self.closest_dist = None
                    self.lost_frame_count = 0 # 카운터 초기화
                    # 재탐색을 위해 함수를 다시 실행하지 않고 다음 프레임 대기
                    return

        # ----------------------------------------
        # [B] 락온 안 된 상태 (최초 락온 시도)
        # ----------------------------------------
        if not self.locked:
            
            filtered_candidates = []
            # ±10° 필터 적용
            for dist, mid, cx, cy, angle_deg in candidates:
                if abs(angle_deg) <= 10.0:
                    filtered_candidates.append((dist, mid, cx, cy, angle_deg))
            
            if not filtered_candidates:
                self.closest_dist = None
                return

            # 가장 가까운 marker 선택
            dist, mid, cx, cy, angle_deg = min(filtered_candidates, key=lambda t: t[0])
            
            # 최초 락온
            self.locked = True
            self.locked_id = mid
            self.lost_frame_count = 0 # 카운터 초기화
            self.get_logger().info(f"[LOCK ON] Marker ID {mid} locked as target (Initial Lock)")
            
            # 선택된 부표 좌표 갱신
            self.center_x = cx
            self.center_y = cy
            self.closest_dist = dist
            
            # 로그 출력
            now = self.get_clock().now()
            if (
                self.last_lidar_log_time is None
                or (now - self.last_lidar_log_time).nanoseconds > 5e8
            ):
                self.get_logger().info(
                    f"[TRACK] id={mid}, x={cx:.2f}, y={cy:.2f}, angle={angle_deg:.1f}°, dist={dist:.2f} (New Lock)"
                )
                self.last_lidar_log_time = now
            return

    def create_circle_path(self):
        path = Path()
        path.header.frame_id = "map"
        self.path_points = []

        for i in range(36):
            angle = i * 10 * math.pi / 180.0
            # 라이다 좌표계 (X: 전방, Y: 좌측)
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
            # 유효한 라이다 타겟(락온된)이 아직 없으면 대기
            if self.closest_dist is None or not self.locked:
                # 락온이 안 됐거나 데이터가 없으면 정지
                self.publish_vel(0.0, 0.0) 
                return

            now = self.get_clock().now()
            if self.last_dist_log_time is None or \
               (now - self.last_dist_log_time).nanoseconds > 5e8:  # 0.5초(=5e8ns)
                self.get_logger().info(
                    f"[APPROACH] 현재 장애물까지 거리 = {self.closest_dist:.2f} m "
                    f"(center=({self.center_x:.2f}, {self.center_y:.2f}), id={self.locked_id})"
                )
                self.last_dist_log_time = now

            # 접근 거리 밖이면 → 직진
            if self.closest_dist > self.approach_dist:
                # 여기서는 단순 직진 (yaw는 카메라/요요로 맞춘 상태라고 가정)
                self.publish_vel(self.linear_speed, 0.0)

            else:
                # 접근 거리 이내 들어오면 → 멈추고 원형 경로 생성 + 선회 시작
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
            # 완료 후 정지
            self.publish_vel(0.0, 0.0)
            return

        # --------------------------------
        # 3) Circle LOS 기반 궤도 추종 (360° 회전)
        # --------------------------------
        # 현재 중심점 좌표가 유효한지 확인
        if self.closest_dist is None:
            self.publish_vel(0.0, 0.0)
            self.get_logger().warn("[CIRCLE] Lost track of center point, halting movement.")
            return

        # 기체에서 중심점까지의 벡터 (라이다 프레임: X전방, Y좌측)
        dx = self.center_x
        dy = self.center_y
        
        # 기체 기준 (0,0)에서 (dx, dy)를 바라보는 각도 (atan2(y, x))
        # Note: 라이다 좌표계 (X:전방, Y:좌측)를 사용. atan2(Y, X)가 맞음.
        theta_c = math.atan2(dy, dx) 

        # 360도 회전 체크 로직
        if self.prev_angle is None:
            self.prev_angle = theta_c
        
        # 절대값 누적으로 360도 체크
        delta_abs = abs(theta_c - self.prev_angle)
        if delta_abs > math.pi: # 180도를 넘어서면 360도 회전한 것으로 간주
            delta_abs = 2 * math.pi - delta_abs

        self.total_angle += delta_abs
        self.prev_angle = theta_c

        if self.total_angle >= 2 * math.pi:
            self.publish_vel(0.0, 0.0)
            self.completed = True
            self.get_logger().info("Completed 360 degree rotation")
            return

        # LOS 제어 (원의 접선 방향으로 Heading 계산)
        if self.turn_dir == 1:
            # 시계 방향 회전 (우측 선회)
            desired_heading = math.atan2(-dx, dy)
        else:
            # 반시계 방향 회전 (좌측 선회)
            desired_heading = math.atan2(dx, -dy)

        # Yaw 에러 계산
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
