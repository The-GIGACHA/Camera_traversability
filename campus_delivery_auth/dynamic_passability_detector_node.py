#!/usr/bin/env python3
"""
dynamic_passability_detector_node.py  (ROS1)
=============================================
Pseudo-BEV 기반 동적 객체 인지 및 통과성 판단 노드.

ROS1 포팅 변경사항:
  - Node 클래스 상속 없음
  - create_subscription/publisher/timer → rospy.Subscriber/Publisher/Timer
  - tf2_ros + tf2_geometry_msgs → tf (ROS1 TF)
  - sensor_msgs_py.point_cloud2 → sensor_msgs.point_cloud2
  - rclpy.duration.Duration → rospy.Duration
  - get_logger() → rospy.log*()
"""

import math
import numpy as np
import rospy

from sensor_msgs.msg import CompressedImage, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Bool, Header
import sensor_msgs.point_cloud2 as pc2

import message_filters

import cv2
from cv_bridge import CvBridge

from ultralytics import YOLO

import tf
from geometry_msgs.msg import PointStamped


# ───────────────────────────── 파라미터 상수 ──────────────────────────────── #

ROBOT_WIDTH_M       = 0.4
SAFETY_MARGIN_M     = 0.4
PASS_THRESHOLD_M    = ROBOT_WIDTH_M + SAFETY_MARGIN_M   # 0.8 m

YOLO_CONF_THRESH    = 0.5
TARGET_CLASSES      = {0, 1}    # person, bicycle

DEPTH_MIN_M         = 0.3
DEPTH_MAX_M         = 6.0

FOV_DEG             = 120.0
MAX_RANGE_M         = 5.0

VIRTUAL_WALL_STEP_M = 0.05
DEPTH_MEDIAN_RATIO  = 0.2
SYNC_SLOP_SEC       = 0.05

# ──────────────────────────────────────────────────────────────────────────── #


class DynamicPassabilityDetector:

    def __init__(self):
        rospy.init_node('dynamic_passability_detector_node', anonymous=False)

        # ── YOLOv8 ──
        self.model = YOLO('yolov8n.pt')
        rospy.loginfo('YOLOv8n loaded (COCO pretrained)')

        # ── CvBridge ──
        self.bridge = CvBridge()

        # ── TF (ROS1) ──
        self.tf_listener = tf.TransformListener()

        # ── Step 1: 동기화 구독 ──
        sub_color = message_filters.Subscriber(
            '/camera/color/image_raw/compressed', CompressedImage
        )
        sub_depth = message_filters.Subscriber(
            '/camera/aligned_depth_to_color/image_raw/compressedDepth', CompressedImage
        )
        sub_info = message_filters.Subscriber(
            '/camera/color/camera_info', CameraInfo
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_color, sub_depth, sub_info],
            queue_size=10,
            slop=SYNC_SLOP_SEC,
        )
        self.sync.registerCallback(self.synced_callback)

        # ── Step 5: 발행 ──
        self.pub_obstacles = rospy.Publisher(
            '/dynamic_obstacles/pointcloud', PointCloud2, queue_size=10
        )
        self.pub_passable = rospy.Publisher(
            '/dynamic_obstacles/passable', Bool, queue_size=10
        )

        rospy.loginfo('dynamic_passability_detector_node ready.')

    # ─────────────────────────────────────────────────────────────────────── #

    def synced_callback(self, color_msg, depth_msg, info_msg):
        color_img, depth_img = self._decode_images(color_msg, depth_msg)
        if color_img is None or depth_img is None:
            return

        detections  = self._run_yolo(color_img)
        cam_points  = self._project_to_3d(detections, depth_img, info_msg)
        robot_points = self._transform_to_base_link(cam_points, info_msg.header)
        filtered    = self._filter_fov(robot_points)
        passable, wall_pts = self._judge_passability(filtered)
        self._publish_results(info_msg.header, filtered, wall_pts, passable)

    # ── Step 2 ───────────────────────────────────────────────────────────── #

    def _decode_images(self, color_msg, depth_msg):
        try:
            color_img = self.bridge.compressed_imgmsg_to_cv2(
                color_msg, desired_encoding='bgr8'
            )
        except Exception as e:
            rospy.logwarn(f'color decode failed: {e}')
            return None, None

        try:
            depth_img = self.bridge.compressed_imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough'
            )
        except Exception as e:
            rospy.logwarn(f'depth decode failed: {e}')
            return None, None

        return color_img, depth_img

    def _run_yolo(self, color_img):
        results = self.model.predict(color_img, conf=YOLO_CONF_THRESH, verbose=False)
        detections = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls not in TARGET_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            detections.append((cx, cy, float(box.conf[0]), cls, x1, y1, x2, y2))
        return detections

    # ── Step 3 ───────────────────────────────────────────────────────────── #

    def _get_robust_depth(self, depth_img, x1, y1, x2, y2):
        h, w = depth_img.shape[:2]
        roi = depth_img[
            max(0, int(y1)):min(h, int(y2)),
            max(0, int(x1)):min(w, int(x2))
        ].astype(np.float32)
        roi_m = roi / 1000.0
        valid = roi_m[(roi_m >= DEPTH_MIN_M) & (roi_m <= DEPTH_MAX_M)]
        if valid.size == 0:
            return None
        return float(np.percentile(valid, DEPTH_MEDIAN_RATIO * 100))

    def _project_to_3d(self, detections, depth_img, info_msg):
        fx = info_msg.K[0]; fy = info_msg.K[4]
        cx = info_msg.K[2]; cy = info_msg.K[5]
        # ROS1 CameraInfo: K는 1D list (9 elements), ROS2는 k

        cam_points = []
        for (u, v, conf, cls, x1, y1, x2, y2) in detections:
            Z = self._get_robust_depth(depth_img, x1, y1, x2, y2)
            if Z is None:
                continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            cam_points.append((X, Y, Z, cls))
        return cam_points

    # ── Step 4-a: TF (ROS1 방식) ─────────────────────────────────────────── #

    def _transform_to_base_link(self, cam_points, header):
        robot_points = []
        for (Xc, Yc, Zc, cls) in cam_points:
            pt = PointStamped()
            pt.header = header
            pt.point.x = float(Xc)
            pt.point.y = float(Yc)
            pt.point.z = float(Zc)

            try:
                # ROS1 tf.TransformListener: canTransform 후 transformPoint
                self.tf_listener.waitForTransform(
                    'base_link', header.frame_id,
                    header.stamp,
                    rospy.Duration(0.05),
                )
                pt_base = self.tf_listener.transformPoint('base_link', pt)
            except (tf.Exception, tf.LookupException,
                    tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logdebug(f'TF transform failed: {e}')
                continue

            robot_points.append((pt_base.point.x, pt_base.point.y, cls))
        return robot_points

    # ── Step 4-b: FOV 필터 ───────────────────────────────────────────────── #

    def _filter_fov(self, robot_points):
        half_fov = math.radians(FOV_DEG / 2.0)
        filtered = []
        for (x, y, cls) in robot_points:
            dist = math.hypot(x, y)
            if dist > MAX_RANGE_M or x <= 0:
                continue
            if abs(math.atan2(y, x)) > half_fov:
                continue
            filtered.append((x, y, cls))
        filtered.sort(key=lambda p: math.hypot(p[0], p[1]))
        return filtered

    # ── Step 4-c: 통과성 판단 ────────────────────────────────────────────── #

    def _judge_passability(self, robot_points):
        n = len(robot_points)
        if n < 2:
            return True, []

        min_dist = float('inf')
        critical_pair = None
        for i in range(n - 1):
            ax, ay, _ = robot_points[i]
            bx, by, _ = robot_points[i + 1]
            d = math.hypot(ax - bx, ay - by)
            if d < min_dist:
                min_dist = d
                critical_pair = (robot_points[i], robot_points[i + 1])

        if min_dist > PASS_THRESHOLD_M:
            return True, []

        return False, self._make_virtual_wall(critical_pair[0], critical_pair[1])

    def _make_virtual_wall(self, pt_a, pt_b):
        ax, ay, _ = pt_a
        bx, by, _ = pt_b
        dist  = math.hypot(ax - bx, ay - by)
        n_pts = max(2, int(dist / VIRTUAL_WALL_STEP_M))
        return [
            (ax + i / n_pts * (bx - ax), ay + i / n_pts * (by - ay), 0.0)
            for i in range(n_pts + 1)
        ]

    # ── Step 5: 발행 ─────────────────────────────────────────────────────── #

    def _publish_results(self, header, robot_points, wall_pts, passable):
        combined = [(x, y, 0.0) for (x, y, _) in robot_points] + wall_pts

        out_header = Header()
        out_header.stamp    = header.stamp
        out_header.frame_id = 'base_link'

        fields = [
            PointField('x', 0,  PointField.FLOAT32, 1),
            PointField('y', 4,  PointField.FLOAT32, 1),
            PointField('z', 8,  PointField.FLOAT32, 1),
        ]
        cloud_msg = pc2.create_cloud(out_header, fields, combined)
        self.pub_obstacles.publish(cloud_msg)

        passable_msg = Bool()
        passable_msg.data = passable
        self.pub_passable.publish(passable_msg)

        rospy.loginfo(
            f'obstacles={len(robot_points)}  wall_pts={len(wall_pts)}  passable={passable}'
        )

    # ─────────────────────────────────────────────────────────────────────── #

    def spin(self):
        rospy.spin()


# ──────────────────────────────────────────────────────────────────────────── #

if __name__ == '__main__':
    node = DynamicPassabilityDetector()
    node.spin()
