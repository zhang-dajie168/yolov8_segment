#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, Point32
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import traceback
import gc  # ✅ 导入 GC 模块
from typing import List, Dict, Optional
import os
from ament_index_python.packages import get_package_share_directory

from .yolov8_segment_model import YOLOv8_Seg
from .BOTSort_rdk import BOTSORT, OSNetReID


class YOLOv8SegROSNode(Node):
    
    def _resolve_model_path(self, p: str) -> str:
        if not p:
            return p
        if os.path.isabs(p):
            return p
        pkg_share = get_package_share_directory('yolov8_segment')
        return os.path.join(pkg_share, p)
    
    def __init__(self):
        super().__init__('yolov8_seg_track_node')

        # ---------- 参数声明 ----------
        self.declare_parameter('model_path', '')
        self.declare_parameter('conf_threshold', 0.3)
        self.declare_parameter('iou_threshold', 0.45)
        # 检测帧率上限：<=0 表示不限制，来一帧处理一帧
        self.declare_parameter('max_processing_fps', 0.0)
        self.declare_parameter('tracking_protection_time', 5.0)
        self.declare_parameter('reid_similarity_threshold', 0.8)
        self.declare_parameter('height_change_threshold', 0.15)
        self.declare_parameter('lost_timeout_threshold', 10.0)
        self.declare_parameter('reid_model_path', 'osnet_64x128_nv12.bin')
        self.declare_parameter('mask_alpha', 0.3)
        self.declare_parameter('draw_detections', False)
        self.declare_parameter('min_mask_points', 10)
        self.declare_parameter('enable_debug_log', False)
        self.declare_parameter('center_rect_width_ratio', 0.3)
        self.declare_parameter('center_confirm_time', 3.0)
        self.declare_parameter('draw_center_zone', True)
        self.declare_parameter('input_topic', '/camera_58/image_raw')
        self.declare_parameter('input_encoding', 'nv12')
        
        # 可视化参数（/seg_track/combined 检测图像：默认开启，不节流，逐检测帧输出）
        self.declare_parameter('enable_visualization', True)
        self.declare_parameter('visual_publish_width', 640)
        # <=0 表示每个检测帧都发布一张图
        self.declare_parameter('visual_publish_fps', 0.0)
        self.declare_parameter('visual_encoding', 'rgb8')
        self.declare_parameter('enable_fps_log', True)
        self.declare_parameter('enable_profile_log', True)
        
        # ReID 节流参数
        self.declare_parameter('reid_feature_interval', 2.0)  # ✅ 从 1.0s 提高到 2.0s
        self.declare_parameter('reid_try_interval', 0.5)      # ✅ 放宽找回间隔
        self.declare_parameter('reid_max_attempts', 3)        # ✅ 减少重试次数
        self.declare_parameter('reid_max_candidates', 2)      # ✅ 减少候选对比
        self.declare_parameter('fps_log_interval', 2.0)
        self.declare_parameter('draw_contour_only_targets', True)
        # 轮廓 / 跟踪框（PolygonStamped）统一发布频率，默认 10Hz
        self.declare_parameter('telemetry_publish_fps', 10.0)
        # ===== [FIX-3] 新增：找回窗口用尽后的二次重置周期系数 =====
        self.declare_parameter('reid_retry_reset_ratio', 0.5)
        # ===== [FIX-3] 新增：非锁定目标清理年龄参数化 =====
        self.declare_parameter('track_cleanup_age', 5.0)

        # ---------- 获取参数 ----------
        model_path = self._resolve_model_path(self.get_parameter('model_path').value)
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.max_processing_fps = self.get_parameter('max_processing_fps').value
        self.tracking_protection_time = self.get_parameter('tracking_protection_time').value
        self.reid_similarity_threshold = self.get_parameter('reid_similarity_threshold').value
        self.height_change_threshold = self.get_parameter('height_change_threshold').value
        self.lost_timeout_threshold = self.get_parameter('lost_timeout_threshold').value
        self.mask_alpha = self.get_parameter('mask_alpha').value
        self.draw_detections = self.get_parameter('draw_detections').value
        self.min_mask_points = self.get_parameter('min_mask_points').value
        self.enable_debug_log = self.get_parameter('enable_debug_log').value
        self.center_rect_width_ratio = self.get_parameter('center_rect_width_ratio').value
        self.center_confirm_time = self.get_parameter('center_confirm_time').value
        self.draw_center_zone = self.get_parameter('draw_center_zone').value
        self.input_topic = self.get_parameter('input_topic').value
        self.input_encoding = self.get_parameter('input_encoding').value
        
        self.enable_visualization = self.get_parameter('enable_visualization').value
        self.visual_publish_width = self.get_parameter('visual_publish_width').value
        self.visual_publish_fps = self.get_parameter('visual_publish_fps').value
        self.visual_encoding = self.get_parameter('visual_encoding').value
        self.enable_fps_log = self.get_parameter('enable_fps_log').value
        self.enable_profile_log = self.get_parameter('enable_profile_log').value
        self.reid_feature_interval = self.get_parameter('reid_feature_interval').value
        self.reid_try_interval = self.get_parameter('reid_try_interval').value
        self.reid_max_attempts = self.get_parameter('reid_max_attempts').value
        self.reid_max_candidates = self.get_parameter('reid_max_candidates').value
        self.fps_log_interval = self.get_parameter('fps_log_interval').value
        self.draw_contour_only_targets = self.get_parameter('draw_contour_only_targets').value
        self.telemetry_publish_fps = self.get_parameter('telemetry_publish_fps').value
        # ===== [FIX-3] =====
        self.reid_retry_reset_ratio = self.get_parameter('reid_retry_reset_ratio').value
        self.track_cleanup_age = self.get_parameter('track_cleanup_age').value

        self.min_process_interval = 1.0 / self.max_processing_fps if self.max_processing_fps > 0 else 0.0
        self.last_process_time = 0.0
        self._processing = False

        # 轮廓 / 跟踪框 发布限频（默认 10Hz，两个话题共用同一频率，各自独立计时）
        self.telemetry_pub_interval = 1.0 / self.telemetry_publish_fps if self.telemetry_publish_fps > 0 else 0.0
        self.last_track_box_pub_time = 0.0
        self.last_contour_pub_time = 0.0

        # 可视化发布限频（0 = 每帧都发，不节流）
        self.visual_publish_interval = 1.0 / self.visual_publish_fps if self.visual_publish_fps > 0 else 0.0
        self.last_visual_publish_time = 0.0

        # 统计与计时
        self._detect_frame_count = 0
        self._last_stat_time = time.time()
        self._reid_time = 0.0
        self._reid_count = 0

        self._prof = {
            'nv12_parse': 0.0, 'infer': 0.0, 'postprocess': 0.0,
            'nv12_to_bgr': 0.0, 'tracker': 0.0, 'contour': 0.0,
            'visual_draw': 0.0, 'visual_publish': 0.0, 'track_box_publish': 0.0,
        }
        self._prof_count = 0

        # ---------- 初始化模型与编码器 ----------
        self.model = YOLOv8_Seg(model_path, self.conf_threshold, self.iou_threshold)
        self.get_logger().info(f"yolo_model模型加载成功: {model_path}")

        reid_model_path = self._resolve_model_path(self.get_parameter('reid_model_path').value)
        self.reid_encoder = None
        try:
            self.reid_encoder = OSNetReID(reid_model_path)
            self.get_logger().info(f"ReID模型加载成功: {reid_model_path}")
        except Exception as e:
            self.get_logger().error(f"ReID模型加载失败: {e}")

        # ---------- 初始化跟踪器 ----------
        tracker_args = {
            'track_high_thresh': 0.25,
            'track_low_thresh': 0.1,
            'new_track_thresh': 0.25,
            'track_buffer': 10,
            'match_thresh': 0.68,
            'fuse_score': False,
            'gmc_method': 'sparseOptFlow',
            'proximity_thresh': 0.5,
            'appearance_thresh': 0.7,
            'with_reid': False,
            'reid_model_path': reid_model_path
        }
        self.tracker = BOTSORT(tracker_args)

        self.bridge = CvBridge()

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.image_sub = self.create_subscription(Image, self.input_topic, self.image_callback, image_qos)
        self._latest_msg = None
        self.process_timer = self.create_timer(0.1, self.on_timer)

        if self.enable_visualization:
            self.detect_pose_pub = self.create_publisher(Image, 'seg_track/combined', image_qos)
        else:
            self.detect_pose_pub = None
            
        self.track_box_pub = self.create_publisher(PolygonStamped, 'seg_track/track_box', 10)
        self.contour_pub = self.create_publisher(PolygonStamped, 'seg_track/contours', 10)

        # ---------- 跟踪状态 ----------
        self.tracked_persons: Dict[int, Dict] = {}
        self.current_tracking_id = None
        self.target_lost_time: Optional[float] = None
        self.last_lock_time: Optional[float] = None
        self.tracked_targets: Dict[int, Dict] = {}
        self.lock_fixed = False
        self.center_candidates: Dict[int, float] = {}
        self.person_class_id = 0

        self._reid_attempt = 0
        self._last_reid_try = 0.0
        self._frame_contours: Dict[int, np.ndarray] = {}

        self.center_half_width = 0.0
        self._nv12_width = 0
        self._nv12_height = 0
        self._nv12_valid = False

        # ✅ 核心：彻底关闭 Python 的垃圾回收，消灭毫秒级毛刺
        gc.disable()

        self.get_logger().info(
            f"Node initialized (input={self.input_topic}, encoding={self.input_encoding}, "
            f"Max FPS={'unlimited' if self.max_processing_fps <= 0 else f'{self.max_processing_fps:g}'}).")
        self.get_logger().info(
            f"Pub policy: combined={'every frame' if self.visual_publish_interval <= 0 else f'{self.visual_publish_fps:g}Hz'} "
            f"(enabled={self.enable_visualization}), "
            f"contours/track_box={self.telemetry_publish_fps:g}Hz, GC disabled.")

    # ---------- 提取轮廓 ----------
    def extract_contour(self, mc, corp, protos, original_h, original_w):
        x1_c = max(0, min(int(corp[0]), 160))
        y1_c = max(0, min(int(corp[1]), 160))
        x2_c = max(0, min(int(corp[2]), 160))
        y2_c = max(0, min(int(corp[3]), 160))
        region_h = y2_c - y1_c
        region_w = x2_c - x1_c
        if region_h <= 0 or region_w <= 0:
            return None
        corp_region = protos[y1_c:y2_c, x1_c:x2_c, :]
        if corp_region.size == 0:
            return None
        mask_region_flat = corp_region.reshape(-1, 32)
        mask_flat = np.dot(mask_region_flat, mc) > 0.0
        mask = mask_flat.reshape(region_h, region_w)
        mask_points = np.column_stack(np.where(mask))
        if len(mask_points) < self.min_mask_points:
            return None
        points_160_global = mask_points.copy()
        points_160_global[:, 1] += x1_c
        points_160_global[:, 0] += y1_c
        mask_160 = np.zeros((160, 160), dtype=np.uint8)
        mask_160[points_160_global[:, 0], points_160_global[:, 1]] = 255
        contours, _ = cv2.findContours(mask_160, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        scale_x = original_w / 160.0
        scale_y = original_h / 160.0
        contour_orig = (largest * np.array([scale_x, scale_y], dtype=np.float32)).astype(np.int32)
        contour_orig = contour_orig.reshape(-1, 2)
        if len(contour_orig) == 0:
            return None
        contour_orig[:, 0] = np.clip(contour_orig[:, 0], 0, original_w - 1)
        contour_orig[:, 1] = np.clip(contour_orig[:, 1], 0, original_h - 1)
        return contour_orig

    # ---------- 特征提取 ----------
    def extract_feature_from_bbox(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return np.zeros(512, dtype=np.float32)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(512, dtype=np.float32)
        if self.reid_encoder is None:
            return np.zeros(512, dtype=np.float32)
        try:
            t = time.time()
            feat = self.reid_encoder.extract_feature(crop)
            self._reid_time += (time.time() - t) * 1000.0
            self._reid_count += 1
            return feat
        except Exception as e:
            self.get_logger().warn(f"Feature extraction failed: {e}")
            return np.zeros(512, dtype=np.float32)

    # ---------- 特征低频更新 ----------
    def save_tracked_target(self, track_id: int, bbox: List[float], image: np.ndarray, timestamp: float):
        if track_id not in self.tracked_targets:
            feature = self.extract_feature_from_bbox(image, bbox)
            self.tracked_targets[track_id] = {
                'bbox': bbox,
                'feature': feature,
                'height_pixels': bbox[3] - bbox[1],
                'first_seen_time': timestamp,
                'last_seen_time': timestamp,
                'last_feature_time': timestamp,
                'lost_frames': 0,
                'is_recovered': False,
                'original_track_id': track_id,
            }
        else:
            target = self.tracked_targets[track_id]
            target['bbox'] = bbox
            target['last_seen_time'] = timestamp
            target['lost_frames'] = 0
            target['height_pixels'] = bbox[3] - bbox[1]
            # ✅ 当时间间隔大于 2 秒才更新特征，大幅降低 NPU 占用
            if timestamp - target.get('last_feature_time', 0.0) > self.reid_feature_interval:
                target['feature'] = self.extract_feature_from_bbox(image, bbox)
                target['last_feature_time'] = timestamp

    # ---------- 找回节流 ----------
    def try_recover_lost_target(self, current_tracks, image, timestamp):
        if self.current_tracking_id is None or self.current_tracking_id not in self.tracked_targets:
            return None
        target = self.tracked_targets[self.current_tracking_id]

        if self.target_lost_time is None:
            self.target_lost_time = timestamp
            self._reid_attempt = 0
            self._last_reid_try = 0.0
            self.get_logger().info(f"目标 {self.current_tracking_id} 丢失，开始ReID匹配找回")
            return None

        if timestamp - self._last_reid_try < self.reid_try_interval:
            return None

        # ===== [FIX-2] 尝试次数用尽后，周期性重置（而非永久放弃） =====
        if self._reid_attempt >= self.reid_max_attempts:
            reset_period = max(self.reid_try_interval, self.lost_timeout_threshold * self.reid_retry_reset_ratio)
            if timestamp - self._last_reid_try > reset_period:
                self._reid_attempt = 0
                self.get_logger().info(
                    f"目标 {self.current_tracking_id} ReID 尝试窗口用尽，重置计数再次尝试")
            else:
                return None
        # ===== [FIX-2] end =====

        self._last_reid_try = timestamp
        self._reid_attempt += 1

        candidate_tracks = [t for t in current_tracks if t['track_id'] != self.current_tracking_id]
        if not candidate_tracks:
            return None

        target_height = target['height_pixels']
        candidate_tracks.sort(
            key=lambda t: abs((t['bbox'][3] - t['bbox'][1]) - target_height)
        )
        candidate_tracks = candidate_tracks[:self.reid_max_candidates]

        target_feature = target['feature']
        best_match_id = None
        best_similarity = 0.0
        for track in candidate_tracks:
            bbox = track['bbox']
            x1, y1, x2, y2 = bbox
            candidate_height = y2 - y1
            height_ratio = candidate_height / target_height if target_height > 0 else 1.0
            height_change = abs(1.0 - height_ratio)
            if height_change > self.height_change_threshold:
                continue
            candidate_feature = self.extract_feature_from_bbox(image, bbox)
            if candidate_feature is not None and np.any(candidate_feature):
                similarity = np.dot(target_feature, candidate_feature) / (
                    np.linalg.norm(target_feature) * np.linalg.norm(candidate_feature) + 1e-8
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = track['track_id']

        if best_match_id is not None and best_similarity >= self.reid_similarity_threshold:
            self.get_logger().info(
                f"目标 {self.current_tracking_id} ReID找回成功，匹配ID {best_match_id}, 相似度 {best_similarity:.3f}")

            # ===== [FIX-1] 找回成功后迁移旧 ID 的历史记录到新 ID，避免轨迹断裂 =====
            old_id = self.current_tracking_id
            new_id = best_match_id
            if old_id in self.tracked_targets:
                old_rec = self.tracked_targets[old_id]
                if new_id in self.tracked_targets:
                    new_rec = self.tracked_targets[new_id]
                    # 保留更早的首次出现时间
                    new_rec['first_seen_time'] = min(
                        old_rec.get('first_seen_time', timestamp),
                        new_rec.get('first_seen_time', timestamp))
                    # 标记为找回，并继承原始 ID
                    new_rec['is_recovered'] = True
                    new_rec['original_track_id'] = old_rec.get('original_track_id', old_id)
                    # 若旧记录有更完整的丢失计数历史，合并（取较小值表示更活跃）
                    new_rec['lost_frames'] = min(
                        new_rec.get('lost_frames', 0), old_rec.get('lost_frames', 0))
                else:
                    # 新 ID 尚未建立记录，直接继承旧记录
                    self.tracked_targets[new_id] = old_rec
                    self.tracked_targets[new_id]['is_recovered'] = True
                    self.tracked_targets[new_id]['original_track_id'] = old_rec.get(
                        'original_track_id', old_id)
                # 删除旧 ID 记录，避免残留与 5s 后清理造成的特征基准漂移
                del self.tracked_targets[old_id]
            # ===== [FIX-1] end =====

            self.current_tracking_id = new_id
            self.target_lost_time = None
            self._reid_attempt = 0
            self._last_reid_try = 0.0
            if new_id in self.tracked_targets:
                self.tracked_targets[new_id]['lost_frames'] = 0
                self.tracked_targets[new_id]['last_seen_time'] = timestamp
            if new_id in self.tracked_persons:
                self.tracked_persons[new_id]['is_tracking'] = True
                self.tracked_persons[new_id]['last_seen_time'] = timestamp
            # 原 ID 在 tracked_persons 中的记录也清理（避免残留）
            if old_id in self.tracked_persons and old_id != new_id:
                old_person = self.tracked_persons.pop(old_id, None)
                if old_person is not None and new_id in self.tracked_persons:
                    # 继承首次出现时间
                    self.tracked_persons[new_id]['first_seen_time'] = min(
                        self.tracked_persons[new_id].get('first_seen_time', timestamp),
                        old_person.get('first_seen_time', timestamp))
            self.lock_fixed = True
            return new_id
        else:
            self.get_logger().debug(
                f"ReID找回失败 (attempt {self._reid_attempt}/{self.reid_max_attempts}), 最佳相似度 {best_similarity:.3f}")
            return None

    # ---------- NV12 解析 ----------
    def parse_nv12_message(self, msg):
        width = msg.width
        total_bytes = len(msg.data)

        if total_bytes % (width * 3 // 2) == 0:
            height = total_bytes // (width * 3 // 2)
        else:
            height = msg.height
            expected = width * height * 3 // 2
            if total_bytes != expected:
                rows = total_bytes // width
                nv12_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((rows, width))
                target_rows = height * 3 // 2
                if rows > target_rows:
                    nv12_array = nv12_array[:target_rows, :]
                return nv12_array, width, height

        nv12_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((height * 3 // 2, width))
        return nv12_array, width, height

    def nv12_to_bgr(self, nv12_array: np.ndarray, width: int, height: int) -> np.ndarray:
        expected_rows = height * 3 // 2
        if nv12_array.shape[0] != expected_rows:
            nv12_array = nv12_array[:expected_rows, :]
        return cv2.cvtColor(nv12_array, cv2.COLOR_YUV2BGR_NV12)

    # ---------- 主回调：来一帧处理一帧（不节流，检测跑满实际算力） ----------
    def image_callback(self, msg):
        self._latest_msg = msg

    def on_timer(self):
        msg = self._latest_msg
        self._latest_msg = None
        if msg is None:
            return
        self.process_frame(msg)

    def process_frame(self, msg):
        current_time = time.time()
        t_start = time.time()
        contour_published = False
        t_nv12_parse = t_infer = t_postprocess = t_nv12_to_bgr = 0.0
        t_tracker = t_contour = t_visual_draw = t_visual_publish = t_track_box_publish = 0.0

        try:
            # 1. NV12 解析
            ts = time.time()
            if msg.encoding.lower() != 'nv12':
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                original_h, original_w = cv_image.shape[:2]
                nv12_input = self.model.bgr2nv12(cv_image)
                outputs = self.model.c2numpy(self.model.forward(nv12_input))
                bgr_for_tracker = cv_image
            else:
                nv12_array, original_w, original_h = self.parse_nv12_message(msg)
                if not self._nv12_valid or self._nv12_width != original_w or self._nv12_height != original_h:
                    self._nv12_width = original_w
                    self._nv12_height = original_h
                    self._nv12_valid = True
                    self.get_logger().info(f"NV12 尺寸确认: {original_w}x{original_h}, data_len={len(msg.data)}")
            t_nv12_parse = (time.time() - ts) * 1000.0

            # 2. 推理
            ts = time.time()
            if msg.encoding.lower() == 'nv12':
                outputs = self.model.c2numpy(self.model.forward_nv12(nv12_array))
            t_infer = (time.time() - ts) * 1000.0

            # 3. 后处理
            ts = time.time()
            ids, scores, bboxes, corpes, mces, protos = self.model.postProcess(outputs)
            t_postprocess = (time.time() - ts) * 1000.0

            # 4. NV12 -> BGR
            ts = time.time()
            if msg.encoding.lower() == 'nv12':
                bgr_for_tracker = self.nv12_to_bgr(nv12_array, original_w, original_h)
            t_nv12_to_bgr = (time.time() - ts) * 1000.0

            self.center_half_width = original_w * self.center_rect_width_ratio / 2.0
            center_x = original_w / 2.0
            center_y = original_h / 2.0

            # 5. 构建检测列表
            detections = []
            mask_infos = []
            for i, (class_id, score, bbox, corp, mc) in enumerate(zip(ids, scores, bboxes, corpes, mces)):
                if score < self.conf_threshold:
                    continue
                if class_id != self.person_class_id:
                    continue
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w / 2, y1 + h / 2
                detections.append([cx, cy, w, h, float(score), int(class_id)])
                mask_infos.append((mc, corp, protos, (x1, y1, x2, y2)))

            # 6. 跟踪器更新
            ts = time.time()
            tracking_results = self.tracker.update(detections, bgr_for_tracker)
            t_tracker = (time.time() - ts) * 1000.0

            # 7. 构建当前帧映射
            current_track_ids = set()
            track_id_to_bbox = {}
            track_id_to_detidx = {}
            for track in tracking_results:
                x, y, w, h, track_id, score, cls, det_idx = track
                track_id = int(track_id)
                current_track_ids.add(track_id)
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                track_id_to_bbox[track_id] = (x1, y1, x2, y2)
                track_id_to_detidx[track_id] = det_idx
                if track_id not in self.tracked_persons:
                    self.tracked_persons[track_id] = {
                        'is_tracking': False,
                        'tracking_start_time': 0.0,
                        'last_seen_time': current_time,
                        'first_seen_time': current_time
                    }
                else:
                    self.tracked_persons[track_id]['last_seen_time'] = current_time

            self._frame_contours.clear()

            # 8. 中心区锁定
            if not self.lock_fixed:
                for track_id, bbox in track_id_to_bbox.items():
                    cx_obj = (bbox[0] + bbox[2]) / 2.0
                    if abs(cx_obj - center_x) <= self.center_half_width:
                        if track_id not in self.center_candidates:
                            self.center_candidates[track_id] = current_time
                        else:
                            duration = current_time - self.center_candidates[track_id]
                            if duration >= self.center_confirm_time:
                                self.current_tracking_id = track_id
                                self.last_lock_time = current_time
                                self.lock_fixed = True
                                if track_id in self.tracked_persons:
                                    self.tracked_persons[track_id]['is_tracking'] = True
                                    self.tracked_persons[track_id]['tracking_start_time'] = current_time
                                bbox = track_id_to_bbox[track_id]
                                # ✅ 直接提取特征并保存
                                self.save_tracked_target(track_id, bbox, bgr_for_tracker, current_time)
                                self.get_logger().info(
                                    f"自动锁定目标 ID {track_id} (在中心区持续 {duration:.1f}s)")
                                self.center_candidates.clear()
                                break
                    else:
                        if track_id in self.center_candidates:
                            del self.center_candidates[track_id]

            # 9. 锁定目标状态更新
            if self.current_tracking_id is not None:
                if self.current_tracking_id not in current_track_ids:
                    if self.target_lost_time is None:
                        self.target_lost_time = current_time
                        self._reid_attempt = 0
                        self._last_reid_try = 0.0
                        self.get_logger().warning(f"目标 {self.current_tracking_id} 丢失")
                    else:
                        current_tracks_dict = [{'track_id': tid, 'bbox': bbox}
                                               for tid, bbox in track_id_to_bbox.items()]
                        recovered_id = self.try_recover_lost_target(
                            current_tracks_dict, bgr_for_tracker, current_time)
                        if recovered_id is not None:
                            self.current_tracking_id = recovered_id
                            self.target_lost_time = None
                            self.lock_fixed = True
                            self.get_logger().info(f"目标找回成功，新ID {recovered_id}")
                        else:
                            if current_time - self.target_lost_time > self.lost_timeout_threshold:
                                self.get_logger().warning(
                                    f"目标 {self.current_tracking_id} 丢失超时，释放锁定")
                                # ===== [FIX-3] 超时释放：重置找回尝试计数，并标记旧目标失效但不删除 =====
                                lost_id = self.current_tracking_id
                                self.current_tracking_id = None
                                self.target_lost_time = None
                                self.last_lock_time = None
                                self.lock_fixed = False
                                self.center_candidates.clear()
                                self._reid_attempt = 0
                                self._last_reid_try = 0.0
                                # 保留 tracked_targets 供后续比对，但标记 is_recovered=False，
                                # 并刷新 last_feature_time，避免该人重新进入中心区时复用过期特征
                                if lost_id in self.tracked_targets:
                                    self.tracked_targets[lost_id]['is_recovered'] = False
                                    self.tracked_targets[lost_id]['last_feature_time'] = 0.0
                                # ===== [FIX-3] end =====
                else:
                    if self.target_lost_time is not None:
                        self.target_lost_time = None
                        self.get_logger().info(f"目标 {self.current_tracking_id} 重新出现")
                    bbox = track_id_to_bbox[self.current_tracking_id]
                    self.save_tracked_target(self.current_tracking_id, bbox, bgr_for_tracker, current_time)
                    det_idx = track_id_to_detidx.get(self.current_tracking_id, -1)
                    if det_idx != -1 and det_idx < len(mask_infos):
                        ts = time.time()
                        mc, corp, protos, _ = mask_infos[det_idx]
                        contour = self.extract_contour(mc, corp, protos, original_h, original_w)
                        t_contour += (time.time() - ts) * 1000.0
                        if contour is not None and len(contour) > 0:
                            self._frame_contours[det_idx] = contour
                            if rclpy.ok():
                                self.publish_contour(contour, self.current_tracking_id, msg.header)
                                contour_published = True

            # 没轮廓时发布空包保持频率
            if rclpy.ok() and not contour_published:
                empty_msg = PolygonStamped()
                empty_msg.header = msg.header
                empty_msg.header.frame_id = "track_none"
                self.contour_pub.publish(empty_msg)

            # 10. 清理过期目标
            self._cleanup_old_tracks(current_time, current_track_ids)

            # 11. 可视化（仅在开启时执行）
            if self.enable_visualization:
                if self.visual_publish_interval <= 0 or (current_time - self.last_visual_publish_time) >= self.visual_publish_interval:
                    self.last_visual_publish_time = current_time

                    ts = time.time()
                    display_image = self.draw_visualization(
                        bgr_for_tracker, tracking_results, mask_infos,
                        center_x, center_y, self.center_half_width, original_w, original_h,
                        self.center_candidates, self.center_confirm_time
                    )
                    t_visual_draw = (time.time() - ts) * 1000.0

                    if self.visual_publish_width > 0 and display_image.shape[1] > self.visual_publish_width:
                        h, w = display_image.shape[:2]
                        scale = float(self.visual_publish_width) / float(w)
                        new_size = (self.visual_publish_width, int(h * scale))
                        display_image = cv2.resize(display_image, new_size, interpolation=cv2.INTER_AREA)

                    ts = time.time()
                    if self.visual_encoding.lower() == 'rgb8':
                        display_out = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                        combined_msg = self.bridge.cv2_to_imgmsg(display_out, encoding='rgb8')
                    else:
                        combined_msg = self.bridge.cv2_to_imgmsg(display_image, encoding='bgr8')
                    combined_msg.header = msg.header
                    if rclpy.ok() and self.detect_pose_pub is not None:
                        self.detect_pose_pub.publish(combined_msg)
                    t_visual_publish = (time.time() - ts) * 1000.0

            # ✅ 12. 发布跟踪框 (限频 10Hz，与轮廓同一频率)
            ts = time.time()
            if rclpy.ok():
                self.publish_track_box(tracking_results, msg.header)
            t_track_box_publish = (time.time() - ts) * 1000.0

            # 13. 性能累计
            self._prof['nv12_parse'] += t_nv12_parse
            self._prof['infer'] += t_infer
            self._prof['postprocess'] += t_postprocess
            self._prof['nv12_to_bgr'] += t_nv12_to_bgr
            self._prof['tracker'] += t_tracker
            self._prof['contour'] += t_contour
            self._prof['visual_draw'] += t_visual_draw
            self._prof['visual_publish'] += t_visual_publish
            self._prof['track_box_publish'] += t_track_box_publish
            self._prof_count += 1

            # 14. fps 日志
            self._detect_frame_count += 1
            t_end = time.time()
            if self.enable_fps_log and (t_end - self._last_stat_time) >= self.fps_log_interval:
                elapsed = t_end - self._last_stat_time
                fps = self._detect_frame_count / elapsed
                total_ms = (t_end - t_start) * 1000.0
                reid_avg = self._reid_time / max(1, self._reid_count)
                reid_cnt = self._reid_count

                if self.enable_profile_log and self._prof_count > 0:
                    n = self._prof_count
                    avg = {k: v / n for k, v in self._prof.items()}
                    self.get_logger().info(
                        f"detect fps: {fps:.2f} | last: {total_ms:.1f} ms | "
                        f"avg(n={n}): "
                        f"nv12_parse={avg['nv12_parse']:.1f} "
                        f"infer={avg['infer']:.1f} "
                        f"post={avg['postprocess']:.1f} "
                        f"nv12_to_bgr={avg['nv12_to_bgr']:.1f} "
                        f"tracker={avg['tracker']:.1f} "
                        f"contour={avg['contour']:.1f} "
                        f"visual_draw={avg['visual_draw']:.1f} "
                        f"visual_pub={avg['visual_publish']:.1f} "
                        f"trackbox_pub={avg['track_box_publish']:.1f} ms | "
                        f"reid_avg={reid_avg:.1f}ms count={reid_cnt}")
                else:
                    self.get_logger().info(
                        f"detect fps: {fps:.2f}, last frame: {total_ms:.1f} ms, "
                        f"reid_avg={reid_avg:.1f}ms count={reid_cnt}")

                self._detect_frame_count = 0
                self._last_stat_time = t_end
                for k in self._prof:
                    self._prof[k] = 0.0
                self._prof_count = 0
                self._reid_time = 0.0
                self._reid_count = 0

        except Exception as e:
            if not rclpy.ok():
                return
            self.get_logger().error(f"Image processing error: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.last_process_time = time.time()
            self._processing = False

    # ---------- 发布轮廓 ----------
    def publish_contour(self, contour, track_id, header):
        polygon_msg = PolygonStamped()
        polygon_msg.header = header
        polygon_msg.header.frame_id = f"track_{track_id}"
        for pt in contour:
            p = Point32()
            if pt.ndim == 2 and pt.shape[0] == 1:
                p.x = float(pt[0][0])
                p.y = float(pt[0][1])
            else:
                p.x = float(pt[0])
                p.y = float(pt[1])
            p.z = 0.0
            polygon_msg.polygon.points.append(p)
        self.contour_pub.publish(polygon_msg)

    # ---------- 发布跟踪框 ----------
    def publish_track_box(self, tracks, header):
        polygon_msg = PolygonStamped()
        polygon_msg.header = header
        polygon_msg.header.frame_id = "camera_link"
        points = []
        if self.current_tracking_id is not None:
            target_track = None
            for track in tracks:
                if int(track[4]) == self.current_tracking_id:
                    target_track = track
                    break
            if target_track is not None:
                x, y, w, h, track_id, score, cls, det_idx = target_track
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                points.append(Point32(x=float(track_id), y=1.0, z=0.0))
                points.append(Point32(x=float(x1), y=float(y1), z=0.0))
                points.append(Point32(x=float(x2), y=float(y2), z=0.0))
            else:
                points.append(Point32(x=float(self.current_tracking_id), y=0.0, z=0.0))
                points.append(Point32(x=0.0, y=0.0, z=0.0))
                points.append(Point32(x=0.0, y=0.0, z=0.0))
        else:
            points = [Point32(x=0.0, y=0.0, z=0.0) for _ in range(3)]
        polygon_msg.polygon.points = points
        self.track_box_pub.publish(polygon_msg)

    # ---------- 可视化：复用轮廓缓存 ----------
    def draw_visualization(self, image, tracks, mask_infos, center_x, center_y, center_half_width,
                           image_w, image_h, center_candidates, center_confirm_time):
        display = image.copy()
        if self.draw_center_zone:
            x1 = int(max(0, center_x - center_half_width))
            x2 = int(min(image_w - 1, center_x + center_half_width))
            y1 = 0
            y2 = int(image_h - 1)

            overlay = display.copy()
            light_yellow = (150, 255, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), light_yellow, -1)
            cv2.addWeighted(overlay, 0.12, display, 0.88, 0, display)

            border_color = (120, 230, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), border_color, 1)
            cv2.line(display, (int(center_x), y1), (int(center_x), y2), (100, 100, 240), 1)
            cv2.putText(display, "Center Zone", (x1 + 5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 1)

        if self.draw_contour_only_targets:
            keep_det_idx = set()
            if self.current_tracking_id is not None:
                for track in tracks:
                    if int(track[4]) == self.current_tracking_id:
                        det_idx = track[7]
                        if det_idx != -1:
                            keep_det_idx.add(det_idx)
                        break
            for track in tracks:
                tid = int(track[4])
                if tid in center_candidates:
                    det_idx = track[7]
                    if det_idx != -1:
                        keep_det_idx.add(det_idx)
        else:
            keep_det_idx = set(range(len(mask_infos)))

        for det_idx, (mc, corp, protos, bbox) in enumerate(mask_infos):
            if det_idx not in keep_det_idx:
                continue
            contour = self._frame_contours.get(det_idx)
            if contour is None:
                contour = self.extract_contour(mc, corp, protos, image.shape[0], image.shape[1])
                if contour is not None:
                    self._frame_contours[det_idx] = contour
            if contour is not None and len(contour) > 0:
                try:
                    cv2.polylines(display, [contour], True, (0, 255, 0), 1)
                except cv2.error:
                    pass

        for track in tracks:
            x, y, w, h, track_id, score, cls, det_idx = track
            track_id = int(track_id)
            if det_idx != -1 and det_idx < len(mask_infos):
                _, _, _, bbox = mask_infos[det_idx]
                x1, y1, x2, y2 = bbox
            else:
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            is_tracking = (track_id == self.current_tracking_id)
            color = (0, 0, 255) if is_tracking else (255, 0, 0)
            if not is_tracking and track_id in center_candidates:
                elapsed = time.time() - center_candidates[track_id]
                remaining = max(0, center_confirm_time - elapsed)
                color = (0, 255, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"Candidate {elapsed:.1f}s", (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(display, f"Remain {remaining:.1f}s", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            else:
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"ID:{track_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return display

    def _cleanup_old_tracks(self, current_time, current_track_ids):
        # ===== [FIX-3] 清理年龄参数化 =====
        max_track_age = self.track_cleanup_age
        for track_id in list(self.tracked_persons.keys()):
            if track_id == self.current_tracking_id:
                continue
            if track_id not in current_track_ids:
                last_seen = self.tracked_persons[track_id]['last_seen_time']
                if current_time - last_seen > max_track_age:
                    del self.tracked_persons[track_id]
                    if track_id in self.tracked_targets:
                        del self.tracked_targets[track_id]
                    if track_id in self.center_candidates:
                        del self.center_candidates[track_id]


def main(args=None):
    rclpy.init(args=args)
    try:
        node = YOLOv8SegROSNode()
        
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
            node.destroy_node()
    except Exception as e:
        print(f"Node initialization failed: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
