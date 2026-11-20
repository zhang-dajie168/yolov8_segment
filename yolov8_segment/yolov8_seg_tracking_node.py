#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, Point32
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
import copy

# 导入分割模型和跟踪器
from .yolov8_segment_model import YOLOv8_Seg, rdk_colors
from .BOTSort_rdk import BOTSORT, OSNetReID


class Yolov8SegTrackNode(Node):
    def __init__(self):
        super().__init__('yolov8_seg_track_node')

        # ---------- 参数声明 ----------
        self.declare_parameter('model_path', '')
        self.declare_parameter('conf_threshold', 0.3)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('max_processing_fps', 15)
        self.declare_parameter('tracking_protection_time', 5.0)        # 锁定后保护时间（防止切换）
        self.declare_parameter('reid_similarity_threshold', 0.8)
        self.declare_parameter('height_change_threshold', 0.15)
        self.declare_parameter('lost_timeout_threshold', 10.0)        # 丢失超时（秒）
        self.declare_parameter('reid_model_path', 'osnet_64x128_nv12.bin')
        self.declare_parameter('mask_alpha', 0.3)
        self.declare_parameter('draw_detections', False)
        self.declare_parameter('min_mask_points', 10)
        self.declare_parameter('enable_debug_log', False)
        # 新增：中心选择参数
        self.declare_parameter('center_radius_ratio', 0.3)             # 中心区域半径比例（相对于图像短边）

        # ---------- 获取参数 ----------
        model_path = self.get_parameter('model_path').value
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
        self.center_radius_ratio = self.get_parameter('center_radius_ratio').value

        self.min_process_interval = 1.0 / self.max_processing_fps
        self.last_process_time = time.time()

        # ---------- 初始化模型 ----------
        self.model = YOLOv8_Seg(model_path, self.conf_threshold, self.iou_threshold)

        # ---------- 初始化ReID ----------
        reid_model_path = self.get_parameter('reid_model_path').value
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
            'with_reid': False,          # 我们自己在外部做ReID，可开启
            'reid_model_path': reid_model_path
        }
        self.tracker = BOTSORT(tracker_args)

        # ---------- CV Bridge ----------
        self.bridge = CvBridge()

        # ---------- 发布与订阅 ----------
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.detect_pose_pub = self.create_publisher(Image, 'seg_track/combined', 10)      # 可视化图像
        self.track_info_pub = self.create_publisher(PolygonStamped, 'seg_track/track_info', 10)  # 跟踪状态
        self.contour_pub = self.create_publisher(PolygonStamped, 'seg_track/contours', 10)       # 轮廓发布

        # ---------- 跟踪状态变量 ----------
        self.tracked_persons: Dict[int, Dict] = {}          # 所有跟踪目标信息
        self.current_tracking_id = None                     # 当前锁定的目标ID
        self.target_lost_time: Optional[float] = None       # 当前锁定目标丢失开始时间
        self.last_lock_time: Optional[float] = None         # 锁定开始时间（用于保护期）
        self.tracked_targets: Dict[int, Dict] = {}          # 存储目标特征等信息（用于ReID）

        # 类别ID：假设person为0
        self.person_class_id = 0

        self.get_logger().info("YOLOv8 Segmentation Tracking Node initialized with center-based auto-lock.")

    # ---------- 辅助函数：提取轮廓 ----------
    def extract_contour(self, mc, corp, protos, original_h, original_w):
        x1_c, y1_c, x2_c, y2_c = corp
        region_h = y2_c - y1_c
        region_w = x2_c - x1_c
        if region_h <= 0 or region_w <= 0:
            return None
        corp_region = protos[y1_c:y2_c, x1_c:x2_c, :]
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
        try:
            if self.reid_encoder is not None:
                return self.reid_encoder.extract_feature(crop)
            else:
                return np.zeros(512, dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"Feature extraction failed: {e}")
            return np.zeros(512, dtype=np.float32)

    # ---------- 保存目标特征 ----------
    def save_tracked_target(self, track_id: int, bbox: List[float], image: np.ndarray, timestamp: float):
        if track_id not in self.tracked_targets:
            feature = self.extract_feature_from_bbox(image, bbox)
            height_pixels = bbox[3] - bbox[1]
            self.tracked_targets[track_id] = {
                'bbox': bbox,
                'feature': feature,
                'height_pixels': height_pixels,
                'first_seen_time': timestamp,
                'last_seen_time': timestamp,
                'lost_frames': 0,
                'is_recovered': False,
                'original_track_id': track_id
            }
        else:
            target = self.tracked_targets[track_id]
            target['bbox'] = bbox
            target['last_seen_time'] = timestamp
            target['lost_frames'] = 0
            # 更新特征（可每帧更新）
            feature = self.extract_feature_from_bbox(image, bbox)
            target['feature'] = feature
            target['height_pixels'] = bbox[3] - bbox[1]

    # ---------- ReID找回 ----------
    def try_recover_lost_target(self, current_tracks: List[Dict], image: np.ndarray, timestamp: float) -> Optional[int]:
        if self.current_tracking_id is None or self.current_tracking_id not in self.tracked_targets:
            return None
        target = self.tracked_targets[self.current_tracking_id]
        if self.target_lost_time is None:
            self.target_lost_time = timestamp
            self.get_logger().info(f"目标 {self.current_tracking_id} 丢失，开始ReID匹配找回")

        # 候选目标：所有当前跟踪中且不是当前锁定ID的（因为锁定ID可能已丢失）
        candidate_tracks = []
        for track in current_tracks:
            track_id = track['track_id']
            if track_id != self.current_tracking_id:
                candidate_tracks.append(track)

        if not candidate_tracks:
            return None

        target_feature = target['feature']
        target_height = target['height_pixels']
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
            self.get_logger().info(f"目标 {self.current_tracking_id} ReID找回成功，匹配ID {best_match_id}, 相似度 {best_similarity:.3f}")
            # 更新锁定ID为新ID
            old_id = self.current_tracking_id
            self.current_tracking_id = best_match_id
            # 清除丢失时间
            self.target_lost_time = None
            # 更新目标信息
            if best_match_id in self.tracked_targets:
                self.tracked_targets[best_match_id]['lost_frames'] = 0
                self.tracked_targets[best_match_id]['last_seen_time'] = timestamp
            # 更新跟踪状态
            if best_match_id in self.tracked_persons:
                self.tracked_persons[best_match_id]['is_tracking'] = True
                self.tracked_persons[best_match_id]['last_seen_time'] = timestamp
            return best_match_id
        else:
            self.get_logger().debug(f"ReID找回失败，最佳相似度 {best_similarity:.3f}")
            return None

    # ---------- 主回调 ----------
    def image_callback(self, msg):
        current_time = time.time()
        if current_time - self.last_process_time < self.min_process_interval:
            return
        self.last_process_time = current_time

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            original_h, original_w = cv_image.shape[:2]

            # 1. 分割推理
            input_tensor = self.model.bgr2nv12(cv_image)
            outputs = self.model.c2numpy(self.model.forward(input_tensor))
            ids, scores, bboxes, corpes, mces, protos = self.model.postProcess(outputs)

            # 2. 构建检测列表和mask信息（只保留person类别）
            detections = []          # [cx, cy, w, h, score, class_id]
            mask_infos = []          # (mc, corp, protos, bbox_xyxy)
            person_bboxes = []       # 用于中心选择

            for i, (class_id, score, bbox, corp, mc) in enumerate(zip(ids, scores, bboxes, corpes, mces)):
                if score < self.conf_threshold:
                    continue
                if class_id != self.person_class_id:
                    continue   # 只跟踪person
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w/2, y1 + h/2
                detections.append([cx, cy, w, h, float(score), int(class_id)])
                mask_infos.append((mc, corp, protos, (x1, y1, x2, y2)))
                person_bboxes.append((x1, y1, x2, y2, i))   # 保存索引i

            # 3. 更新跟踪器
            tracking_results = self.tracker.update(detections, cv_image)
            # tracking_results 每个元素: [x, y, w, h, track_id, score, cls, det_idx]

            # 4. 处理跟踪结果，更新状态
            current_track_ids = set()
            for track in tracking_results:
                x, y, w, h, track_id, score, cls, det_idx = track
                track_id = int(track_id)
                current_track_ids.add(track_id)
                bbox_xyxy = (int(x), int(y), int(x+w), int(y+h))

                # 更新tracked_persons
                if track_id not in self.tracked_persons:
                    self.tracked_persons[track_id] = {
                        'is_tracking': False,
                        'tracking_start_time': 0.0,
                        'last_seen_time': current_time,
                        'first_seen_time': current_time
                    }
                else:
                    self.tracked_persons[track_id]['last_seen_time'] = current_time

                # 如果当前锁定此ID，提取并保存特征
                if self.current_tracking_id == track_id:
                    self.save_tracked_target(track_id, bbox_xyxy, cv_image, current_time)
                    # 提取轮廓并发布
                    if det_idx != -1 and det_idx < len(mask_infos):
                        mc, corp, protos, _ = mask_infos[det_idx]
                        contour = self.extract_contour(mc, corp, protos, original_h, original_w)
                        if contour is not None:
                            self.publish_contour(contour, track_id, msg.header)

            # 5. 锁定逻辑：如果没有锁定目标，选择中心最近的人
            if self.current_tracking_id is None:
                if len(person_bboxes) > 0:
                    # 计算图像中心
                    center_x = original_w / 2.0
                    center_y = original_h / 2.0
                    # 选择距离最近的人
                    best_idx = -1
                    best_dist = float('inf')
                    for idx, (x1, y1, x2, y2, det_idx) in enumerate(person_bboxes):
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = det_idx
                    if best_idx != -1:
                        # 获取该检测对应的跟踪ID（通过det_idx匹配）
                        # 在tracking_results中查找det_idx == best_idx
                        locked_track_id = None
                        for track in tracking_results:
                            if track[7] == best_idx:   # det_idx
                                locked_track_id = track[4]
                                break
                        if locked_track_id is not None:
                            self.current_tracking_id = int(locked_track_id)
                            self.last_lock_time = current_time
                            # 标记该目标为跟踪中
                            if self.current_tracking_id in self.tracked_persons:
                                self.tracked_persons[self.current_tracking_id]['is_tracking'] = True
                                self.tracked_persons[self.current_tracking_id]['tracking_start_time'] = current_time
                            self.get_logger().info(f"自动锁定目标 ID {self.current_tracking_id} (中心最近)")

            # 6. 如果当前有锁定目标，检查是否还在跟踪结果中
            if self.current_tracking_id is not None:
                if self.current_tracking_id not in current_track_ids:
                    # 目标丢失
                    if self.target_lost_time is None:
                        self.target_lost_time = current_time
                        self.get_logger().warning(f"目标 {self.current_tracking_id} 丢失")
                    else:
                        # 尝试ReID找回
                        recovered_id = self.try_recover_lost_target(tracking_results, cv_image, current_time)
                        if recovered_id is not None:
                            # 找回成功，更新锁定ID
                            self.current_tracking_id = recovered_id
                            self.target_lost_time = None
                            self.get_logger().info(f"目标找回成功，新ID {recovered_id}")
                        else:
                            # 检查是否超时
                            if current_time - self.target_lost_time > self.lost_timeout_threshold:
                                self.get_logger().warning(f"目标 {self.current_tracking_id} 丢失超时，释放锁定")
                                self.current_tracking_id = None
                                self.target_lost_time = None
                                self.last_lock_time = None
                else:
                    # 目标重新出现，重置丢失时间
                    if self.target_lost_time is not None:
                        self.target_lost_time = None
                        self.get_logger().info(f"目标 {self.current_tracking_id} 重新出现")

            # 7. 清理过期目标（非锁定）
            self._cleanup_old_tracks(current_time, current_track_ids)

            # 8. 发布可视化图像
            display_image = self.draw_visualization(cv_image, tracking_results, mask_infos)
            combined_msg = self.bridge.cv2_to_imgmsg(display_image, encoding='bgr8')
            combined_msg.header = msg.header
            self.detect_pose_pub.publish(combined_msg)

            # 9. 发布跟踪状态信息
            self.publish_track_info(tracking_results, msg.header)

        except Exception as e:
            self.get_logger().error(f"Image processing error: {str(e)}")

    # ---------- 发布轮廓 ----------
    def publish_contour(self, contour, track_id, header):
        polygon_msg = PolygonStamped()
        polygon_msg.header = header
        polygon_msg.header.frame_id = f"track_{track_id}"
        for pt in contour:
            p = Point32()
            p.x = float(pt[0][0]) if pt.ndim == 2 else float(pt[0])
            p.y = float(pt[0][1]) if pt.ndim == 2 else float(pt[1])
            p.z = 0.0
            polygon_msg.polygon.points.append(p)
        self.contour_pub.publish(polygon_msg)

    # ---------- 发布跟踪状态 ----------
    def publish_track_info(self, tracks, header):
        polygon_msg = PolygonStamped()
        polygon_msg.header = header
        polygon_msg.header.frame_id = "track_info"
        if self.current_tracking_id is not None:
            # 检查是否在当前帧出现
            in_frame = any(t[4] == self.current_tracking_id for t in tracks)
            p = Point32()
            p.x = float(self.current_tracking_id)
            p.y = 1.0 if in_frame else 0.0
            p.z = 0.0
            polygon_msg.polygon.points = [p]
        else:
            polygon_msg.polygon.points = [Point32(x=0.0, y=0.0, z=0.0)]
        self.track_info_pub.publish(polygon_msg)

    # ---------- 可视化 ----------
    def draw_visualization(self, image, tracks, mask_infos):
        display = image.copy()
        # 绘制所有检测的轮廓（可选）
        for mc, corp, protos, bbox in mask_infos:
            contour = self.extract_contour(mc, corp, protos, image.shape[0], image.shape[1])
            if contour is not None:
                cv2.polylines(display, [contour], True, (0, 255, 0), 1)
        # 绘制跟踪框
        for track in tracks:
            x, y, w, h, track_id, score, cls, det_idx = track
            x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
            is_tracking = (track_id == self.current_tracking_id)
            color = (0, 0, 255) if is_tracking else (255, 0, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id}"
            cv2.putText(display, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return display

    # ---------- 清理旧目标 ----------
    def _cleanup_old_tracks(self, current_time, current_track_ids):
        max_track_age = 5.0
        for track_id in list(self.tracked_persons.keys()):
            if track_id == self.current_tracking_id:
                continue
            if track_id not in current_track_ids:
                last_seen = self.tracked_persons[track_id]['last_seen_time']
                if current_time - last_seen > max_track_age:
                    del self.tracked_persons[track_id]
                    if track_id in self.tracked_targets:
                        del self.tracked_targets[track_id]


def main(args=None):
    rclpy.init(args=args)
    try:
        node = Yolov8SegTrackNode()
        executor = MultiThreadedExecutor()
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