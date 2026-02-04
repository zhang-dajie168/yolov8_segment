#!/user/bin/env python
# Copyright (c) 2024，WuChao D-Robotics.
# Licensed under the Apache License, Version 2.0 (the "License");

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, Point32
from cv_bridge import CvBridge
import cv2
import numpy as np

# 导入模型模块
from .yolov8_segment_model import YOLOv8_Seg, rdk_colors, compute_and_process_mask

class YOLOv8SegROSNode(Node):
    def __init__(self):
        super().__init__('yolov8_seg_ros_node')

        # 声明参数
        self.declare_parameter('model_path',
                               '/home/sunrise/Ebike_Human_Follower/src/yolov8_segment/models/yolov8n_instance_seg_bayese_640x640_nv12_modified2.0.bin')
        self.declare_parameter('conf_threshold', 0.6)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('input_image_topic', '/camera/color/image_raw')
        self.declare_parameter('max_fps', 15.0)
        self.declare_parameter('x_min_ratio', 1/6)  # x坐标过滤的最小比例
        self.declare_parameter('x_max_ratio', 5/6)  # x坐标过滤的最大比例
        self.declare_parameter('mask_alpha', 0.3)   # 掩码透明度
        self.declare_parameter('draw_detections', False)  # 是否绘制检测框
        self.declare_parameter('draw_min_rectangles', True)  # 是否绘制最小矩形
        self.declare_parameter('create_mask_image', True)  # 是否创建掩码图像
        self.declare_parameter('min_mask_points', 10)  # 最小掩码点数阈值
        self.declare_parameter('enable_debug_log', False)  # 是否启用调试日志
        self.declare_parameter('publish_zero_when_no_target', True)  # 没有目标时是否发布零点
        self.declare_parameter('publish_lowest_target', True)  # 是否只发布Y最大的目标（中心点最低）

        # 获取参数
        model_path = self.get_parameter('model_path').value
        conf_threshold = self.get_parameter('conf_threshold').value
        iou_threshold = self.get_parameter('iou_threshold').value
        input_topic = self.get_parameter('input_image_topic').value
        max_fps = self.get_parameter('max_fps').value
        self.x_min_ratio = self.get_parameter('x_min_ratio').value
        self.x_max_ratio = self.get_parameter('x_max_ratio').value
        self.mask_alpha = self.get_parameter('mask_alpha').value
        self.draw_detections = self.get_parameter('draw_detections').value
        self.draw_min_rectangles = self.get_parameter('draw_min_rectangles').value
        self.create_mask_image = self.get_parameter('create_mask_image').value
        self.min_mask_points = self.get_parameter('min_mask_points').value
        self.enable_debug_log = self.get_parameter('enable_debug_log').value
        self.publish_zero_when_no_target = self.get_parameter('publish_zero_when_no_target').value
        self.publish_lowest_target = self.get_parameter('publish_lowest_target').value

        self.get_logger().info(f"Model path: {model_path}")
        self.get_logger().info(f"Confidence threshold: {conf_threshold}")
        self.get_logger().info(f"IoU threshold: {iou_threshold}")
        self.get_logger().info(f"Input topic: {input_topic}")
        self.get_logger().info(f"Max FPS: {max_fps}")
        self.get_logger().info(f"X filter range: {self.x_min_ratio:.2f} - {self.x_max_ratio:.2f}")
        self.get_logger().info(f"Mask alpha: {self.mask_alpha}")
        self.get_logger().info(f"Draw detections: {self.draw_detections}")
        self.get_logger().info(f"Draw min rectangles: {self.draw_min_rectangles}")
        self.get_logger().info(f"Create mask image: {self.create_mask_image}")
        self.get_logger().info(f"Min mask points: {self.min_mask_points}")
        self.get_logger().info(f"Publish zero when no target: {self.publish_zero_when_no_target}")
        self.get_logger().info(f"Publish lowest target only: {self.publish_lowest_target}")

        # 初始化模型
        self.model = YOLOv8_Seg(model_path, conf_threshold, iou_threshold)

        # 初始化CV bridge
        self.bridge = CvBridge()

        # 创建订阅和发布
        self.image_sub = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10
        )

        # 创建发布者：combined图像和矩形顶点
        self.combined_pub = self.create_publisher(Image, 'segmentation/combined', 10)
        self.rect_pub = self.create_publisher(PolygonStamped, 'segmentation/rectangles', 10)

        # FPS控制
        self.min_interval = 1.0 / max_fps
        self.last_process_time = 0

        # 预计算过滤阈值
        self.x_min_threshold_160 = int(160 * self.x_min_ratio)
        self.x_max_threshold_160 = int(160 * self.x_max_ratio)
        
        # 缓存变量
        self.class_colors = {}
        
        # 统计信息
        self.frame_count = 0
        self.detection_count = 0
        self.processing_times = []
        
        # 零点矩形（用于没有目标时发布）
        self.zero_rectangle = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)

        self.get_logger().info("YOLOv8 Segmentation ROS Node initialized successfully")

    def publish_rectangle(self, box_points, header, class_id, score, is_zero_rectangle=False):
        """发布矩形顶点"""
        polygon_msg = PolygonStamped()
        polygon_msg.header = header
        
        # 添加类别ID和置信度到消息中
        if is_zero_rectangle:
            polygon_msg.header.frame_id = "no_target"
        else:
            polygon_msg.header.frame_id = f"class_{class_id}_score_{score:.2f}"
        
        # 预分配列表以提高性能
        polygon_msg.polygon.points = [Point32() for _ in range(4)]
        for i, point in enumerate(box_points):
            polygon_msg.polygon.points[i].x = float(point[0])
            polygon_msg.polygon.points[i].y = float(point[1])
            polygon_msg.polygon.points[i].z = 0.0
        
        self.rect_pub.publish(polygon_msg)
        if self.enable_debug_log:
            if is_zero_rectangle:
                self.get_logger().debug("Published zero rectangle (no target detected)")
            else:
                self.get_logger().debug(f"Published rectangle for class {class_id} with score {score:.2f}")

    def draw_detection_fast(self, img, bbox, score, class_id):
        """快速绘制检测框（简化版）"""
        if not self.draw_detections:
            return
        
        x1, y1, x2, y2 = bbox
        
        # 使用缓存的颜色
        if class_id not in self.class_colors:
            self.class_colors[class_id] = rdk_colors[class_id % len(rdk_colors)]
        
        # 只画矩形框，不画标签
        cv2.rectangle(img, (x1, y1), (x2, y2), self.class_colors[class_id], 2)

    def draw_min_rectangle_fast(self, img, box_points, is_selected=False):
        """快速绘制最小矩形"""
        if not self.draw_min_rectangles:
            return
        
        # 如果是最低目标，用红色标出
        color = (0, 255, 0)  # 绿色
        if is_selected:
            color = (0, 0, 255)  # 红色
        
        # 直接画多边形，比drawContours更快
        cv2.polylines(img, [box_points], True, color, 2)
        
        # 如果是选中的目标，在中心点画一个圆
        if is_selected:
            # 计算中心点
            center_x = np.mean(box_points[:, 0])
            center_y = np.mean(box_points[:, 1])
            cv2.circle(img, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

    def compute_mask_fast(self, protos, mc, corp_region):
        """快速计算mask"""
        mask_region_flat = corp_region.reshape(-1, 32)
        mask_flat = np.dot(mask_region_flat, mc) > 0.0
        mask = mask_flat.reshape(corp_region.shape[0], corp_region.shape[1])
        mask_points = np.column_stack(np.where(mask))
        return mask, mask_points

    def filter_and_transform_points_fast(self, mask_points, x1_corp, y1_corp, scale_x, scale_y, 
                                       original_w, original_h):
        """快速过滤和转换点坐标"""
        if len(mask_points) == 0:
            return []
        
        # 转换到160坐标系
        points_160 = mask_points.copy()
        points_160[:, 1] += x1_corp  # x坐标
        points_160[:, 0] += y1_corp  # y坐标
        
        # 过滤x坐标
        x_filter_mask = (points_160[:, 1] >= self.x_min_threshold_160) & \
                       (points_160[:, 1] <= self.x_max_threshold_160)
        filtered_points_160 = points_160[x_filter_mask]
        
        if len(filtered_points_160) == 0:
            return []
        
        # 向量化转换到原始坐标系
        points_original = np.empty((len(filtered_points_160), 2), dtype=np.int32)
        points_original[:, 0] = (filtered_points_160[:, 1] * scale_x).astype(np.int32)  # x
        points_original[:, 1] = (filtered_points_160[:, 0] * scale_y).astype(np.int32)  # y
        
        # 边界检查
        valid_mask = (points_original[:, 0] >= 0) & (points_original[:, 0] < original_w) & \
                    (points_original[:, 1] >= 0) & (points_original[:, 1] < original_h)
        
        return points_original[valid_mask].tolist()

    def find_lowest_target(self, all_targets_info):
        """找到Y坐标最大的目标（中心点最低）"""
        if not all_targets_info:
            return None
        
        lowest_target = None
        max_center_y = -1
        
        for target_info in all_targets_info:
            box_points = target_info['box_points']
            # 计算中心点Y坐标
            center_y = np.mean(box_points[:, 1])
            
            # 找到Y最大的目标（图像坐标系中Y越大，位置越低）
            if center_y > max_center_y:
                max_center_y = center_y
                lowest_target = target_info
        
        return lowest_target

    def image_callback(self, msg):
        """图像回调函数"""
        import time
        start_time = time.time()
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_process_time < self.min_interval:
            return

        self.last_process_time = current_time
        self.frame_count += 1

        try:
            # 转换图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if cv_image is None:
                self.get_logger().error("Failed to convert image")
                return

            # 保存原始图像大小
            original_h, original_w = cv_image.shape[:2]
            
            # 预计算缩放因子
            scale_x = original_w / 160.0
            scale_y = original_h / 160.0
            
            # 创建原始图像的副本用于绘制
            display_image = cv_image.copy()

            # 准备输入数据
            input_tensor = self.model.bgr2nv12(cv_image)

            # 推理
            outputs = self.model.c2numpy(self.model.forward(input_tensor))

            # 后处理
            ids, scores, bboxes, corpes, mces, protos = self.model.postProcess(outputs)

            # 更新统计信息
            detection_count = len(ids)
            self.detection_count += detection_count
            
            inference_time = time.time() - start_time

            # 创建掩码图像（如果需要）
            if self.create_mask_image:
                mask_image = np.zeros((original_h, original_w, 3), dtype=np.uint8)
            else:
                mask_image = None

            # 收集所有有效目标的信息
            all_targets_info = []
            
            # 处理每个检测结果
            for i, (class_id, score, bbox, corp, mc) in enumerate(zip(ids, scores, bboxes, corpes, mces)):
                # 跳过低置信度检测
                if score < 0.5:
                    continue
                    
                x1, y1, x2, y2 = bbox
                x1_corp, y1_corp, x2_corp, y2_corp = corp
                
                # 边界检查和裁剪
                x1_corp = max(0, min(x1_corp, 159))
                y1_corp = max(0, min(y1_corp, 159))
                x2_corp = max(x1_corp + 1, min(x2_corp, 160))
                y2_corp = max(y1_corp + 1, min(y2_corp, 160))
                
                if x2_corp <= x1_corp or y2_corp <= y1_corp:
                    continue
                
                # 快速绘制检测框
                self.draw_detection_fast(display_image, bbox, score, class_id)
                
                # 计算mask区域大小
                region_h = y2_corp - y1_corp
                region_w = x2_corp - x1_corp
                
                if region_h <= 0 or region_w <= 0:
                    continue
                
                # 获取mask区域
                corp_region = protos[y1_corp:y2_corp, x1_corp:x2_corp, :]
                
                # 计算mask
                mask, mask_points = self.compute_mask_fast(protos, mc, corp_region)
                
                if len(mask_points) < self.min_mask_points:
                    continue
                
                # 过滤和转换点坐标
                filtered_points = self.filter_and_transform_points_fast(
                    mask_points, x1_corp, y1_corp, scale_x, scale_y, original_w, original_h
                )
                
                if len(filtered_points) >= 4:
                    filtered_points_np = np.array(filtered_points, dtype=np.int32)
                    
                    # 计算最小外接矩形
                    try:
                        rect = cv2.minAreaRect(filtered_points_np)
                        box_points = cv2.boxPoints(rect)
                        box_points = np.int32(box_points)
                        
                        # 保存目标信息
                        target_info = {
                            'box_points': box_points,
                            'class_id': class_id,
                            'score': score,
                            'bbox': bbox,
                            'filtered_points': filtered_points
                        }
                        all_targets_info.append(target_info)
                        
                    except Exception as e:
                        if self.enable_debug_log:
                            self.get_logger().debug(f"Failed to compute min area rect: {e}")
                        continue

            # 根据策略选择发布的目标
            target_to_publish = None
            is_zero_rectangle = False
            
            if not all_targets_info:
                # 没有检测到目标
                if self.publish_zero_when_no_target:
                    # 发布零点矩形
                    target_to_publish = {
                        'box_points': self.zero_rectangle,
                        'class_id': -1,
                        'score': 0.0
                    }
                    is_zero_rectangle = True
                    if self.enable_debug_log:
                        self.get_logger().debug("No targets detected, publishing zero rectangle")
            else:
                # 检测到至少一个目标
                if self.publish_lowest_target:
                    # 只发布Y最大的目标（中心点最低）
                    target_to_publish = self.find_lowest_target(all_targets_info)
                    if self.enable_debug_log:
                        self.get_logger().debug(f"Found {len(all_targets_info)} targets, selected lowest one")
                else:
                    # 发布所有目标（原始逻辑）
                    # 为了保持兼容性，这里暂时只发布第一个目标
                    target_to_publish = all_targets_info[0]
            
            # 绘制所有检测到的矩形，但只突出显示要发布的目标
            for i, target_info in enumerate(all_targets_info):
                is_selected = (target_to_publish is not None and 
                              np.array_equal(target_info['box_points'], target_to_publish['box_points']))
                self.draw_min_rectangle_fast(display_image, target_info['box_points'], is_selected)
                
                # 在掩码图像上绘制点
                if mask_image is not None and target_info.get('filtered_points'):
                    color = rdk_colors[target_info['class_id'] % len(rdk_colors)]
                    # 批量绘制点
                    for x, y in target_info['filtered_points']:
                        mask_image[y, x] = color
            
            # 发布选中的目标矩形
            if target_to_publish is not None:
                self.publish_rectangle(
                    target_to_publish['box_points'], 
                    msg.header, 
                    target_to_publish['class_id'], 
                    target_to_publish['score'],
                    is_zero_rectangle
                )
            
            # 创建组合图像
            if mask_image is not None and len(all_targets_info) > 0:
                # 使用更高效的图像混合
                combined_image = cv2.addWeighted(display_image, 1.0, mask_image, self.mask_alpha, 0)
            else:
                combined_image = display_image

            # 发布combined结果
            combined_msg = self.bridge.cv2_to_imgmsg(combined_image, encoding='bgr8')
            combined_msg.header = msg.header
            self.combined_pub.publish(combined_msg)
            
            # 计算处理时间
            total_time = time.time() - start_time
            self.processing_times.append(total_time)
            
            # 定期输出统计信息
            if self.frame_count % 30 == 0:
                avg_detections = self.detection_count / self.frame_count
                avg_time = np.mean(self.processing_times[-30:]) if len(self.processing_times) > 0 else 0
                fps = 1.0 / avg_time if avg_time > 0 else 0
                
                self.get_logger().info(f"Frame {self.frame_count}: "
                                      f"{detection_count} detections, "
                                      f"{len(all_targets_info)} valid, "
                                      f"FPS: {fps:.1f}, "
                                      f"Process time: {total_time*1000:.1f}ms")

            if self.enable_debug_log:
                if target_to_publish is not None and not is_zero_rectangle:
                    # 计算中心点
                    box_points = target_to_publish['box_points']
                    center_x = np.mean(box_points[:, 0])
                    center_y = np.mean(box_points[:, 1])
                    self.get_logger().debug(f"Published target with center at ({center_x:.1f}, {center_y:.1f})")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    def destroy_node(self):
        """节点销毁时输出统计信息"""
        if self.frame_count > 0:
            avg_detections = self.detection_count / self.frame_count
            avg_time = np.mean(self.processing_times) if len(self.processing_times) > 0 else 0
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            self.get_logger().info(f"Final statistics: {self.frame_count} frames processed, "
                                  f"average {avg_detections:.2f} detections per frame, "
                                  f"average FPS: {fps:.1f}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    try:
        node = YOLOv8SegROSNode()
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