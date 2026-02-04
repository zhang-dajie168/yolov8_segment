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
from .yolov8_segment_model import YOLOv8_Seg, draw_detection, rdk_colors, filter_mask_and_get_min_rectangle

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

        self.get_logger().info(f"Model path: {model_path}")
        self.get_logger().info(f"Confidence threshold: {conf_threshold}")
        self.get_logger().info(f"IoU threshold: {iou_threshold}")
        self.get_logger().info(f"Input topic: {input_topic}")
        self.get_logger().info(f"Max FPS: {max_fps}")
        self.get_logger().info(f"X filter range: {self.x_min_ratio:.2f} - {self.x_max_ratio:.2f}")
        self.get_logger().info(f"Mask alpha: {self.mask_alpha}")
        self.get_logger().info(f"Draw detections: {self.draw_detections}")
        self.get_logger().info(f"Draw min rectangles: {self.draw_min_rectangles}")

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

        # 统计信息
        self.frame_count = 0
        self.detection_count = 0

        self.get_logger().info("YOLOv8 Segmentation ROS Node initialized successfully")

    def publish_rectangle(self, box_points, header, class_id, score):
        """发布矩形顶点"""
        polygon_msg = PolygonStamped()
        polygon_msg.header = header
        
        # 添加类别ID和置信度到消息中（使用header的frame_id字段）
        polygon_msg.header.frame_id = f"class_{class_id}_score_{score:.2f}"
        
        for point in box_points:
            point_msg = Point32()
            point_msg.x = float(point[0])
            point_msg.y = float(point[1])
            point_msg.z = 0.0
            polygon_msg.polygon.points.append(point_msg)
        
        self.rect_pub.publish(polygon_msg)
        self.get_logger().debug(f"Published rectangle for class {class_id} with score {score:.2f}")

    def draw_min_rectangle_on_image(self, cv_image, box_points_original):
        """在图像上绘制最小矩形和顶点"""
        if self.draw_min_rectangles:
            # 在图像上绘制最小矩形
            cv2.drawContours(cv_image, [box_points_original], 0, (0, 255, 0), 2)
            
            # 在图像上标记顶点
            for j, point in enumerate(box_points_original):
                cv2.circle(cv_image, tuple(point), 5, (0, 0, 255), -1)
                cv2.putText(cv_image, f'P{j}', tuple(point), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def image_callback(self, msg):
        """图像回调函数"""
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

            # 准备输入数据
            input_tensor = self.model.bgr2nv12(cv_image)

            # 推理
            outputs = self.model.c2numpy(self.model.forward(input_tensor))

            # 后处理
            ids, scores, bboxes, corpes, mces, protos = self.model.postProcess(outputs)

            # 更新统计信息
            self.detection_count += len(ids)

            # 渲染结果
            self.get_logger().debug(f"Frame {self.frame_count}: Detected {len(ids)} objects")

            # 创建掩码图像
            zeros = np.zeros((160, 160, 3), dtype=np.uint8)

            # 绘制检测框和计算掩码
            for i, (class_id, score, bbox, corp, mc) in enumerate(zip(ids, scores, bboxes, corpes, mces)):
                # 绘制检测框
                x1, y1, x2, y2 = bbox
                self.get_logger().debug(f"Object {i}: ({x1}, {y1}, {x2}, {y2}) -> score: {score:.2f}")
                
                if self.draw_detections:
                    draw_detection(cv_image, (x1, y1, x2, y2), score, class_id)

                # 处理掩码
                x1_corp, y1_corp, x2_corp, y2_corp = corp
                # 确保边界不越界
                x1_corp = max(0, x1_corp)
                y1_corp = max(0, y1_corp)
                x2_corp = min(160, x2_corp)
                y2_corp = min(160, y2_corp)

                if x2_corp > x1_corp and y2_corp > y1_corp:
                    # 计算mask
                    mask_region = protos[y1_corp:y2_corp, x1_corp:x2_corp, :]
                    mask = (np.sum(mc[np.newaxis, np.newaxis, :] * mask_region, axis=2) > 0.0).astype(np.int32)
                    
                    # 获取掩码点的坐标（在160x160坐标系中）
                    mask_points = np.argwhere(mask == 1)  # 形状为 (N, 2)，每个元素是[y, x]
                    
                    if len(mask_points) > 0:
                        # 过滤掩码点并计算最小外接矩形
                        filtered_points, box_points = filter_mask_and_get_min_rectangle(
                            mask_points, x1_corp, y1_corp,
                            x_min_ratio=self.x_min_ratio,
                            x_max_ratio=self.x_max_ratio
                        )
                        
                        if box_points is not None:
                            # 将矩形顶点从160x160坐标系转换到原始图像坐标系
                            scale_x = original_w / 160.0
                            scale_y = original_h / 160.0
                            box_points_original = (box_points * [scale_x, scale_y]).astype(np.int32)
                            
                            # 发布矩形顶点
                            self.publish_rectangle(box_points_original, msg.header, class_id, score)
                            
                            # 在图像上绘制最小矩形
                            self.draw_min_rectangle_on_image(cv_image, box_points_original)
                        
                        # 绘制掩码（仅显示过滤后的点）
                        mask_filtered = np.zeros_like(mask)
                        for point in mask_points:
                            # 检查对应的原始x坐标是否在范围内
                            x_original = point[1] + x1_corp
                            x_min_threshold = int(160 * self.x_min_ratio)
                            x_max_threshold = int(160 * self.x_max_ratio)
                            if x_min_threshold <= x_original <= x_max_threshold:
                                mask_filtered[point[0], point[1]] = 1
                        
                        zeros[y1_corp:y2_corp, x1_corp:x2_corp, :][mask_filtered == 1] = rdk_colors[class_id % 20]

            # 调整掩码大小到原始图像尺寸
            zeros_resized = cv2.resize(zeros, (original_w, original_h), cv2.INTER_LINEAR)

            # 创建组合图像（原图 + alpha倍掩码）
            add_result = np.clip(cv_image + self.mask_alpha * zeros_resized, 0, 255).astype(np.uint8)

            # 发布combined结果
            combined_msg = self.bridge.cv2_to_imgmsg(add_result, encoding='bgr8')
            combined_msg.header = msg.header
            self.combined_pub.publish(combined_msg)

            # 定期输出统计信息
            if self.frame_count % 30 == 0:
                avg_detections = self.detection_count / self.frame_count
                self.get_logger().info(f"Statistics: {self.frame_count} frames processed, "
                                      f"average {avg_detections:.2f} detections per frame")

            self.get_logger().debug(f"Published combined segmentation result for {len(ids)} objects")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    def destroy_node(self):
        """节点销毁时输出统计信息"""
        if self.frame_count > 0:
            avg_detections = self.detection_count / self.frame_count
            self.get_logger().info(f"Final statistics: {self.frame_count} frames processed, "
                                  f"average {avg_detections:.2f} detections per frame")
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