#!/user/bin/env python
# Copyright (c) 2024，WuChao D-Robotics.
# Licensed under the Apache License, Version 2.0 (the "License");

import cv2
import numpy as np
from scipy.special import softmax
from hobot_dnn import pyeasy_dnn as dnn
from time import time
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(name)s] [%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO_SEG_MODEL")


class BaseModel:
    def __init__(self, model_file: str) -> None:
        try:
            begin_time = time()
            self.quantize_model = dnn.load(model_file)
            logger.debug("Load D-Robotics Quantize model time = %.2f ms" % (
                1000 * (time() - begin_time)))
        except Exception as e:
            logger.error("❌ Failed to load model file: %s" % (model_file))
            logger.error("You can download the model file from the following docs: ./models/download.md")
            logger.error(e)
            exit(1)

        logger.info("-> input tensors")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            logger.info(
                f"intput[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        logger.info("-> output tensors")
        for i, quantize_input in enumerate(self.quantize_model[0].outputs):
            logger.info(
                f"output[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        self.model_input_height, self.model_input_weight = self.quantize_model[0].inputs[0].properties.shape[2:4]

    def resizer(self, img: np.ndarray) -> np.ndarray:
        img_h, img_w = img.shape[0:2]
        self.y_scale, self.x_scale = img_h / self.model_input_height, img_w / self.model_input_weight
        return cv2.resize(img, (self.model_input_height, self.model_input_weight), interpolation=cv2.INTER_LINEAR)

    def bgr2nv12(self, bgr_img: np.ndarray) -> np.ndarray:
        begin_time = time()
        bgr_img = self.resizer(bgr_img)
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        return nv12

    def nv12_resize(self, nv12_img: np.ndarray) -> np.ndarray:
        """
        对 NV12 图像做 resize，输出目标尺寸的 NV12。

        输入 nv12_img 形状: (h * 3 // 2, w)，其中:
          - 前 h 行是 Y 平面
          - 后 h//2 行是 UV 交错平面（每行有 w//2 个 UV 对）

        正确做法：分别提取 Y、U、V 平面，各自 resize，然后重新组装 NV12。
        """
        total_rows = nv12_img.shape[0]
        w = nv12_img.shape[1]
        h = total_rows * 2 // 3

        # 提取 Y 平面 (h, w)
        y_plane = nv12_img[:h, :]

        # 提取 UV 平面 (h//2, w)，每行是 UV 交错
        uv_interleaved = nv12_img[h:, :]  # (h//2, w)

        # 分离 U 和 V：reshape 成 (h//2, w//2, 2)，然后取 [:,:,0] 和 [:,:,1]
        uv_pairs = uv_interleaved.reshape(h // 2, w // 2, 2)
        u_plane = uv_pairs[:, :, 0]  # (h//2, w//2)
        v_plane = uv_pairs[:, :, 1]  # (h//2, w//2)

        th, tw = self.model_input_height, self.model_input_weight

        # resize Y 平面到 (th, tw)
        y_resized = cv2.resize(y_plane, (tw, th), interpolation=cv2.INTER_LINEAR)

        # resize U、V 平面到 (th//2, tw//2)
        u_resized = cv2.resize(u_plane, (tw // 2, th // 2), interpolation=cv2.INTER_LINEAR)
        v_resized = cv2.resize(v_plane, (tw // 2, th // 2), interpolation=cv2.INTER_LINEAR)

        # 重新组装 NV12
        nv12_out = np.empty((th * 3 // 2, tw), dtype=np.uint8)
        nv12_out[:th, :] = y_resized

        # UV 交错：把 U 和 V 重新打包成 (th//2, tw//2, 2) 再 reshape 成 (th//2, tw)
        uv_out = np.empty((th // 2, tw // 2, 2), dtype=np.uint8)
        uv_out[:, :, 0] = u_resized
        uv_out[:, :, 1] = v_resized
        nv12_out[th:, :] = uv_out.reshape(th // 2, tw)

        # 记录缩放比例（用于 bbox 还原）
        self.y_scale = h / th
        self.x_scale = w / tw

        return nv12_out

    def forward(self, input_tensor: np.array) -> list:
        begin_time = time()
        quantize_outputs = self.quantize_model[0].forward(input_tensor)
        return quantize_outputs

    def c2numpy(self, outputs) -> list:
        outputs = [dnnTensor.buffer for dnnTensor in outputs]
        return outputs

    def forward_nv12(self, nv12_img: np.ndarray):
        """
        直接接收 NV12 numpy 数组，resize 后送入模型。
        """
        total_rows = nv12_img.shape[0]
        w = nv12_img.shape[1]
        h = total_rows * 2 // 3

        # 记录原始尺寸缩放比例
        self.y_scale = h / self.model_input_height
        self.x_scale = w / self.model_input_weight

        # resize NV12 到模型输入尺寸
        nv12_resized = self.nv12_resize(nv12_img)

        # 送入模型
        outputs = self.quantize_model[0].forward(nv12_resized)
        return outputs


class YOLOv8_Seg(BaseModel):
    def __init__(self, model_file: str, conf: float, iou: float):
        super().__init__(model_file)

        # 根据输出形状判断是否为单类别
        self.is_single_class = True  # 如果是单类别模型设置为 True

        # 将反量化系数准备好
        self.scales = {}
        for i, output in enumerate(self.quantize_model[0].outputs):
            if hasattr(output.properties, 'scale_data') and output.properties.scale_data is not None:
                self.scales[i] = output.properties.scale_data
                logger.info(f"output[{i}] scale shape: {output.properties.scale_data.shape}")

        if 0 in self.scales:
            self.s_bboxes_scale = self.scales[0][np.newaxis, :]
        else:
            self.s_bboxes_scale = None
        if 2 in self.scales:
            self.s_mces_scale = self.scales[2][np.newaxis, :]
        else:
            self.s_mces_scale = None
        if 3 in self.scales:
            self.m_bboxes_scale = self.scales[3][np.newaxis, :]
        else:
            self.m_bboxes_scale = None
        if 5 in self.scales:
            self.m_mces_scale = self.scales[5][np.newaxis, :]
        else:
            self.m_mces_scale = None
        if 6 in self.scales:
            self.l_bboxes_scale = self.scales[6][np.newaxis, :]
        else:
            self.l_bboxes_scale = None
        if 8 in self.scales:
            self.l_mces_scale = self.scales[8][np.newaxis, :]
        else:
            self.l_mces_scale = None

        logger.info(f"Single class model: {self.is_single_class}")

        # DFL求期望的系数
        self.weights_static = np.array([i for i in range(16)]).astype(np.float32)[np.newaxis, np.newaxis, :]

        # anchors
        self.s_anchor = np.stack([np.tile(np.linspace(0.5, 79.5, 80), reps=80),
                                  np.repeat(np.arange(0.5, 80.5, 1), 80)], axis=0).transpose(1, 0)
        self.m_anchor = np.stack([np.tile(np.linspace(0.5, 39.5, 40), reps=40),
                                  np.repeat(np.arange(0.5, 40.5, 1), 40)], axis=0).transpose(1, 0)
        self.l_anchor = np.stack([np.tile(np.linspace(0.5, 19.5, 20), reps=20),
                                  np.repeat(np.arange(0.5, 20.5, 1), 20)], axis=0).transpose(1, 0)

        self.input_image_size = 640
        self.conf = conf
        self.iou = iou
        self.conf_inverse = -np.log(1 / conf - 1)
        logger.info("iou threshold = %.2f, conf threshold = %.2f" % (iou, conf))
        logger.info("sigmoid_inverse threshold = %.2f" % self.conf_inverse)

        self.x_scale_corp, self.y_scale_corp = 160 / self.model_input_height, 160 / self.model_input_weight

    def postProcess(self, outputs: list) -> tuple:
        begin_time = time()

        s_bboxes = outputs[0].reshape(-1, 64)

        if self.is_single_class:
            s_scores_raw = outputs[1].reshape(-1)
        else:
            s_clses = outputs[1].reshape(-1, 80)

        s_mces = outputs[2].reshape(-1, 32)
        m_bboxes = outputs[3].reshape(-1, 64)

        if self.is_single_class:
            m_scores_raw = outputs[4].reshape(-1)
        else:
            m_clses = outputs[4].reshape(-1, 80)

        m_mces = outputs[5].reshape(-1, 32)
        l_bboxes = outputs[6].reshape(-1, 64)

        if self.is_single_class:
            l_scores_raw = outputs[7].reshape(-1)
        else:
            l_clses = outputs[7].reshape(-1, 80)

        l_mces = outputs[8].reshape(-1, 32)
        protos = outputs[9][0, :, :, :]

        if self.is_single_class:
            s_valid_indices = np.flatnonzero(s_scores_raw >= self.conf_inverse)
            m_valid_indices = np.flatnonzero(m_scores_raw >= self.conf_inverse)
            l_valid_indices = np.flatnonzero(l_scores_raw >= self.conf_inverse)

            s_ids = np.zeros(len(s_valid_indices), dtype=np.int32)
            m_ids = np.zeros(len(m_valid_indices), dtype=np.int32)
            l_ids = np.zeros(len(l_valid_indices), dtype=np.int32)

            s_scores = 1 / (1 + np.exp(-s_scores_raw[s_valid_indices]))
            m_scores = 1 / (1 + np.exp(-m_scores_raw[m_valid_indices]))
            l_scores = 1 / (1 + np.exp(-l_scores_raw[l_valid_indices]))
        else:
            s_max_scores = np.max(s_clses, axis=1)
            s_valid_indices = np.flatnonzero(s_max_scores >= self.conf_inverse)
            s_ids = np.argmax(s_clses[s_valid_indices, :], axis=1)
            s_scores = s_max_scores[s_valid_indices]

            m_max_scores = np.max(m_clses, axis=1)
            m_valid_indices = np.flatnonzero(m_max_scores >= self.conf_inverse)
            m_ids = np.argmax(m_clses[m_valid_indices, :], axis=1)
            m_scores = m_max_scores[m_valid_indices]

            l_max_scores = np.max(l_clses, axis=1)
            l_valid_indices = np.flatnonzero(l_max_scores >= self.conf_inverse)
            l_ids = np.argmax(l_clses[l_valid_indices, :], axis=1)
            l_scores = l_max_scores[l_valid_indices]

            s_scores = 1 / (1 + np.exp(-s_scores))
            m_scores = 1 / (1 + np.exp(-m_scores))
            l_scores = 1 / (1 + np.exp(-l_scores))

        if self.s_bboxes_scale is not None and len(s_valid_indices) > 0:
            s_bboxes_float32 = s_bboxes[s_valid_indices, :].astype(np.float32) * self.s_bboxes_scale
        elif len(s_valid_indices) > 0:
            s_bboxes_float32 = s_bboxes[s_valid_indices, :].astype(np.float32)
        else:
            s_bboxes_float32 = np.zeros((0, 64), dtype=np.float32)

        if self.m_bboxes_scale is not None and len(m_valid_indices) > 0:
            m_bboxes_float32 = m_bboxes[m_valid_indices, :].astype(np.float32) * self.m_bboxes_scale
        elif len(m_valid_indices) > 0:
            m_bboxes_float32 = m_bboxes[m_valid_indices, :].astype(np.float32)
        else:
            m_bboxes_float32 = np.zeros((0, 64), dtype=np.float32)

        if self.l_bboxes_scale is not None and len(l_valid_indices) > 0:
            l_bboxes_float32 = l_bboxes[l_valid_indices, :].astype(np.float32) * self.l_bboxes_scale
        elif len(l_valid_indices) > 0:
            l_bboxes_float32 = l_bboxes[l_valid_indices, :].astype(np.float32)
        else:
            l_bboxes_float32 = np.zeros((0, 64), dtype=np.float32)

        s_dbboxes = self._process_bboxes(s_bboxes_float32, s_valid_indices, self.s_anchor, 8)
        m_dbboxes = self._process_bboxes(m_bboxes_float32, m_valid_indices, self.m_anchor, 16)
        l_dbboxes = self._process_bboxes(l_bboxes_float32, l_valid_indices, self.l_anchor, 32)

        if self.s_mces_scale is not None and len(s_valid_indices) > 0:
            s_mces_float32 = s_mces[s_valid_indices, :].astype(np.float32) * self.s_mces_scale
        elif len(s_valid_indices) > 0:
            s_mces_float32 = s_mces[s_valid_indices, :].astype(np.float32)
        else:
            s_mces_float32 = np.zeros((0, 32), dtype=np.float32)

        if self.m_mces_scale is not None and len(m_valid_indices) > 0:
            m_mces_float32 = m_mces[m_valid_indices, :].astype(np.float32) * self.m_mces_scale
        elif len(m_valid_indices) > 0:
            m_mces_float32 = m_mces[m_valid_indices, :].astype(np.float32)
        else:
            m_mces_float32 = np.zeros((0, 32), dtype=np.float32)

        if self.l_mces_scale is not None and len(l_valid_indices) > 0:
            l_mces_float32 = l_mces[l_valid_indices, :].astype(np.float32) * self.l_mces_scale
        elif len(l_valid_indices) > 0:
            l_mces_float32 = l_mces[l_valid_indices, :].astype(np.float32)
        else:
            l_mces_float32 = np.zeros((0, 32), dtype=np.float32)

        dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes), axis=0) \
            if len(s_dbboxes) + len(m_dbboxes) + len(l_dbboxes) > 0 else np.zeros((0, 4))
        scores = np.concatenate((s_scores, m_scores, l_scores), axis=0) \
            if len(s_scores) + len(m_scores) + len(l_scores) > 0 else np.zeros((0,))
        ids = np.concatenate((s_ids, m_ids, l_ids), axis=0) \
            if len(s_ids) + len(m_ids) + len(l_ids) > 0 else np.zeros((0,), dtype=np.int32)
        mces = np.concatenate((s_mces_float32, m_mces_float32, l_mces_float32), axis=0) \
            if len(s_mces_float32) + len(m_mces_float32) + len(l_mces_float32) > 0 else np.zeros((0, 32), dtype=np.float32)

        if len(dbboxes) > 0:
            indices = cv2.dnn.NMSBoxes(dbboxes.tolist(), scores.tolist(), self.conf, self.iou)
            if len(indices) > 0:
                indices = indices.flatten()
                bboxes = (dbboxes[indices] * np.array(
                    [self.x_scale, self.y_scale, self.x_scale, self.y_scale])).astype(np.int32)
                scores = scores[indices]
                ids = ids[indices]
                corpes = (dbboxes[indices] * np.array(
                    [self.x_scale_corp, self.y_scale_corp, self.x_scale_corp, self.y_scale_corp])).astype(np.int32)
                mces = mces[indices]
            else:
                bboxes, scores, ids, corpes, mces = np.zeros((0, 4)), np.zeros((0,)), np.zeros(
                    (0,), dtype=np.int32), np.zeros((0, 4)), np.zeros((0, 32))
        else:
            bboxes, scores, ids, corpes, mces = np.zeros((0, 4)), np.zeros((0,)), np.zeros(
                (0,), dtype=np.int32), np.zeros((0, 4)), np.zeros((0, 32))

        return ids, scores, bboxes, corpes, mces, protos

    def _process_bboxes(self, bboxes_float32, valid_indices, anchor, scale_factor):
        if len(valid_indices) == 0 or len(bboxes_float32) == 0:
            return np.zeros((0, 4))

        ltrb_indices = np.sum(softmax(bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        anchor_indices = anchor[valid_indices, :]
        x1y1 = anchor_indices - ltrb_indices[:, 0:2]
        x2y2 = anchor_indices + ltrb_indices[:, 2:4]
        dbboxes = np.hstack([x1y1, x2y2]) * scale_factor
        return dbboxes


# 颜色定义
rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72), (23, 204, 146),
    (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
    (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]


def draw_detection(img: np.array,
                   bbox: tuple,
                   score: float,
                   class_id: int,
                   colors: list = rdk_colors) -> None:
    x1, y1, x2, y2 = bbox
    color = colors[class_id % len(colors)]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"object: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def compute_and_process_mask(protos, mc, bbox, corp,
                             x_min_ratio=1/6, x_max_ratio=5/6,
                             original_h=None, original_w=None):
    x1_corp, y1_corp, x2_corp, y2_corp = corp
    x1, y1, x2, y2 = bbox

    x1_corp = max(0, x1_corp)
    y1_corp = max(0, y1_corp)
    x2_corp = min(160, x2_corp)
    y2_corp = min(160, y2_corp)

    if x2_corp <= x1_corp or y2_corp <= y1_corp:
        return None, None, None

    mask_region = protos[y1_corp:y2_corp, x1_corp:x2_corp, :]
    mask = (np.sum(mc[np.newaxis, np.newaxis, :] * mask_region, axis=2) > 0.0).astype(np.int32)

    mask_points = np.argwhere(mask == 1)

    if len(mask_points) == 0:
        return None, None, None

    original_width_160 = 160
    x_min_threshold = int(original_width_160 * x_min_ratio)
    x_max_threshold = int(original_width_160 * x_max_ratio)

    filtered_points_160 = []
    for point in mask_points:
        y_local, x_local = point
        x_160 = x_local + x1_corp
        y_160 = y_local + y1_corp

        if x_min_threshold <= x_160 <= x_max_threshold:
            if original_h is not None and original_w is not None:
                x_original = int(x_160 * original_w / 160.0)
                y_original = int(y_160 * original_h / 160.0)
            else:
                x_original = int(x1 + x_local * (x2 - x1) / (x2_corp - x1_corp))
                y_original = int(y1 + y_local * (y2 - y1) / (y2_corp - y1_corp))

            filtered_points_160.append([x_original, y_original])

    if len(filtered_points_160) >= 4:
        filtered_points_np = np.array(filtered_points_160, dtype=np.int32)
        rect = cv2.minAreaRect(filtered_points_np)
        box_points = cv2.boxPoints(rect)
        box_points = np.int32(box_points)
        return filtered_points_160, box_points, mask_points

    return None, None, mask_points
