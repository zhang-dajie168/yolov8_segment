# botsort_rdk.py
import numpy as np
import cv2
from collections import deque, OrderedDict
from scipy import linalg
from scipy.optimize import linear_sum_assignment
import os
import time

try:
    from hobot_dnn import pyeasy_dnn as dnn
except ImportError:
    print("警告: hobot_dnn未找到，ReID功能将不可用")


# -------------------- 基础跟踪类 --------------------
class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    _count = 0

    def __init__(self):
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.history = OrderedDict()
        self.features = []
        self.curr_feature = None
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def reset_id():
        BaseTrack._count = 0


# -------------------- 卡尔曼滤波器 --------------------
class KalmanFilterXYWH:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        if mean is None:
            return None, None
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel])
        motion_cov = np.diag(sqr)
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose()
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        if len(mean) == 0:
            return mean, covariance
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance


# -------------------- OSNet ReID模型 --------------------
class OSNetReID:
    def __init__(self, model_path, input_size=(64, 128)):
        self.input_size = input_size
        self.width, self.height = input_size
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        self.nv12_buffer = np.zeros((self.height * 3 // 2, self.width), dtype=np.uint8)
        self.y_plane = self.nv12_buffer[:self.height, :]
        self.uv_plane = self.nv12_buffer[self.height:, :].reshape(self.height // 2, self.width // 2, 2)
        try:
            self.models = dnn.load(model_path)
            self.model = self.models[0]
            print(f"OSNet模型加载成功: {model_path}")
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")

    def bgr2nv12(self, image):
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV)
        np.copyto(self.y_plane, yuv[:, :, 0])
        u_down = cv2.resize(yuv[:, :, 1], (self.width // 2, self.height // 2), interpolation=cv2.INTER_LINEAR)
        v_down = cv2.resize(yuv[:, :, 2], (self.width // 2, self.height // 2), interpolation=cv2.INTER_LINEAR)
        self.uv_plane[:, :, 0] = u_down
        self.uv_plane[:, :, 1] = v_down
        return self.nv12_buffer

    def extract_feature(self, image):
        nv12_data = self.bgr2nv12(image)
        input_tensor = nv12_data[np.newaxis, :]
        outputs = self.model.forward(input_tensor)
        if hasattr(outputs[0], 'buffer'):
            features = outputs[0].buffer
        else:
            features = outputs[0]
        features = np.squeeze(features)
        if features.ndim > 1:
            features = features.flatten()
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features

    def extract_features_batch(self, bboxes, orig_img):
        if len(bboxes) == 0:
            return []
        crops = []
        img_h, img_w = orig_img.shape[:2]
        for box in bboxes:
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)
            if x2 > x1 and y2 > y1:
                crop = orig_img[y1:y2, x1:x2]
                crops.append(crop)
        if not crops:
            return []
        features = []
        for crop in crops:
            try:
                feat = self.extract_feature(crop)
                features.append(feat)
            except Exception as e:
                print(f"特征提取失败: {e}")
                features.append(np.zeros(512, dtype=np.float32))
        return features


# -------------------- 跟踪轨迹类 --------------------
class BOTrack(BaseTrack):
    shared_kalman = KalmanFilterXYWH()

    def __init__(self, xywh, score, cls, feat=None, feat_history=50, det_idx=-1):
        super().__init__()
        self._tlwh = np.asarray(xywh[:4], dtype=np.float32)
        self.kalman_filter = KalmanFilterXYWH()
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[4] if len(xywh) > 4 else -1
        self.det_idx = det_idx

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        if self.mean is None:
            return
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[4] = 0
            mean_state[5] = 0
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def update(self, new_track, frame_id):
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls
        self.det_idx = new_track.det_idx

    def re_activate(self, new_track, frame_id, new_id=False):
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.det_idx = new_track.det_idx

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][4] = 0
                multi_mean[i][5] = 0
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov


# -------------------- 匹配工具 --------------------
def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = [[row_ind[i], col_ind[i]] for i in range(len(row_ind)) if cost_matrix[row_ind[i], col_ind[i]] <= thresh]
    unmatched_a = list(set(range(cost_matrix.shape[0])) - set([m[0] for m in matches]))
    unmatched_b = list(set(range(cost_matrix.shape[1])) - set([m[1] for m in matches]))
    return matches, unmatched_a, unmatched_b


def iou_distance(atracks, btracks):
    """✅ 优化：使用 numpy 向量化计算 IOU，避免纯 Python 双重循环"""
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    
    atlbrs = np.array([track.tlwh for track in atracks], dtype=np.float32)
    btlbrs = np.array([track.tlwh for track in btracks], dtype=np.float32)
    
    # 计算交集
    a_x1 = atlbrs[:, 0:1]
    a_y1 = atlbrs[:, 1:2]
    a_x2 = a_x1 + atlbrs[:, 2:3]
    a_y2 = a_y1 + atlbrs[:, 3:4]
    
    b_x1 = btlbrs[:, 0:1]
    b_y1 = btlbrs[:, 1:2]
    b_x2 = b_x1 + btlbrs[:, 2:3]
    b_y2 = b_y1 + btlbrs[:, 3:4]
    
    inter_x1 = np.maximum(a_x1, b_x1.T)
    inter_y1 = np.maximum(a_y1, b_y1.T)
    inter_x2 = np.minimum(a_x2, b_x2.T)
    inter_y2 = np.minimum(a_y2, b_y2.T)
    
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    a_area = atlbrs[:, 2:3] * atlbrs[:, 3:4]
    b_area = btlbrs[:, 2:3] * btlbrs[:, 3:4]
    union_area = a_area + b_area.T - inter_area
    
    ious = np.where(union_area > 0, inter_area / union_area, 0)
    return 1 - ious


def embedding_distance(tracks, detections, metric='cosine'):
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)
    track_features = np.array([track.smooth_feat for track in tracks if track.smooth_feat is not None])
    det_features = np.array([det.curr_feat for det in detections if det.curr_feat is not None])
    if len(track_features) == 0 or len(det_features) == 0:
        return np.ones((len(tracks), len(detections)), dtype=np.float32)
    if metric == 'cosine':
        similarity = np.dot(track_features, det_features.T)
        return 1 - similarity
    else:
        dists = np.zeros((len(track_features), len(det_features)))
        for i, t_feat in enumerate(track_features):
            for j, d_feat in enumerate(det_features):
                dists[i, j] = np.linalg.norm(t_feat - d_feat)
        return dists


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim


# -------------------- 主跟踪器 --------------------
class BOTSORT:
    def __init__(self, args=None, frame_rate=30):
        self.args = {
            'track_high_thresh': 0.25,
            'track_low_thresh': 0.1,
            'new_track_thresh': 0.25,
            'track_buffer': 30,
            'match_thresh': 0.8,
            'fuse_score': True,
            'gmc_method': 'sparseOptFlow',
            'proximity_thresh': 0.5,
            'appearance_thresh': 0.8,
            'with_reid': False,
            'reid_model_path': '/home/sunrise/Ebike_Human_Follower/src/yolov11_pose_tracking/models/osnet_64x128_nv12.bin'
        }
        if args is not None:
            if isinstance(args, dict):
                self.args.update(args)
            else:
                for key in self.args.keys():
                    if hasattr(args, key):
                        self.args[key] = getattr(args, key)
        self.frame_id = 0
        self.max_time_lost = int(frame_rate / 30.0 * self.args['track_buffer'])
        self.kalman_filter = KalmanFilterXYWH()
        self.encoder = None
        if self.args['with_reid']:
            try:
                self.encoder = OSNetReID(self.args['reid_model_path'])
                print("ReID模型初始化成功")
            except Exception as e:
                print(f"ReID模型初始化失败: {e}")
                self.args['with_reid'] = False
        self.reset()

    def reset(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        BaseTrack.reset_id()

    def init_track(self, detections, img=None):
        if len(detections) == 0:
            return []
        bboxes = []
        scores = []
        classes = []
        det_indices = []
        for i, det in enumerate(detections):
            if len(det) >= 5:
                bboxes.append(det[:4])
                scores.append(det[4])
                classes.append(det[5] if len(det) > 5 else 0)
                det_indices.append(i)
        if len(bboxes) == 0:
            return []
        features = []
        if self.args['with_reid'] and self.encoder is not None and img is not None:
            features = self.encoder.extract_features_batch(bboxes, img)
        tracks = []
        for i, (bbox, score, cls, idx) in enumerate(zip(bboxes, scores, classes, det_indices)):
            feat = features[i] if i < len(features) else None
            tracks.append(BOTrack(bbox, score, cls, feat, det_idx=idx))
        return tracks

    def get_dists(self, tracks, detections):
        dists = iou_distance(tracks, detections)
        dists_mask = dists > (1 - self.args['proximity_thresh'])
        if self.args['fuse_score']:
            dists = fuse_score(dists, detections)
        if self.args['with_reid'] and self.encoder is not None:
            emb_dists = embedding_distance(tracks, detections)
            emb_dists[emb_dists > (1 - self.args['appearance_thresh'])] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)
        return dists

    def update(self, detections, img=None):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        detections = self.init_track(detections, img)

        scores = [det.score for det in detections]
        remain_inds = [i for i, s in enumerate(scores) if s >= self.args['track_high_thresh']]
        inds_low = [i for i, s in enumerate(scores) if s > self.args['track_low_thresh']]
        inds_high = [i for i, s in enumerate(scores) if s < self.args['track_high_thresh']]
        inds_second = [i for i in inds_low if i in inds_high]
        detections_second = [detections[i] for i in inds_second]
        detections = [detections[i] for i in remain_inds]

        strack_pool = self.joint_stracks(self.tracked_stracks, self.lost_stracks)
        BOTrack.multi_predict(strack_pool)

        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = linear_assignment(dists, self.args['match_thresh'])

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, 0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args['new_track_thresh']:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        results = []
        for track in output_stracks:
            x1, y1, w, h = track.tlwh
            results.append([x1, y1, w, h, track.track_id, track.score, track.cls, track.det_idx])
        return results

    @staticmethod
    def joint_stracks(tlista, tlistb):
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]