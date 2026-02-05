from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from easyMetrics.core.base import Metric
from .utils import calculate_iou, compute_ap_coco
from .matcher import BaseMatcher, GreedyIoUMatcher

class MeanAveragePrecision(Metric):
    """
    计算目标检测的平均精度 (mAP) 和平均召回率 (AR)。
    与 COCO 评估指标对齐。
    """
    def __init__(self, 
                 iou_thresholds: Optional[List[float]] = None,
                 rec_thresholds: Optional[List[float]] = None,
                 max_detection_thresholds: Optional[List[int]] = None,
                 matcher: Optional[BaseMatcher] = None):
        super().__init__()
        # COCO 默认 IoU 阈值: 0.50 到 0.95，步长 0.05
        self.iou_thresholds = iou_thresholds or np.linspace(0.5, 0.95, 10)
        # COCO 默认召回率阈值: 0 到 1，共 101 个点
        self.rec_thresholds = rec_thresholds or np.linspace(0.0, 1.00, 101)
        # COCO 默认 AR 计算的最大检测数: 1, 10, 100
        self.max_detection_thresholds = max_detection_thresholds or [1, 10, 100]
        # 匹配策略
        self.matcher = matcher or GreedyIoUMatcher()
        
        self.preds = []
        self.targets = []

    def reset(self):
        self.preds = []
        self.targets = []

    def _ensure_numpy_array(self, data, dtype=None):
        """
        确保输入数据是 numpy 数组，如果不是则转换为 numpy 数组。
        
        参数:
            data: 输入数据，可以是列表、numpy 数组、标量或其他可转换类型。
            dtype: 目标数据类型，默认为 None（自动推断）。
            
        返回:
            numpy.ndarray: 转换后的 numpy 数组。
        """
        import numpy as np
        if isinstance(data, np.ndarray):
            if dtype is not None and data.dtype != dtype:
                return data.astype(dtype)
            # 特殊处理：如果是单个框的坐标 [x1, y1, x2, y2]，转换为 (1, 4) 形状
            if data.ndim == 1 and len(data) == 4:
                return data.reshape(1, 4)
            # 确保返回至少一维数组
            if data.ndim == 0:
                return data.reshape(1)
            return data
        elif data is None:
            return np.array([], dtype=dtype)
        else:
            # 如果输入是标量，转换为一维数组
            try:
                # 尝试迭代，如果是标量会抛出 TypeError
                iter(data)
                arr = np.array(data, dtype=dtype)
                # 特殊处理：如果是单个框的坐标 [x1, y1, x2, y2]，转换为 (1, 4) 形状
                if arr.ndim == 1 and len(arr) == 4:
                    return arr.reshape(1, 4)
                return arr
            except TypeError:
                # 输入是标量，转换为一维数组
                return np.array([data], dtype=dtype)

    def update(self, preds: List[Dict[str, Any]], target: List[Dict[str, Any]]):
        """
        参数:
            preds: 字典列表。每个字典包含 'boxes', 'scores', 'labels'。
            target: 字典列表。每个字典包含 'boxes', 'labels'。
        """
        # 自动转换 preds 中的数据类型
        converted_preds = []
        for pred in preds:
            # 确保 boxes 是二维数组
            boxes = self._ensure_numpy_array(pred['boxes'], dtype=float)
            if boxes.ndim == 1 and len(boxes) == 4:
                boxes = boxes.reshape(1, 4)
            
            # 确保 scores 是一维数组
            scores = self._ensure_numpy_array(pred['scores'], dtype=float)
            if scores.ndim == 0:
                scores = scores.reshape(1)
            
            # 确保 labels 是一维数组
            labels = self._ensure_numpy_array(pred['labels'], dtype=int)
            if labels.ndim == 0:
                labels = labels.reshape(1)
            
            converted_preds.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })
        
        # 自动转换 targets 中的数据类型
        converted_targets = []
        for tgt in target:
            # 确保 boxes 是二维数组
            boxes = self._ensure_numpy_array(tgt['boxes'], dtype=float)
            if boxes.ndim == 1 and len(boxes) == 4:
                boxes = boxes.reshape(1, 4)
            
            # 确保 labels 是一维数组
            labels = self._ensure_numpy_array(tgt['labels'], dtype=int)
            if labels.ndim == 0:
                labels = labels.reshape(1)
            
            converted_targets.append({
                'boxes': boxes,
                'labels': labels
            })
        
        self.preds.extend(converted_preds)
        self.targets.extend(converted_targets)

    def compute(self, n_jobs: int = 1, score_criteria: Optional[List[Tuple[float, float]]] = None, progress: bool = True) -> Dict[str, float]:
        """
        计算 mAP 和 AR 指标。
        
        参数:
            n_jobs: 并行计算的线程数。默认为 1 (串行)。设置为 -1 表示使用所有可用 CPU。
            score_criteria: 可选。计算指定 IoU 和 精度下的最佳置信度阈值。
                            格式为列表: [(iou_thresh, min_precision), ...]
                            例如 [(0.5, 0.9)] 表示寻找 IoU=0.5 时精度至少为 0.9 的最低置信度。
            progress: 是否显示进度条。默认为 True。
            
        返回包含以下键的字典:
            - mAP, mAP_50, mAP_75...
            - BestScore_IoU{iou}_P{prec}_{cls_id}: 满足条件的最佳置信度
        """
        # 1. 识别所有唯一类别
        unique_classes = set()
        for t in self.targets:
            unique_classes.update(t['labels'].tolist())
        for p in self.preds:
            unique_classes.update(p['labels'].tolist())
        
        sorted_classes = sorted(list(unique_classes))
        
        # 定义尺度范围 (COCO 标准)
        area_rngs = {
            'all': (0, 1e10),
            'small': (0, 32 ** 2),
            'medium': (32 ** 2, 96 ** 2),
            'large': (96 ** 2, 1e10)
        }
        
        results = {}
        cls_id_to_idx = {cls_id: i for i, cls_id in enumerate(sorted_classes)}
        
        # 内部函数：计算单个类别的统计信息
        def _process_class(cls_id, area_rng):
            # score_criteria 仅在 'all' 尺度下计算，避免混淆
            criteria = score_criteria if area_name == 'all' else None
            return cls_id, self._compute_class_stats(cls_id, area_rng, criteria)

        # 对每个尺度分别进行评估
        for area_name, area_rng in area_rngs.items():
            # 初始化聚合器
            aps = np.zeros((len(sorted_classes), len(self.iou_thresholds)))
            ars = np.zeros((len(sorted_classes), len(self.iou_thresholds), len(self.max_detection_thresholds)))
            
            # 收集额外的分数指标
            class_score_results = {} 

            # 初始化进度条
            progress_bar = None
            if progress and len(sorted_classes) > 0:
                try:
                    from tqdm import tqdm
                    progress_bar = tqdm(total=len(sorted_classes), desc=f"评估 {area_name}")
                except ImportError:
                    # 如果没有 tqdm，使用简单的打印
                    print(f"开始评估 {area_name}...")

            # 并行或串行处理
            if n_jobs == 1:
                for cls_id in sorted_classes:
                    idx = cls_id_to_idx[cls_id]
                    # score_criteria 仅在 'all' 尺度下计算
                    criteria = score_criteria if area_name == 'all' else None
                    cls_stats = self._compute_class_stats(cls_id, area_rng, criteria)
                    
                    # 解包结果
                    if criteria:
                        cls_aps, cls_ars, cls_scores = cls_stats
                        class_score_results[cls_id] = cls_scores
                    else:
                        cls_aps, cls_ars = cls_stats
                        
                    aps[idx, :] = cls_aps
                    ars[idx, :, :] = cls_ars
                    
                    # 更新进度条
                    if progress_bar:
                        progress_bar.update(1)
            else:
                max_workers = None if n_jobs == -1 else n_jobs
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(_process_class, cls_id, area_rng) for cls_id in sorted_classes]
                    for future in futures:
                        cls_id, cls_stats = future.result()
                        idx = cls_id_to_idx[cls_id]
                        
                        # score_criteria 仅在 'all' 尺度下计算
                        criteria = score_criteria if area_name == 'all' else None
                        if criteria:
                            cls_aps, cls_ars, cls_scores = cls_stats
                            class_score_results[cls_id] = cls_scores
                        else:
                            cls_aps, cls_ars = cls_stats
                            
                        aps[idx, :] = cls_aps
                        ars[idx, :, :] = cls_ars
                        
                        # 更新进度条
                        if progress_bar:
                            progress_bar.update(1)
            
            # 关闭进度条
            if progress_bar:
                progress_bar.close()

            # 计算聚合指标
            if aps.size > 0:
                mean_ap = float(np.mean(aps))
                
                if area_name == 'all':
                    results['mAP'] = mean_ap
                    
                    # mAP @ 0.5
                    idx_50 = np.where(np.isclose(self.iou_thresholds, 0.5))[0]
                    if len(idx_50) > 0:
                        results['mAP_50'] = float(np.mean(aps[:, idx_50[0]]))
                        
                    # mAP @ 0.75
                    idx_75 = np.where(np.isclose(self.iou_thresholds, 0.75))[0]
                    if len(idx_75) > 0:
                        results['mAP_75'] = float(np.mean(aps[:, idx_75[0]]))
                        
                    # 特定类别的 AP (仅在 'all' 尺度下报告)
                    for cls_id in sorted_classes:
                        idx = cls_id_to_idx[cls_id]
                        results[f'AP_{cls_id}'] = float(np.mean(aps[idx, :]))
                        
                        # 添加 AP@50 和 AP@75 的类别细分
                        if len(idx_50) > 0:
                            results[f'AP_50_{cls_id}'] = float(aps[idx, idx_50[0]])
                        if len(idx_75) > 0:
                            results[f'AP_75_{cls_id}'] = float(aps[idx, idx_75[0]])
                            
                        # 添加最佳置信度指标
                        if cls_id in class_score_results:
                            for k, v in class_score_results[cls_id].items():
                                results[f'{k}_{cls_id}'] = v
                        
                    # 计算 AR (仅在 'all' 尺度下报告 AR_1, AR_10, AR_100)
                    if ars.size > 0:
                        mean_ars = np.mean(ars, axis=(0, 1)) # [NumMaxDets]
                        for i, max_det in enumerate(self.max_detection_thresholds):
                            results[f'AR_{max_det}'] = float(mean_ars[i])

                elif area_name == 'small':
                    results['mAP_s'] = mean_ap
                elif area_name == 'medium':
                    results['mAP_m'] = mean_ap
                elif area_name == 'large':
                    results['mAP_l'] = mean_ap
            else:
                if area_name == 'all':
                    results['mAP'] = 0.0

        return results

    def _prepare_data(self, cls_id: int, area_rng: Tuple[float, float]) -> Tuple[List[Tuple], Dict, int]:
        """
        准备特定类别和面积范围的数据。
        """
        min_area, max_area = area_rng
        class_preds = [] 
        class_gt = {}    
        n_pos = 0

        # 收集 GT
        for img_idx, target in enumerate(self.targets):
            mask = target['labels'] == cls_id
            boxes = target['boxes'][mask]
            
            # 检查 boxes 是否为空
            if len(boxes) == 0:
                n_pos += 0
                class_gt[img_idx] = {
                    'boxes': np.array([]),
                    'used': np.array([], dtype=bool)
                }
                continue
            
            # 确保 boxes 是二维数组
            if boxes.ndim == 1:
                # 如果是一维数组，说明是空的或者格式不对，跳过
                n_pos += 0
                class_gt[img_idx] = {
                    'boxes': np.array([]),
                    'used': np.array([], dtype=bool)
                }
                continue
            
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            valid_area_mask = (areas >= min_area) & (areas < max_area)
            valid_boxes = boxes[valid_area_mask]
            
            n_pos += len(valid_boxes)
            class_gt[img_idx] = {
                'boxes': valid_boxes,
                'used': np.zeros(len(valid_boxes), dtype=bool)
            }

        # 收集预测值
        limit_dets = max(self.max_detection_thresholds) if self.max_detection_thresholds else 100
        
        for img_idx, pred in enumerate(self.preds):
            mask = pred['labels'] == cls_id
            scores = pred['scores'][mask]
            boxes = pred['boxes'][mask]
            
            # 检查 boxes 是否为空
            if len(boxes) == 0:
                continue
            
            # 确保 boxes 是二维数组
            if boxes.ndim == 1:
                # 如果是一维数组，说明是空的或者格式不对，跳过
                continue
            
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            valid_area_mask = (areas >= min_area) & (areas < max_area)
            
            scores = scores[valid_area_mask]
            boxes = boxes[valid_area_mask]
            
            if len(scores) > 0:
                order = np.argsort(-scores)
                scores = scores[order][:limit_dets]
                boxes = boxes[order][:limit_dets]
                ranks = np.arange(len(scores))
            else:
                scores, boxes, ranks = [], [], []
            
            for s, b, r in zip(scores, boxes, ranks):
                class_preds.append((s, img_idx, b, r))
        
        # 全局按分数排序
        class_preds.sort(key=lambda x: x[0], reverse=True)
        
        return class_preds, class_gt, n_pos

    def _match_predictions(self, class_preds: List[Tuple], class_gt: Dict, pred_gt_ious: List[np.ndarray], iou_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        在特定 IoU 阈值下进行贪婪匹配。
        """
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        
        # 重置 GT 使用状态
        for img_data in class_gt.values():
            img_data['used'][:] = False
            
        for i, ious in enumerate(pred_gt_ious):
            if len(ious) == 0:
                fp[i] = 1
                continue
            
            # 找到最佳匹配
            best_gt_idx = np.argmax(ious)
            best_iou = ious[best_gt_idx]
            
            img_idx = class_preds[i][1]
            
            if best_iou >= iou_thresh:
                if not class_gt[img_idx]['used'][best_gt_idx]:
                    tp[i] = 1
                    class_gt[img_idx]['used'][best_gt_idx] = True
                else:
                    fp[i] = 1 # 重复检测
            else:
                fp[i] = 1
                
        return tp, fp

    def _compute_class_stats(self, cls_id: int, area_rng: Tuple[float, float], 
                             score_criteria: Optional[List[Tuple[float, float]]] = None) -> Any:
        """
        计算特定类别在所有阈值下的 AP 和 Recall。
        如果提供了 score_criteria，则返回 (aps, recalls, scores_dict)
        否则返回 (aps, recalls)
        """
        # 1. 准备数据
        class_preds, class_gt, n_pos = self._prepare_data(cls_id, area_rng)

        num_iou = len(self.iou_thresholds)
        num_dets = len(self.max_detection_thresholds)
        
        # 结果初始化
        aps = np.zeros(num_iou)
        recs = np.zeros((num_iou, num_dets))
        scores_result = {} # key: "BestScore_IoU{}_P{}"

        if n_pos == 0 or len(class_preds) == 0:
            if score_criteria:
                return aps, recs, scores_result
            return aps, recs
        
        # 提取排名以供快速过滤
        pred_ranks = np.array([p[3] for p in class_preds])
        # 提取分数 (class_preds 已经是按分数降序排列)
        pred_scores = np.array([p[0] for p in class_preds])

        # 2. 预计算 IoU
        pred_gt_ious = []
        for _, img_idx, pred_box, _ in class_preds:
            gt_boxes = class_gt[img_idx]['boxes']
            if len(gt_boxes) == 0:
                pred_gt_ious.append(np.array([]))
            else:
                ious = calculate_iou(pred_box[None, :], gt_boxes)[0]
                pred_gt_ious.append(ious)

        # 3. 遍历阈值
        for t_idx, iou_thresh in enumerate(self.iou_thresholds):
            tp, fp = self.matcher.match(class_preds, class_gt, pred_gt_ious, iou_thresh)
            
            # 计算 AP
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            
            recall = cum_tp / n_pos
            precision = cum_tp / (cum_tp + cum_fp)
            
            aps[t_idx] = compute_ap_coco(recall, precision)
            
            # 计算最佳置信度 (如果需要)
            if score_criteria:
                for (crit_iou, crit_prec) in score_criteria:
                    # 使用 isclose 比较浮点数
                    if np.isclose(iou_thresh, crit_iou):
                        # 找到满足 precision >= crit_prec 的所有索引
                        valid_mask = precision >= crit_prec
                        if np.any(valid_mask):
                            # 我们想要最大的 recall，即 valid_mask 中最后一个 True 的位置
                            # 因为 class_preds 是按 score 降序 (recall 升序) 排列的
                            # 最后一个满足条件的点对应最低的 score (但在满足精度的前提下 recall 最大)
                            best_idx = np.where(valid_mask)[0][-1]
                            best_score = pred_scores[best_idx]
                            
                            key = f"BestScore_IoU{crit_iou:.2f}_P{crit_prec:.2f}"
                            scores_result[key] = float(best_score)

            # 计算 AR (针对每个 max_det 阈值)
            for d_idx, max_det in enumerate(self.max_detection_thresholds):
                valid_mask = pred_ranks < max_det
                tp_sum = np.sum(tp[valid_mask])
                recs[t_idx, d_idx] = tp_sum / n_pos

        if score_criteria:
            return aps, recs, scores_result
        return aps, recs

    def _match_predictions(self, *args, **kwargs):
        """
        [已弃用] 请使用 self.matcher.match
        为了兼容性暂时保留，但会抛出错误
        """
        raise DeprecationWarning("Please use self.matcher.match instead of _match_predictions")
